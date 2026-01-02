from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
)

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.modeling_outputs import ModelOutput

from src.basic_classifier_heads import ClassifierHeadConfig
from src.training_utils import get_activation_function
from src.mole.deberta.ops import StableDropout, ACT2FN

@dataclass
class LikelihoodSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    std_logits: Optional[torch.FloatTensor] = None

def forward_loss(
    logits: torch.FloatTensor,
    std_logits: Optional[torch.FloatTensor],
    labels: Optional[torch.LongTensor],
    problem_type: str = "",
    num_labels: int = 1,
):
    if labels is None:
        raise ValueError("Labels must be provided for loss computation")
    if problem_type == "regression":
        loss_fct = gaussian_negative_log_likelihood
        if num_labels != 1:
            raise ValueError("Must have num_labels = 1")
        else:
            if std_logits is None:
                raise ValueError(
                    "std_logits must be provided for regression likelihood loss computation"
                )
            loss = loss_fct(mean_output=logits, std_output=std_logits, labels=labels)
    elif problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    elif problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    return loss


def gaussian_negative_log_likelihood(
    mean_output: torch.FloatTensor,
    std_output: torch.FloatTensor,
    labels: torch.FloatTensor,
    std_output_positive: bool = True,
    EPS: float = 1e-5,
):

    if std_output_positive:
        std = std_output.squeeze()
    else:
        std = torch.clamp(
            torch.exp(std_output.squeeze()),
            min=torch.Tensor(EPS, device=std_output.device),
        )
    nll = torch.sum(
        0.5 * torch.pow((mean_output.squeeze() - labels.squeeze()) / std, 2)
        + torch.log(std)
    )
    return nll

# MolBERT

class RobertaLikelihoodClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if config.problem_type == "regression":
            self.std_EPS = torch.tensor(1e-5)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels * 2)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if self.config.problem_type == "regression":
            mean = x[:, 0]
            std = torch.clamp(torch.exp(x[:, 1]), min=self.std_EPS.to(x.device))
            return {"logits": mean, "std_logits": std}
        else:
            return x


class RobertaLikelihoodClassificationHeadCalibrated(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if config.problem_type == "regression":
            self.std_EPS = torch.tensor(1e-5)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels * 2)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # Calibration coefficient
        self.calibration_coeff = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if self.config.problem_type == "regression":
            mean = x[:, 0]
            std = torch.clamp(
                torch.exp(x[:, 1] + self.calibration_coeff),
                min=self.std_EPS.to(x.device),
            )
            return {"logits": mean, "std_logits": std}
        else:
            return x


class RobertaLikelihoodClassificationHeadCustomActivation(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: ClassifierHeadConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.activation = get_activation_function(config.classifier_activation_func)
        self.dropout = nn.Dropout(classifier_dropout)
        if config.problem_type == "regression":
            self.std_EPS = torch.tensor(1e-5)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels * 2)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if self.config.problem_type == "regression":
            mean = x[:, 0]
            std = torch.clamp(torch.exp(x[:, 1]), min=self.std_EPS.to(x.device))
            return {"logits": mean, "std_logits": std}
        else:
            return x


class RobertaLikelihoodClassificationHeadCustomActivationCalibrated(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: ClassifierHeadConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.activation = get_activation_function(config.classifier_activation_func)
        self.dropout = nn.Dropout(classifier_dropout)
        if config.problem_type == "regression":
            self.std_EPS = torch.tensor(1e-5)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels * 2)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # Calibration coefficient
        self.calibration_coeff = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if self.config.problem_type == "regression":
            mean = x[:, 0]
            std = torch.clamp(
                torch.exp(x[:, 1] + self.calibration_coeff),
                min=self.std_EPS.to(x.device),
            )
            return {"logits": mean, "std_logits": std}
        else:
            return x


# Molformer

class MolformerLikelihoodClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: ClassifierHeadConfig):
        super().__init__()
        self.config = config
        self.classifier_skip_connection = config.classifier_skip_connection

        self.fcs = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.classifier_hidden_size)]
            + [nn.Linear(config.classifier_hidden_size, config.classifier_hidden_size)]
            * (config.classifier_hidden_layers - 1)
        )
        if config.classifier_dropout_prob is None:
            raise ValueError("classifier_dropout_prob must be specified")
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.classifier_dropout_prob)]
            * config.classifier_hidden_layers
        )
        self.activation = get_activation_function(config.classifier_activation_func)
        if config.problem_type == "regression":
            self.std_EPS = torch.tensor(1e-5)
            self.final = nn.Linear(config.classifier_hidden_size, config.num_labels * 2)
        else:
            self.final = nn.Linear(config.classifier_hidden_size, config.num_labels)

    def forward(self, smiles_emb, **kwargs):
        x_in = smiles_emb

        x_out = None
        
        for fc_layer, dropout in zip(self.fcs, self.dropouts):
            x_out = fc_layer(x_in)
            x_out = dropout(x_out)
            x_out = self.activation(x_out)
            if self.classifier_skip_connection is True:
                x_out = x_out + x_in
            x_in = x_out

        z = self.final(x_out)

        if self.config.problem_type == "regression":
            mean = z[:, 0]
            std = torch.clamp(torch.exp(z[:, 1]), min=self.std_EPS.to(z.device))
            return {"logits": mean, "std_logits": std}
        else:
            return z


class MolformerLikelihoodClassificationHeadCalibrated(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: ClassifierHeadConfig):
        super().__init__()
        self.config = config
        self.classifier_skip_connection = config.classifier_skip_connection

        self.fcs = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.classifier_hidden_size)]
            + [nn.Linear(config.classifier_hidden_size, config.classifier_hidden_size)]
            * (config.classifier_hidden_layers - 1)
        )
        if config.classifier_dropout_prob is None:
            raise ValueError("classifier_dropout_prob must be specified")
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.classifier_dropout_prob)]
            * config.classifier_hidden_layers
        )
        self.activation = get_activation_function(config.classifier_activation_func)
        if config.problem_type == "regression":
            self.std_EPS = torch.tensor(1e-5)
            self.final = nn.Linear(config.classifier_hidden_size, config.num_labels * 2)
        else:
            self.final = nn.Linear(config.classifier_hidden_size, config.num_labels)

        # Calibration coefficient
        self.calibration_coeff = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, smiles_emb, **kwargs):
        x_in = smiles_emb
        
        x_out = None

        for fc_layer, dropout in zip(self.fcs, self.dropouts):
            x_out = fc_layer(x_in)
            x_out = dropout(x_out)
            x_out = self.activation(x_out)
            if self.classifier_skip_connection is True:
                x_out = x_out + x_in
            x_in = x_out

        z = self.final(x_out)

        if self.config.problem_type == "regression":
            mean = z[:, 0]
            std = torch.clamp(
                torch.exp(z[:, 1] + self.calibration_coeff),
                min=self.std_EPS.to(z.device),
            )
            return {"logits": mean, "std_logits": std}
        else:
            return z


# MolE Prediction Head

class MolELikelihoodPredictionHead(nn.Module):
    def __init__(
        self,
        config,
        num_tasks: int = 1,
        num_labels: int = 1,
        problem_type: str = "regression",
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.problem_type = problem_type

        self.dense = torch.nn.Linear(
            self.config["hidden_size"], self.config["hidden_size"]
        )
        if num_tasks > 1:
            raise NotImplementedError(
                "Multi-task not implemented yet for MolELikelihoodPredictionHead"
            )
        self.classifier = torch.nn.Linear(
            self.config["hidden_size"], num_tasks * num_labels * 2
        )
        dropout = self.config["hidden_dropout_prob"] if dropout is None else dropout
        self.dropout = StableDropout(dropout)

    def forward(self, context_token):
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config["hidden_act"]](pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.problem_type == "regression":
            mean = logits[:, 0]
            std = torch.clamp(torch.exp(logits[:, 1]), min=1e-5)
            return {"logits": mean, "std_logits": std}
        else:
            return logits


class MolELikelihoodPredictionHeadCalibrated(nn.Module):
    def __init__(
        self,
        config,
        num_tasks: int = 1,
        num_labels: int = 1,
        problem_type: str = "regression",
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.problem_type = problem_type

        self.dense = torch.nn.Linear(
            self.config["hidden_size"], self.config["hidden_size"]
        )
        if num_tasks > 1:
            raise NotImplementedError(
                "Multi-task not implemented yet for MolELikelihoodPredictionHeadCalibrated"
            )
        self.classifier = torch.nn.Linear(
            self.config["hidden_size"], num_tasks * num_labels * 2
        )
        dropout = self.config["hidden_dropout_prob"] if dropout is None else dropout
        self.dropout = StableDropout(dropout)

        # Calibration coefficient
        self.calibration_coeff = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, context_token):
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config["hidden_act"]](pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.problem_type == "regression":
            mean = logits[:, 0]
            std = torch.clamp(
                torch.exp(logits[:, 1] + self.calibration_coeff), min=1e-5
            )
            return {"logits": mean, "std_logits": std}
        else:
            return logits
