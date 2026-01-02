from dataclasses import dataclass
import torch.nn as nn

from src.training_utils import get_activation_function

@dataclass
class ClassifierHeadConfig:
    hidden_size: int = 768
    classifier_hidden_size: int = 768
    classifier_dropout_prob: float | None = 0.2
    classifier_activation_func: str = "relu"
    classifier_skip_connection: bool = True
    classifier_hidden_layers: int = 2  # only for molformer classifier head
    hidden_dropout_prob: float = 0.1
    num_labels: int = 1
    problem_type: str | None = "regression"


class RobertaClassificationHeadCustomActivation(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: ClassifierHeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.classifier_hidden_size)
        classifier_dropout = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.activation = get_activation_function(config.classifier_activation_func)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.classifier_hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class MolformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: ClassifierHeadConfig):
        super().__init__()
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
        self.final = nn.Linear(config.classifier_hidden_size, config.num_labels)

    def forward(self, smiles_emb):
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

        return z
