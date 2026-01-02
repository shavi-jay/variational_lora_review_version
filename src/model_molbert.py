import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from src.molbert.base import MolbertPretrainedModel, BertConfigExtras
from src.molbert.tokenizer import MolbertTokenizer
from src.utils import create_file_path_string, path_to_local_data
from src.training_utils import get_activation_function
from src.basic_classifier_heads import (
    RobertaClassificationHeadCustomActivation,
    MolformerClassificationHead,
)
from src.deepchem_hf_models import HuggingFaceModel
from src.likelihood_model import (
    ClassifierHeadConfig,
    LikelihoodSequenceClassifierOutput,
    forward_loss,
    RobertaLikelihoodClassificationHeadCustomActivation,
    RobertaLikelihoodClassificationHeadCustomActivationCalibrated,
    MolformerLikelihoodClassificationHead,
    MolformerLikelihoodClassificationHeadCalibrated,
)


@dataclass
class MolbertConfig(ClassifierHeadConfig):
    vocab_size = 42
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    hidden_act = "gelu"
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    max_position_embeddings = 512
    type_vocab_size = 2
    initializer_range = 0.02
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_head_type: str = "roberta"
    num_labels: int = 1
    problem_type: str | None = "regression"
    max_sequence_length: int = 512
    use_return_dict: bool = (False,)

    # peft BaseTuner performs to_dict() on the config object
    def to_dict(self):
        return asdict(self)


class MolbertForSequenceClassificationBase(nn.Module):
    def __init__(self, config: MolbertConfig):
        super().__init__()
        self.config = config
        config_dict = asdict(config)
        config_dict.pop(
            "use_return_dict", None
        )  # remove use_return_dict from config_dict as it is not in RobertaConfig
        self.bert_config = BertConfigExtras(**config_dict)
        self.molbert = MolbertPretrainedModel(self.bert_config)

        self.post_init()

    def load_from_pretrained(
        self, load_path: str, from_pretrained_molbert: bool = True
    ):
        if from_pretrained_molbert:
            self.molbert.load_from_pretrained(load_path)
        else:
            self.load_state_dict(torch.load(load_path)["model_state_dict"])

    def post_init(self):
        self.to(self.config.device)
        self.num_labels = self.config.num_labels


class MolbertForSequenceClassification(MolbertForSequenceClassificationBase):
    def __init__(self, config: MolbertConfig):
        super().__init__(config=config)
        if config.classifier_head_type == "roberta":
            self.classifier = RobertaClassificationHeadCustomActivation(config)
        elif config.classifier_head_type == "molformer":
            self.classifier = MolformerClassificationHead(config)
        else:
            raise ValueError(
                f"Unrecognised classifier head type: {config.classifier_head_type}"
            )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        valid: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.molbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits: torch.Tensor = self.classifier(outputs["pooler_output"])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MolbertForSequenceClassificationLikelihoodLoss(
    MolbertForSequenceClassificationBase
):
    def __init__(self, config: MolbertConfig):
        super().__init__(config=config)
        if config.classifier_head_type == "roberta":
            self.classifier = RobertaLikelihoodClassificationHeadCustomActivation(
                config
            )
        elif config.classifier_head_type == "roberta-calibrated":
            self.classifier = (
                RobertaLikelihoodClassificationHeadCustomActivationCalibrated(config)
            )
        elif config.classifier_head_type == "molformer":
            self.classifier = MolformerLikelihoodClassificationHead(config)
        elif config.classifier_head_type == "molformer-calibrated":
            self.classifier = MolformerLikelihoodClassificationHeadCalibrated(config)
        else:
            raise ValueError(
                f"Unrecognised classifier head type: {config.classifier_head_type}"
            )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        valid: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.molbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.config.problem_type == "regression":
            classifier_output = self.classifier(outputs["pooler_output"])
            logits: torch.Tensor = classifier_output["logits"]
            std_logits = classifier_output["std_logits"]
        else:
            logits: torch.Tensor = self.classifier(outputs["pooler_output"])
            std_logits = None

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            loss = forward_loss(
                logits=logits,
                std_logits=std_logits,
                labels=labels,
                problem_type=self.config.problem_type,
                num_labels=self.num_labels,
            )

        return LikelihoodSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            std_logits=std_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


DEFAULT_MOLBERT_PATH = path_to_local_data(
    "pretrained_molbert/molbert_100epochs/checkpoints/bert.ckpt"
)


class MolbertDeepchem(HuggingFaceModel):
    def __init__(
        self,
        task: str,
        load_path: str = DEFAULT_MOLBERT_PATH,
        n_tasks: int = 1,
        num_labels: Optional[int] = None,
        config: Dict[Any, Any] = {},
        sequence_classifier: MolbertPretrainedModel = MolbertForSequenceClassification,
        **kwargs,
    ):
        self.n_tasks = n_tasks

        molbert_config = MolbertConfig(**config)
        tokenizer = MolbertTokenizer(
            max_sequence_length=molbert_config.max_sequence_length
        )

        if task == "mtr":
            problem_type = "regression"
            molbert_config.problem_type = "regression"
            num_labels = 1 if num_labels is None else num_labels
            molbert_config.num_labels = num_labels
        elif task == "regression":
            problem_type = "regression"
            molbert_config.problem_type = "regression"
            num_labels = 1 if num_labels is None else num_labels
            molbert_config.num_labels = num_labels
        elif task == "classification":
            problem_type = "single_label_classification"
            molbert_config.problem_type = "single_label_classification"
            num_labels = 2 if num_labels is None else num_labels
            molbert_config.num_labels = num_labels
        elif task == "multi_label_classification":
            problem_type = "multi_label_classification"
            molbert_config.problem_type = "multi_label_classification"
            num_labels = 2 if num_labels is None else num_labels
            molbert_config.num_labels = num_labels
        else:
            raise ValueError(f"Invalid task specification: {task}")

        self.num_labels = num_labels
        model: MolbertForSequenceClassificationBase = sequence_classifier(
            molbert_config
        )
        if load_path is None:
            load_path = DEFAULT_MOLBERT_PATH
        model.load_from_pretrained(load_path)

        super().__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            n_tasks=num_labels,
            problem_type=problem_type,
            **kwargs,
        )
