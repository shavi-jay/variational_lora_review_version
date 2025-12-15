import yaml
import torch
import torch.nn as nn
from typing import Any, Optional, Union, Tuple, Dict
from dataclasses import dataclass, asdict
from src.mole.encoder import Encoder
from src.mole.bert import MolEPredictionHead
from src.utils import create_file_path_string, path_to_local_data
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from src.deepchem_hf_models import HuggingFaceModel
from src.likelihood_model import (
    MolELikelihoodPredictionHead,
    MolELikelihoodPredictionHeadCalibrated,
    LikelihoodSequenceClassifierOutput,
    forward_loss,
)
from src.mole.mole_tokenizer import MolETokenizer


@dataclass
class MolEExtraConfig:
    classifier_head_type: str = "mole"
    num_labels: int = 1
    num_tasks: int = 1
    vocab_size_inp: int = 211
    freeze_encoder: bool = False
    problem_type: str = "regression"
    use_return_dict: bool = (False,)  # for compatibility with peft
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self):
        # for compatibility with peft
        return asdict(self)


def deberta_config():
    PATH_LIST_TO_DEBERTA_CONFIG = ["pretrained_mole", "deberta_config.yaml"]
    path = create_file_path_string(PATH_LIST_TO_DEBERTA_CONFIG, local_path=True)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

class MolEForSequenceClassificationBase(nn.Module):
    def __init__(self, config: MolEExtraConfig):
        super().__init__()
        self.config = config
        self.deberta_config = deberta_config()
        self.mole = Encoder(
            self.deberta_config,
            freeze_encoder=config.freeze_encoder,
            vocab_size_inp=config.vocab_size_inp,
            num_tasks=config.num_tasks,
            num_classes=config.num_labels,
        )

        self.post_init()

    def load_from_pretrained(
        self, load_path: str, from_pretrained_molbert: bool = True
    ):
        if from_pretrained_molbert:
            self.mole.load_from_pretrained(load_path)
        else:
            self.load_state_dict(torch.load(load_path)["model_state_dict"])

    def post_init(self):
        self.to(self.config.device)
        self.num_labels = self.config.num_labels


class MolEForSequenceClassification(MolEForSequenceClassificationBase):
    def __init__(self, config: MolEExtraConfig):
        super().__init__(config=config)

        if config.classifier_head_type == "mole":
            self.classifier = MolEPredictionHead(
                config=self.deberta_config,
                num_tasks=1,
                num_labels=self.config.num_labels,
                problem_type=self.config.problem_type,
                dropout=None,
            )
        else:
            raise ValueError(
                f"Incompatible classifier head type: {config.classifier_head_type}"
            )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_mask: Optional[torch.FloatTensor] = None,
        relative_pos: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.mole(
            input_ids,
            input_mask=input_mask,
            relative_pos=relative_pos,
        )

        logits: torch.Tensor = self.classifier(outputs)

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
        )


class MolEForSequenceClassificationLikelihoodLoss(MolEForSequenceClassificationBase):
    def __init__(self, config: MolEExtraConfig):
        super().__init__(config=config)
        if config.classifier_head_type == "mole":
            self.classifier = MolELikelihoodPredictionHead(
                config=self.deberta_config,
                num_tasks=1,
                num_labels=self.config.num_labels,
                problem_type=self.config.problem_type,
                dropout=None,
            )
        elif config.classifier_head_type == "mole-calibrated":
            self.classifier = MolELikelihoodPredictionHeadCalibrated(
                config=self.deberta_config,
                num_tasks=1,
                num_labels=self.config.num_labels,
                problem_type=self.config.problem_type,
                dropout=None,
            )
        else:
            raise ValueError(
                f"Incompatible classifier head type: {config.classifier_head_type}"
            )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_mask: Optional[torch.FloatTensor] = None,
        relative_pos: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.mole(
            input_ids,
            input_mask=input_mask,
            relative_pos=relative_pos,
        )

        if self.config.problem_type == "regression":
            classifier_output = self.classifier(outputs)
            logits: torch.Tensor = classifier_output["logits"]
            std_logits = classifier_output["std_logits"]
        else:
            logits: torch.Tensor = self.classifier(outputs)
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
        )


DEFAULT_MOLE_PATH = path_to_local_data("pretrained_mole/mole_model_state_dict.pth")


class MolEDeepchem(HuggingFaceModel):
    def __init__(
        self,
        task: str,
        load_path: str = DEFAULT_MOLE_PATH,
        n_tasks: int = 1,
        num_labels: Optional[int] = None,
        config: Dict[Any, Any] = {},
        sequence_classifier: MolEForSequenceClassificationBase = MolEForSequenceClassificationLikelihoodLoss,
        **kwargs,
    ):
        self.n_tasks = n_tasks

        mole_config = MolEExtraConfig(**config)
        tokenizer = MolETokenizer()

        if task == "mtr":
            problem_type = "regression"
            mole_config.problem_type = "regression"
            num_labels = 1 if num_labels is None else num_labels
            mole_config.num_labels = num_labels
        elif task == "regression":
            problem_type = "regression"
            mole_config.problem_type = "regression"
            num_labels = 1 if num_labels is None else num_labels
            mole_config.num_labels = num_labels
        elif task == "classification":
            problem_type = "single_label_classification"
            mole_config.problem_type = "single_label_classification"
            num_labels = 2 if num_labels is None else num_labels
            mole_config.num_labels = num_labels
        elif task == "multi_label_classification":
            problem_type = "multi_label_classification"
            mole_config.problem_type = "multi_label_classification"
            num_labels = 2 if num_labels is None else num_labels
            mole_config.num_labels = num_labels
        else:
            raise ValueError(f"Invalid task specification: {task}")

        self.num_labels = num_labels
        model: MolEForSequenceClassificationBase = sequence_classifier(mole_config)
        if load_path is None:
            load_path = DEFAULT_MOLE_PATH
        model.load_from_pretrained(load_path)

        super().__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            n_tasks=num_labels,
            problem_type=problem_type,
            **kwargs,
        )
