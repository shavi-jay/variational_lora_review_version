import os
from dataclasses import dataclass, field, asdict
from functools import partial
import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any

# import args    # from molformer/finetune

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.transformers import TransformerEncoder
from fast_transformers.masking import LengthMask as LM

from src.molformer.molformer_rotate_builder import (
    RotateEncoderBuilder as rotate_builder,
)
from src.molformer.molformer_tokenizer import MolTranBertTokenizer
from src.deepchem_hf_models import HuggingFaceModel

from src.utils import load_yaml_config, create_file_path_string, path_to_local_data
from src.training_utils import get_activation_function
from src.likelihood_model import (
    forward_loss,
    LikelihoodSequenceClassifierOutput,
    RobertaLikelihoodClassificationHeadCustomActivation,
    RobertaLikelihoodClassificationHeadCustomActivationCalibrated,
    MolformerLikelihoodClassificationHead,
    MolformerLikelihoodClassificationHeadCalibrated,
)
from src.basic_classifier_heads import (
    RobertaClassificationHeadCustomActivation,
    MolformerClassificationHead,
    ClassifierHeadConfig,
)


@dataclass
class MolformerConfig(ClassifierHeadConfig):
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    num_feats: int = 32
    embedding_dropout_prob: float = 0.2
    n_vocab: int = 2362
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_head_type: str = "molformer"
    num_labels: int = 1
    problem_type: str | None = "regression"
    use_return_dict: bool = (True,)

    # peft BaseTuner performs to_dict() on the config object
    def to_dict(self):
        return asdict(self)


class MolformerModel(nn.Module):
    def __init__(self, config: MolformerConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(self.config.n_vocab, self.config.hidden_size)
        self.embedding_dropout = nn.Dropout(self.config.embedding_dropout_prob)
        self.create_encoder()
        self.to(self.config.device)

    def create_encoder(self):
        builder = rotate_builder.from_kwargs(
            n_layers=self.config.n_layer,
            n_heads=self.config.n_head,
            query_dimensions=self.config.hidden_size // self.config.n_head,
            value_dimensions=self.config.hidden_size // self.config.n_head,
            feed_forward_dimensions=self.config.hidden_size,
            attention_type="linear",
            feature_map=partial(
                GeneralizedRandomFeatures, n_dims=self.config.num_feats
            ),
            activation="gelu",
        )
        self.encoder: TransformerEncoder = builder.get()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        token_embeddings = self.embeddings(
            input_ids
        )  # each index maps to a (learnable) vector
        embedding_output = self.embedding_dropout(token_embeddings)

        encoder_output = self.encoder(
            embedding_output, length_mask=LM(attention_mask.sum(-1))
        )

        # average pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(encoder_output.size()).float()
        )
        sum_embeddings = torch.sum(encoder_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=encoder_output[0],
            pooler_output=pooled_output,
        )

    def load_from_pretrained(
        self,
        load_path: str,
        from_pretrained_molformer: bool = True,
    ):
        data = torch.load(load_path)
        if from_pretrained_molformer:
            self.embeddings.load_state_dict(data["tok_emb"])
            self.encoder.load_state_dict(data["blocks"])
        else:
            self.load_state_dict(data["model"])


class MolformerPretrainedModel(nn.Module):
    def __init__(self, config: MolformerConfig):
        super().__init__()
        self.config = config
        self.molformer = MolformerModel(config)

    def post_init(self):
        self.to(self.config.device)
        self.num_labels = self.config.num_labels

    def load_from_pretrained(
        self,
        load_path: str,
        from_pretrained_molformer: bool = True,
    ):
        data = torch.load(load_path)
        if from_pretrained_molformer:
            self.molformer.load_from_pretrained(load_path)
        else:
            self.load_state_dict(data["model"])

        self.to(self.config.device)


class MolformerForSequenceClassification(MolformerPretrainedModel):
    def __init__(self, config: MolformerConfig):
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.molformer(
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


class MolformerForSequenceClassificationLikelihoodLoss(MolformerPretrainedModel):
    def __init__(self, config: MolformerConfig):
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.molformer(
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


DEFAULT_MOLFORMER_PATH = path_to_local_data(
    "pretrained_molformer/pytorch_checkpoints/N-Step-Checkpoint_3_30000.ckpt"
)


class MolformerDeepchem(HuggingFaceModel):
    def __init__(
        self,
        task: str,
        load_path: str = DEFAULT_MOLFORMER_PATH,
        n_tasks: int = 1,
        num_labels: Optional[int] = None,
        config: Dict[Any, Any] = {},
        from_pretrained_molformer: bool = True,
        sequence_classifier: MolformerPretrainedModel = MolformerForSequenceClassification,
        **kwargs,
    ):
        self.n_tasks = n_tasks

        vocab_file_path = os.path.join(
            create_file_path_string(["molformer"]), "bert_vocab.txt"
        )

        tokenizer = MolTranBertTokenizer(vocab_file_path)

        molformer_config = MolformerConfig(**config)

        if task == "mtr":
            problem_type = "regression"
            molformer_config.problem_type = "regression"
            num_labels = 1 if num_labels is None else num_labels
            molformer_config.num_labels = num_labels
        elif task == "regression":
            problem_type = "regression"
            molformer_config.problem_type = "regression"
            num_labels = 1 if num_labels is None else num_labels
            molformer_config.num_labels = num_labels
        elif task == "classification":
            problem_type = "single_label_classification"
            molformer_config.problem_type = "single_label_classification"
            num_labels = 2 if num_labels is None else num_labels
            molformer_config.num_labels = num_labels
        elif task == "multi_label_classification":
            problem_type = "multi_label_classification"
            molformer_config.problem_type = "multi_label_classification"
            num_labels = 2 if num_labels is None else num_labels
            molformer_config.num_labels = num_labels
        else:
            raise ValueError(f"Invalid task specification: {task}")

        self.num_labels = num_labels
        model: MolformerPretrainedModel = sequence_classifier(molformer_config)
        model.load_from_pretrained(load_path, from_pretrained_molformer)

        super().__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            n_tasks=num_labels,
            problem_type=problem_type,
            **kwargs,
        )
