from typing import Optional, Iterable, Callable, Dict, Literal
from dataclasses import dataclass, field
import numpy as np
import json
import os

import torch
import torch.nn as nn

from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from deepchem.data import Dataset
from deepchem.utils.typing import OneOrMany

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from src.dataset_tasks import get_dataset_task
from src.utils import (
    tensor_to_numpy,
    set_seed,
    create_file_path_string,
    load_yaml_config,
)
from src.training_utils import get_finetuning_datasets, get_optimizer
from src.deepchem_hf_models import HuggingFaceModel
from src.model_molformer import (
    MolformerDeepchem,
    MolformerConfig,
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.model_molbert import (
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
    MolbertDeepchem,
    MolbertConfig,
)
from src.model_mole import (
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
    MolEDeepchem,
    MolEExtraConfig,
)

from src.uncertainty_quantification_regression import (
    UncertaintyQuantificationRegressionHF,
    UncertaintyRegressionPredictionOutput,
)

from src.uncertainty_quantification import (
    UncertaintyQuantificationBaseConfig,
    UncertaintyQuantificationBase,
)
from src.uncertainty_quantification_basic_single import (
    UncertaintyQuantificationBasicSingleConfig,
    UncertaintyQuantificationBasicSingle,
)
from src.uncertainty_quantification_regression import (
    UncertaintyQuantificationRegressionHF,
)
from src.mole.deberta.ops import StableDropout

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class UncertaintyQuantificationMCDropoutConfig(
    UncertaintyQuantificationBasicSingleConfig
):
    dropout_rate: float = 0.1
    num_mc_samples: int = 50


class UncertaintyQuantificationMCDropoutSingle(UncertaintyQuantificationBase):
    def __init__(self, config: UncertaintyQuantificationMCDropoutConfig):
        super().__init__(config)

        self.config: UncertaintyQuantificationMCDropoutConfig

    def _load_finetune_model(self):
        if self.config.model_type == "molformer":
            self._load_molformer_model()
        elif self.config.model_type == "molbert":
            self._load_molbert_model()
        elif self.config.model_type == "mole":
            self._load_mole_model()
        else:
            raise ValueError(f"Unrecognised model type: {self.config.model_type}")

        self._add_finetune_method()

        self.finetune_model.restore()

    def _instantiate_molformer(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolformerForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolformerForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

        self.finetune_model = MolformerDeepchem(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            load_path=self.molformer_load_path,
            from_pretrained_molformer=self.from_pretrained_molformer,
            n_tasks=self.config.n_tasks,
            config=self.molformer_config_dict,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
        )

    def _load_molformer_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.molformer_config = MolformerConfig(**pretrained_model_config)
        self.molformer_config_dict = pretrained_model_config

        self.molformer_load_path = os.path.join(
            create_file_path_string(
                ["pretrained_molformer", "pytorch_checkpoints"], local_path=True
            ),
            "N-Step-Checkpoint_3_30000.ckpt",
        )
        self.from_pretrained_molformer = True

        self._instantiate_molformer()

    def _instantiate_molbert(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolbertForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolbertForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

        self.finetune_model = MolbertDeepchem(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.molbert_config_dict,
            batch_size=self.config.batch_size,
            sequence_classifier=sequence_classifier,
            num_labels=self.config.num_labels,
        )

    def _load_molbert_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.molbert_config = MolbertConfig(**pretrained_model_config)
        self.molbert_config_dict = pretrained_model_config

        self._instantiate_molbert()

    def _instantiate_mole(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolEForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolEForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

        self.finetune_model = MolEDeepchem(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.mole_config_dict,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
        )

    def _load_mole_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.mole_config_dict = pretrained_model_config
        self.mole_config = MolEExtraConfig(**pretrained_model_config)

        self._instantiate_mole()

    def _lora_method(self):
        task_type_string = self.config.base_lora_config.get("task_type", "SEQ_CLS")
        if task_type_string == "SEQ_CLS":
            task_type = TaskType.SEQ_CLS
        elif task_type_string == "SEQ_2_SEQ_LM":
            task_type = TaskType.SEQ_2_SEQ_LM
        else:
            raise ValueError("task_type not recognised")

        lora_config = {
            **self.config.base_lora_config,
            **self.config.additional_lora_config,
            "task_type": task_type,
        }
        peft_config = LoraConfig(**lora_config)

        self.finetune_model.model = get_peft_model(
            self.finetune_model.model, peft_config
        )

    def _classifier_only_method(self):
        pass

    def _freeze_early_layers_method(self):
        pass

    def _full_finetune_method(self):
        pass

    def _add_finetune_method(self):
        if self.config.finetune_type == "lora":
            self._lora_method()
            
    @torch.no_grad()
    def predict_mean_and_std_on_batch(
        self, inputs, compute_std: bool = False, **kwargs
    ):
        self.finetune_model.model.eval()
        self.enable_dropout(self.config.dropout_rate)

        if self.config.sequence_classifier_type == "default":
            if self.config.num_mc_samples == 1:
                if compute_std:
                    raise ValueError("Cannot compute std with sample size 1")
                else:
                    return UncertaintyRegressionPredictionOutput(
                        mean=self.finetune_model.model(**inputs).get("logits"), std=None
                    )

            model_prediction_mean = []

            for _ in range(self.config.num_mc_samples):
                model_prediction_mean.append(
                    self.finetune_model.model(**inputs).get("logits")
                )

            model_prediction_mean = torch.hstack(model_prediction_mean)

            ensemble_mean = torch.mean(model_prediction_mean, dim=1)

            ensemble_std = None

            if compute_std:
                ensemble_mean_difference_squared = torch.pow(
                    model_prediction_mean.T - ensemble_mean, 2
                )

                ensemble_sample_variance = torch.sum(
                    ensemble_mean_difference_squared, dim=0
                )

                ensemble_sample_variance = torch.div(
                    ensemble_sample_variance,
                    self.config.num_mc_samples - 1,
                )

                ensemble_std = torch.sqrt(ensemble_sample_variance)

            return UncertaintyRegressionPredictionOutput(
                mean=ensemble_mean, std=ensemble_std
            )

        elif self.config.sequence_classifier_type == "likelihood":
            if self.config.num_mc_samples == 1:
                outputs = self.finetune_model.model(**inputs)
                return UncertaintyRegressionPredictionOutput(
                    mean=outputs.get("logits"), std=outputs.get("std_logits")
                )

            model_prediction_mean = []
            model_prediction_std = []

            for _ in range(self.config.num_mc_samples):
                model_output = self.finetune_model.model(**inputs)
                model_prediction_mean.append(model_output.get("logits").unsqueeze(1))
                model_prediction_std.append(model_output.get("std_logits").unsqueeze(1))

            model_prediction_mean = torch.hstack(model_prediction_mean)
            model_prediction_std = torch.hstack(model_prediction_std)

            ensemble_mean = torch.mean(model_prediction_mean, dim=1)
            mean_predicted_var = torch.mean(model_prediction_std.pow(2), dim=1)

            ensemble_std = None

            if compute_std:
                ensemble_mean_difference_squared = torch.pow(
                    model_prediction_mean.T - ensemble_mean, 2
                )

                ensemble_mean_sample_variance = torch.sum(
                    ensemble_mean_difference_squared, dim=0
                )

                ensemble_mean_sample_variance = torch.div(
                    ensemble_mean_sample_variance,
                    self.config.num_mc_samples - 1,
                )

                ensemble_std = torch.sqrt(
                    ensemble_mean_sample_variance + mean_predicted_var
                )

            return UncertaintyRegressionPredictionOutput(
                mean=ensemble_mean, std=ensemble_std
            )

        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

    def enable_dropout(self, dropout_rate: float):
        """
        Enable dropout in the module by setting the dropout layers to training mode.
        """
        for module in self.finetune_model.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = dropout_rate
            if isinstance(module, StableDropout):
                module.train()
                module.drop_prob = dropout_rate


class UncertaintyQuantificationMCDropoutRegression(
    UncertaintyQuantificationMCDropoutSingle, UncertaintyQuantificationRegressionHF
):
    pass


def uncertainty_quantification_mc_dropout_single(
    metric_string_list: list[str] = ["ece"],
    problem_type="regression",
    dataset_type: Literal["test", "val"] = "test",
    **kwargs,
):
    config = UncertaintyQuantificationMCDropoutConfig(**kwargs)

    if problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationMCDropoutRegression(config)
        scores = {}
        if "ece" in metric_string_list:
            ece = uncertainty_quantifier.regression_expected_calibration_error(
                dataset=(
                    uncertainty_quantifier.test_dataset
                    if dataset_type == "test"
                    else uncertainty_quantifier.val_dataset
                )
            )
            scores.update({"ece_score": ece})
        if "nll" in metric_string_list:
            nll = uncertainty_quantifier.gaussian_negative_log_likelihood(
                dataset=(
                    uncertainty_quantifier.test_dataset
                    if dataset_type == "test"
                    else uncertainty_quantifier.val_dataset
                )
            )
            scores.update({"nll_score": nll})
        if "rms" in metric_string_list:
            rms = uncertainty_quantifier.root_mean_squared_error(
                dataset=(
                    uncertainty_quantifier.test_dataset
                    if dataset_type == "test"
                    else uncertainty_quantifier.val_dataset
                )
            )
            scores.update({"rms_score": rms})

        return scores
    else:
        raise ValueError(f"Problem type {problem_type} not recognised.")
