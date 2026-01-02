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
)s
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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class UncertaintyQuantificationBasicSingleConfig(UncertaintyQuantificationBaseConfig):
    finetune_type: str = "full_finetune"
    base_lora_config: dict = field(default_factory=dict)
    """Defined in config file"""
    additional_lora_config: dict = field(default_factory=dict)
    """Defined in finetune function"""
    base_pretrained_model_config: dict = field(default_factory=dict)
    """Defined in config file"""
    additional_pretrained_model_config: dict = field(default_factory=dict)
    """Defined in finetune function"""
    batch_size: int = 100
    sequence_classifier_type: Literal["default", "likelihood"] = "default"


class UncertaintyQuantificationBasicSingle(UncertaintyQuantificationBase):
    def __init__(self, config: UncertaintyQuantificationBasicSingleConfig):
        super().__init__(config)

        self.config: UncertaintyQuantificationBasicSingleConfig

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

        print(self.finetune_model.model_dir)

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
        if self.config.finetune_type is "lora":
            self._lora_method()

    def predict_mean_and_std_on_batch(self, inputs, compute_std: bool = False):
        self.finetune_model.model.eval()

        if self.config.sequence_classifier_type == "default":
            raise ValueError(
                "Uncertainty quantification not implemented for default sequence classifier type."
            )

        elif self.config.sequence_classifier_type == "likelihood":
            outputs = self.finetune_model.model(**inputs)
            return UncertaintyRegressionPredictionOutput(
                mean=outputs.get("logits"), std=outputs.get("std_logits")
            )


class UncertaintyQuantificationBasicSingleRegression(
    UncertaintyQuantificationBasicSingle, UncertaintyQuantificationRegressionHF
):
    pass

def uncertainty_quantification_basic_single_model(
    dataset: str,
    metric_string_list: list[str],
    problem_type: str,
    **kwargs,
):

    config = UncertaintyQuantificationBasicSingleConfig(**kwargs, dataset=dataset)
    output = {}

    if problem_type == "classification":
        raise NotImplementedError("Classification not implemented")

    elif problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationBasicSingleRegression(config)

        for metric in metric_string_list:
            result = None
            if metric == "ece":
                result = uncertainty_quantifier.regression_expected_calibration_error(
                    dataset=uncertainty_quantifier.test_dataset
                )
            if metric == "nll":
                result = uncertainty_quantifier.gaussian_negative_log_likelihood(
                    dataset=uncertainty_quantifier.test_dataset
                )
            if result is not None:
                output.update({f"{metric}_score": result})

    return output


def uncertainty_quantification_basic_single_model_calibration_levels(
    metric_string_list: list[str],
    problem_type: str,
    calibration_levels: list[float],
    **kwargs,
):

    config = UncertaintyQuantificationBasicSingleConfig(**kwargs)
    output = {}

    if problem_type == "classification":
        raise NotImplementedError("Classification not implemented")

    elif problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationBasicSingleRegression(config)
        print(
            f"Currrent calibration coeff: {uncertainty_quantifier.finetune_model.model.classifier.calibration_coeff.data}"
        )
        for calibration_coeff in calibration_levels:
            uncertainty_quantifier.finetune_model.model.classifier.calibration_coeff.data = torch.tensor(
                calibration_coeff
            )
            for metric in metric_string_list:
                result = None
                if metric == "ece":
                    result = (
                        uncertainty_quantifier.regression_expected_calibration_error(
                            dataset=uncertainty_quantifier.test_dataset
                        )
                    )
                if metric == "nll":
                    result = uncertainty_quantifier.gaussian_negative_log_likelihood(
                        dataset=uncertainty_quantifier.test_dataset
                    )
                if result is not None:
                    output.update({f"{calibration_coeff}_{metric}_score": result})

    return output
