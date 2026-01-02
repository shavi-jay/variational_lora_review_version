from typing import Optional, Iterable, Callable, Dict
from dataclasses import dataclass, field
import numpy as np
import json
import os

import torch
import torch.nn as nn
s
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
    DEFAULT_MOLFORMER_PATH,
)
from src.model_molbert import (
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
    MolbertDeepchem,
    MolbertConfig,
    DEFAULT_MOLBERT_PATH,
)
from src.model_mole import MolEExtraConfig, DEFAULT_MOLE_PATH

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
)

from src.ensemble_models import BasicEnsembleModelConfig, DeepChemBasicEnsembleModel

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(kw_only=True)
class UncertaintyQuantificationBasicEnsembleConfig(
    UncertaintyQuantificationBasicSingleConfig
):
    ensemble_size: int
    ensemble_member_config: dict = field(default_factory=dict)


class UncertaintyQuantificationBasicEnsemble(UncertaintyQuantificationBase):
    def __init__(self, config: UncertaintyQuantificationBasicSingleConfig):
        super().__init__(config)

        self.config: UncertaintyQuantificationBasicEnsembleConfig

    def _load_finetune_model(self):
        if self.config.model_type == "chemberta":
            self._load_chemberta_model()
        elif self.config.model_type == "molformer":
            self._load_molformer_model()
        elif self.config.model_type == "molbert":
            self._load_molbert_model()
        elif self.config.model_type == "mole":
            self._load_mole_model()
        else:
            raise ValueError(f"Unrecognised model type: {self.config.model_type}")

        self._add_finetune_method()

        self.finetune_model.model.set_trainable_parameters()

        self.finetune_model.restore()

        self.finetune_model.model.eval()
        
    def _load_molformer_model(self):
        basic_ensemble_config = BasicEnsembleModelConfig(
            ensemble_model_type="molformer",
            ensemble_member_config=MolformerConfig(
                **self.config.ensemble_member_config
            ),
            ensemble_size=self.config.ensemble_size,
            num_labels=self.config.n_tasks,
            sequence_classifier_type=self.config.sequence_classifier_type,
        )

        self.finetune_model = DeepChemBasicEnsembleModel(
            model_config=basic_ensemble_config,
            task=self.deepchem_task_type,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            model_dir=self.load_model_dir,
        )

        load_path = DEFAULT_MOLFORMER_PATH
        self.finetune_model.load_from_pretrained(
            model_dir=load_path, from_no_finetune=True, from_local_checkpoint=False
        )

    def _load_molbert_model(self):
        basic_ensemble_config = BasicEnsembleModelConfig(
            ensemble_model_type="molbert",
            ensemble_member_config=MolbertConfig(**self.config.ensemble_member_config),
            ensemble_size=self.config.ensemble_size,
            num_labels=self.config.n_tasks,
            sequence_classifier_type=self.config.sequence_classifier_type,
        )

        self.finetune_model = DeepChemBasicEnsembleModel(
            model_config=basic_ensemble_config,
            task=self.deepchem_task_type,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            model_dir=self.load_model_dir,
        )

        load_path = DEFAULT_MOLBERT_PATH
        self.finetune_model.load_from_pretrained(
            model_dir=load_path, from_no_finetune=True, from_local_checkpoint=False
        )

    def _load_mole_model(self):
        basic_ensemble_config = BasicEnsembleModelConfig(
            ensemble_model_type="mole",
            ensemble_member_config=MolEExtraConfig(
                **self.config.ensemble_member_config
            ),
            ensemble_size=self.config.ensemble_size,
            num_labels=self.config.n_tasks,
            sequence_classifier_type=self.config.sequence_classifier_type,
        )

        self.finetune_model = DeepChemBasicEnsembleModel(
            model_config=basic_ensemble_config,
            task=self.deepchem_task_type,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            model_dir=self.load_model_dir,
        )

        load_path = DEFAULT_MOLE_PATH
        self.finetune_model.load_from_pretrained(
            model_dir=load_path, from_no_finetune=True, from_local_checkpoint=False
        )

    def _classifier_only_method(self):
        for (
            member_name,
            pretrained_model,
        ) in self.finetune_model.model.pretrained_models.items():
        pass
    
    def _freeze_early_layers_method(self):
        for (
            member_name,
            pretrained_model,
        ) in self.finetune_model.model.pretrained_models.items():
        pass

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

        for (
            member_name,
            pretrained_model,
        ) in self.finetune_model.model.pretrained_models.items():
            pretrained_model = get_peft_model(pretrained_model, peft_config)

    def _full_finetune_method(self):
        pass

    def _add_finetune_method(self):
        if self.config.finetune_type == "lora":
            self._lora_method()


class UncertaintyQuantificationBasicEnsembleRegression(
    UncertaintyQuantificationBasicEnsemble, UncertaintyQuantificationRegressionHF
):
    pass


def uncertainty_quantification_basic_ensemble_model(
    metric_string_list: list[str],
    problem_type: str,
    **kwargs,
):

    config = UncertaintyQuantificationBasicEnsembleConfig(**kwargs)

    output = {}
    result = None

    if problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationBasicEnsembleRegression(
            config
        )

        for metric in metric_string_list:
            if metric == "ece":
                result = uncertainty_quantifier.regression_expected_calibration_error(
                    dataset=uncertainty_quantifier.test_dataset
                )
            output.update({f"{metric}_score": result})
            if metric == "nll":
                result = uncertainty_quantifier.gaussian_negative_log_likelihood(
                    dataset=uncertainty_quantifier.test_dataset
                )
                output.update({f"{metric}_score": result})
    else:
        raise ValueError(f"Problem type {problem_type} not recognised.")
    
    return output