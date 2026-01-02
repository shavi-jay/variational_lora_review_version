from typing import Union, Dict, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import json
import os

import numpy as np

from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from deepchem.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType

from transformers.modeling_outputs import SequenceClassifierOutput
from deepchem.utils.typing import OneOrMany

from src.uncertainty_quantification import (
    UncertaintyQuantificationBaseConfig,
    UncertaintyQuantificationBase,
)
from src.uncertainty_quantification_regression import (
    UncertaintyQuantificationRegressionHF,
    UncertaintyRegressionPredictionOutput,
    generate_reliability_plot_regression,
)
from src.variational_inference.variational_layer import MFVILinear
from src.variational_model import (
    MFVIConfig,
    VariationalMolformerSingle,
    VariationalMolbertSingle,
    VariationalMoleSingle,
)

from src.finetune_mfvi_single import (
    _replace_linear_submodule,
    get_mfvi_lora_target_modules,
    _replace_all_linear_submodules,
)

from src.model_molformer import (
    MolformerConfig,
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.likelihood_model import (
    RobertaLikelihoodClassificationHead,
    RobertaLikelihoodClassificationHeadCustomActivation,
    MolformerLikelihoodClassificationHead,
)
from src.model_molbert import (
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
    MolbertConfig,
)
from src.model_mole import (
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
    MolEExtraConfig,
)

from src.training_utils import get_optimizer
from src.utils import create_file_path_string, tensor_to_numpy

from src.dataset_tasks import get_dataset_task

@dataclass(kw_only=True)
class UncertaintyQuantificationVariationalLoraConfig(
    UncertaintyQuantificationBaseConfig
):
    optimizer_type: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 100
    finetune_type: str = "lora"
    base_lora_config: dict = field(default_factory=dict)
    """Defined in config file"""
    additional_lora_config: dict = field(default_factory=dict)
    """Defined in finetune function"""
    base_pretrained_model_config: dict = field(default_factory=dict)
    """Defined in config file"""
    additional_pretrained_model_config: dict = field(default_factory=dict)
    """Defined in finetune function"""
    base_mfvi_config: dict = field(default_factory=dict)
    additional_mfvi_config: dict = field(default_factory=dict)
    sequence_classifier_type: str = "default"


class UncertaintyQuantificationVariational(UncertaintyQuantificationBase):

    def _load_finetune_model(self):
        mfvi_config_list = {
            **self.config.base_mfvi_config,
            **self.config.additional_mfvi_config,
        }

        self.mfvi_config = MFVIConfig(**mfvi_config_list)

        if self.config.model_type == "molformer":
            self._load_molformer_model()
        elif self.config.model_type == "molbert":
            self._load_molbert_model()
        elif self.config.model_type == "mole":
            self._load_mole_model()
        else:
            raise ValueError(f"Unrecognised model type: {self.config.model_type}")
        self._add_finetune_method()

        self.add_variational_layers()

        print("LOAD DIR")
        print(self.load_model_dir)

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
        self.finetune_model = VariationalMolformerSingle(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            load_path=self.molformer_load_path,
            n_tasks=self.config.n_tasks,
            config=self.molformer_config_dict,
            optimizer=get_optimizer(
                self.config.optimizer_type, self.config.learning_rate
            ),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mfvi_config=self.mfvi_config,
            train_dataset_size=None,
            sequence_classifier=sequence_classifier,
            num_labels=self.config.num_labels,
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

        self.finetune_model = VariationalMolbertSingle(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.molbert_config_dict,
            optimizer=get_optimizer(
                self.config.optimizer_type, self.config.learning_rate
            ),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mfvi_config=self.mfvi_config,
            train_dataset_size=None,
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

        self.finetune_model = VariationalMoleSingle(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.mole_config_dict,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
            mfvi_config=self.mfvi_config,
            train_dataset_size=None,
        )

    def _load_mole_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.mole_config = MolEExtraConfig(**pretrained_model_config)
        self.mole_config_dict = pretrained_model_config

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


    def add_variational_layers(self):
        if self.mfvi_config.target_all_modules:
            _replace_all_linear_submodules(
                module=self.finetune_model.model,
                new_layer=MFVILinear,
                device=self.finetune_model.device,
                EPS=self.mfvi_config.eps,
            )
        else:
            for target_submodule_name in self.mfvi_config.mfvi_target_modules:
                _replace_linear_submodule(
                    module=self.finetune_model.model,
                    submodule_name=target_submodule_name,
                    new_layer=MFVILinear,
                    device=self.finetune_model.device,
                    EPS=self.mfvi_config.eps,
                )

    @torch.no_grad()
    def predict_mean_and_std_on_batch(self, inputs, compute_std: bool = False):
        self.finetune_model.model.eval()

        if self.config.sequence_classifier_type == "default":
            if self.finetune_model.mfvi_config.samples_per_prediction == 1:
                if compute_std:
                    raise ValueError("Cannot compute std with sample size 1")
                else:
                    return UncertaintyRegressionPredictionOutput(
                        mean=self.finetune_model.model(**inputs).get("logits"), std=None
                    )

            model_prediction_mean = []

            for _ in range(self.finetune_model.mfvi_config.samples_per_prediction):
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
                    self.finetune_model.mfvi_config.samples_per_prediction - 1,
                )

                ensemble_std = torch.sqrt(ensemble_sample_variance)

            return UncertaintyRegressionPredictionOutput(
                mean=ensemble_mean, std=ensemble_std
            )

        elif self.config.sequence_classifier_type == "likelihood":
            if self.finetune_model.mfvi_config.samples_per_prediction == 1:
                outputs = self.finetune_model.model(**inputs)
                return UncertaintyRegressionPredictionOutput(
                    mean=outputs.get("logits"), std=outputs.get("std_logits")
                )

            model_prediction_mean = []
            model_prediction_std = []

            for _ in range(self.finetune_model.mfvi_config.samples_per_prediction):
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
                    self.finetune_model.mfvi_config.samples_per_prediction - 1,
                )

                ensemble_std = torch.sqrt(
                    ensemble_mean_sample_variance + mean_predicted_var
                )

            return UncertaintyRegressionPredictionOutput(
                mean=ensemble_mean, std=ensemble_std
            )

class UncertaintyQuantificationVariationalLoraRegression(
    UncertaintyQuantificationVariational, UncertaintyQuantificationRegressionHF
):
    pass

def uncertainty_quantification_mfvi_model(
    metric_string_list: list[str] = ["ece"], problem_type="regression", **kwargs
):
    config = UncertaintyQuantificationVariationalLoraConfig(**kwargs)

    if problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationVariationalLoraRegression(
            config
        )
        scores = {}
        if "ece" in metric_string_list:
            ece = uncertainty_quantifier.regression_expected_calibration_error(
                dataset=uncertainty_quantifier.test_dataset
            )
            scores.update({"ece_score": ece})
        if "nll" in metric_string_list:
            nll = uncertainty_quantifier.gaussian_negative_log_likelihood(
                dataset=uncertainty_quantifier.test_dataset
            )
            scores.update({"nll_score": nll})

        return scores
    if problem_type == "classification":
        raise NotImplementedError("Classification not implemented.")