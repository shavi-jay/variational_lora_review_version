import numpy as np
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from src.finetune_basic_single import FinetuneSingleModelConfig, FinetunerSingleModel
from src.deepchem_hf_models import HuggingFaceModel
from src.model_molformer import (
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.model_molbert import (
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from src.model_mole import (
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
)
from src.variational_inference.variational_layer import MFVILinear
from src.variational_model import (
    VariationalMolformerSingle,
    VariationalMolbertSingle,
    VariationalMoleSingle,
    MFVIConfig,
)
from src.likelihood_model import (
    MolformerLikelihoodClassificationHeadCalibrated,
    RobertaLikelihoodClassificationHeadCalibrated,
    RobertaLikelihoodClassificationHeadCustomActivationCalibrated,
    MolELikelihoodPredictionHeadCalibrated,
)


from dataclasses import dataclass, field

from src.utils import short_timer
from src.training_utils import get_optimizer, count_parameters

EPS = 1e-5  # for numerical stability


@dataclass(kw_only=True)
class FinetuneMFVIModelConfig(FinetuneSingleModelConfig):
    base_mfvi_config: dict = field(default_factory=dict)
    additional_mfvi_config: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class CalibrateMFVIModelConfig(FinetuneMFVIModelConfig):
    new_finetune_model_dir_list: list[str] | None = None


class FinetuneMFVIModel(FinetunerSingleModel):
    def __init__(self, config: FinetuneMFVIModelConfig):
        mfvi_config_list = {**config.base_mfvi_config, **config.additional_mfvi_config}

        self.mfvi_config = MFVIConfig(**mfvi_config_list)

        super().__init__(config)

        self.config = config

        self.add_variational_layers()  # variational layers need to be added AFTER PEFT

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
            model_dir=self.finetune_model_dir,
            load_path=self.molformer_load_path,
            from_pretrained_molformer=self.from_pretrained_molformer,
            wandb_logger=self.wandb_logger,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            config=self.molformer_config_dict,
            optimizer=get_optimizer(
                self.config.optimizer_type, self.config.learning_rate
            ),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
            mfvi_config=self.mfvi_config,
            train_dataset_size=len(self.train_dataset),
        )

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
            model_dir=self.finetune_model_dir,
            wandb_logger=self.wandb_logger,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            config=self.molbert_config_dict,
            optimizer=get_optimizer(
                self.config.optimizer_type, self.config.learning_rate
            ),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
            mfvi_config=self.mfvi_config,
            train_dataset_size=len(self.train_dataset),
        )

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
            model_dir=self.finetune_model_dir,
            wandb_logger=self.wandb_logger,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            config=self.mole_config_dict,
            optimizer=get_optimizer(
                self.config.optimizer_type, self.config.learning_rate
            ),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
            mfvi_config=self.mfvi_config,
            train_dataset_size=len(self.train_dataset),
        )

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

    def _calibrate_likelihood_head_method(self):
        model_calibrated = False

        for module in self.finetune_model.model.modules():
            if hasattr(module, "mfvi_config"):
                self.finetune_model.model.mfvi_config.beta = 0.0

        for param in self.finetune_model.model.parameters():
            param.requires_grad = False

        for module in self.finetune_model.model.modules():
            if isinstance(
                module,
                (
                    MolformerLikelihoodClassificationHeadCalibrated,
                    RobertaLikelihoodClassificationHeadCalibrated,
                    RobertaLikelihoodClassificationHeadCustomActivationCalibrated,
                    MolELikelihoodPredictionHeadCalibrated,
                ),
            ):
                module.calibration_coeff.requires_grad = True
                model_calibrated = True

        if not model_calibrated:
            raise ValueError("Model does not have a calibrated likelihood head.")


def _replace_linear_submodule(
    module: nn.Module,
    submodule_name: str,
    new_layer: nn.Module,
    device: torch.device,
    **layer_kwargs,
):
    submodules = submodule_name.split(".")
    for name in submodules[:-1]:
        module = getattr(module, name)
    parent_module = module
    submodule_name = submodules[-1]
    old_submodule = getattr(parent_module, submodule_name)
    if isinstance(old_submodule, nn.Linear):
        in_features = old_submodule.in_features
        out_features = old_submodule.out_features
        bias = old_submodule.bias is not None
        new_submodule = new_layer(
            in_features, out_features, bias=bias, **layer_kwargs
        ).to(device)

        setattr(parent_module, submodule_name, new_submodule)


def _replace_all_linear_submodules(
    module: nn.Module,
    new_layer: nn.Module,
    device: torch.device,
    **layer_kwargs,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(
            p.requires_grad for p in child.parameters()
        ):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            new_submodule = new_layer(
                in_features, out_features, bias=bias, **layer_kwargs
            ).to(device)

            setattr(module, name, new_submodule)
        else:
            _replace_all_linear_submodules(
                child, new_layer=new_layer, device=device, **layer_kwargs
            )


def finetune_mfvi_model(**kwargs):
    config = FinetuneMFVIModelConfig(**kwargs)

    finetuner = FinetuneMFVIModel(config)

    finetuner.train()

    finetuner.set_batch_size(100)

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    finetuner.remove_all_checkpoints_but_final()

    return results


def count_mfvi_model(**kwargs):
    config = FinetuneMFVIModelConfig(**kwargs)

    finetuner = FinetuneMFVIModel(config)

    param_count = count_parameters(finetuner.finetune_model.model)

    return param_count


def reevaluate_mfvi_model(**kwargs):
    config = FinetuneMFVIModelConfig(**kwargs)

    finetuner = FinetuneMFVIModel(config)

    finetuner.finetune_model.restore()

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    return results


def calibrate_mfvi_model(**kwargs):
    config = CalibrateMFVIModelConfig(**kwargs)

    finetuner = FinetuneMFVIModel(config)

    finetuner.finetune_model.restore(strict=False, load_optimizer=False)

    finetuner._calibrate_likelihood_head_method()

    if config.new_finetune_model_dir_list is not None:
        finetuner.update_finetune_model_dir(
            new_finetune_model_dir_list=config.new_finetune_model_dir_list
        )
    else:
        ValueError(
            "new_finetune_model_dir_list must be provided to save the calibrated model."
        )

    finetuner.finetune_model.rebuild()

    print("New trainable parameters:")
    print(count_parameters(finetuner.finetune_model.model))

    finetuner.train(dataset_type="val")

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    finetuner.remove_all_checkpoints_but_final()

    return results


def get_mfvi_lora_target_modules(target_modules: list[str]):
    return [
        f"base_model.model.{module_name}.lora_A.default"
        for module_name in target_modules
    ] + [
        f"base_model.model.{module_name}.lora_B.default"
        for module_name in target_modules
    ]
