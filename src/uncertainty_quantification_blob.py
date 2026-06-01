from dataclasses import dataclass, field
import torch
import os

from peft import get_peft_model, LoraConfig, TaskType

from src.uncertainty_quantification import (
    UncertaintyQuantificationBaseConfig,
    UncertaintyQuantificationBase,
)
from src.uncertainty_quantification_regression import (
    UncertaintyQuantificationRegressionHF,
    UncertaintyRegressionPredictionOutput,
)
from src.variational_inference.blob_layer import BLoBConfig, BLoBLoraLinear
from src.variational_model_blob import (
    VariationalMolformerBLoBSingle,
    VariationalMolbertBLoBSingle,
    VariationalMoleBLoBSingle,
)
from src.finetune_mfvi_single import _replace_linear_submodule
from src.model_molformer import (
    MolformerConfig,
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
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
from src.utils import create_file_path_string


@dataclass(kw_only=True)
class UncertaintyQuantificationBLoBConfig(UncertaintyQuantificationBaseConfig):
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
    base_blob_config: dict = field(default_factory=dict)
    additional_blob_config: dict = field(default_factory=dict)
    sequence_classifier_type: str = "default"


class UncertaintyQuantificationBLoB(UncertaintyQuantificationBase):

    def _load_finetune_model(self):
        blob_config_dict = {
            **self.config.base_blob_config,
            **self.config.additional_blob_config,
        }
        self.blob_config = BLoBConfig(**blob_config_dict)

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

        if self.blob_config.inference_no_sample:
            self._set_no_sample_mode()

    def _set_no_sample_mode(self):
        for module in self.finetune_model.model.modules():
            if isinstance(module, BLoBLoraLinear):
                module.sampling = False
        self.blob_config.samples_per_prediction = 1

    def _instantiate_molformer(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolformerForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolformerForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )
        self.finetune_model = VariationalMolformerBLoBSingle(
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
            mfvi_config=self.blob_config,
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

        self.finetune_model = VariationalMolbertBLoBSingle(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.molbert_config_dict,
            optimizer=get_optimizer(
                self.config.optimizer_type, self.config.learning_rate
            ),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mfvi_config=self.blob_config,
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

        self.finetune_model = VariationalMoleBLoBSingle(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.mole_config_dict,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
            mfvi_config=self.blob_config,
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
        for target_submodule_name in self.blob_config.mfvi_target_modules:
            _replace_linear_submodule(
                module=self.finetune_model.model,
                submodule_name=target_submodule_name,
                new_layer=BLoBLoraLinear,
                device=self.finetune_model.device,
                prior_std=self.blob_config.prior_std,
                init_eps=self.blob_config.init_eps,
                use_flipout=self.blob_config.use_flipout,
            )

    @torch.no_grad()
    def predict_mean_and_std_on_batch(self, inputs, compute_std: bool = False):
        self.finetune_model.model.eval()

        if self.config.sequence_classifier_type == "default":
            if self.blob_config.samples_per_prediction == 1:
                if compute_std:
                    raise ValueError("Cannot compute std with sample size 1")
                else:
                    return UncertaintyRegressionPredictionOutput(
                        mean=self.finetune_model.model(**inputs).get("logits"), std=None
                    )

            model_prediction_mean = []

            for _ in range(self.blob_config.samples_per_prediction):
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
                    self.blob_config.samples_per_prediction - 1,
                )

                ensemble_std = torch.sqrt(ensemble_sample_variance)

            return UncertaintyRegressionPredictionOutput(
                mean=ensemble_mean, std=ensemble_std
            )

        elif self.config.sequence_classifier_type == "likelihood":
            if self.blob_config.samples_per_prediction == 1:
                outputs = self.finetune_model.model(**inputs)
                return UncertaintyRegressionPredictionOutput(
                    mean=outputs.get("logits"), std=outputs.get("std_logits")
                )

            model_prediction_mean = []
            model_prediction_std = []

            for _ in range(self.blob_config.samples_per_prediction):
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
                    self.blob_config.samples_per_prediction - 1,
                )

                ensemble_std = torch.sqrt(
                    ensemble_mean_sample_variance + mean_predicted_var
                )

            return UncertaintyRegressionPredictionOutput(
                mean=ensemble_mean, std=ensemble_std
            )


class UncertaintyQuantificationBLoBLoraRegression(
    UncertaintyQuantificationBLoB, UncertaintyQuantificationRegressionHF
):
    pass


def uncertainty_quantification_blob_model(
    metric_string_list: list[str] = ["ece"], problem_type="regression", **kwargs
):
    config = UncertaintyQuantificationBLoBConfig(**kwargs)

    if problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationBLoBLoraRegression(config)
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
    else:
        raise ValueError(f"Problem type {problem_type} not recognised.")
