import os
import tempfile
import json
from typing import Callable, Dict, Literal
from dataclasses import dataclass, field
from typing import Optional
from deepchem_hf_models import HuggingFaceModel

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from src.model_molformer import (
    MolformerDeepchem,
    MolformerConfig,
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.model_molbert import (
    MolbertDeepchem,
    MolbertConfig,
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
)
from src.model_mole import (
    MolEDeepchem,
    MolEExtraConfig,
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
)
from src.utils import set_seed, create_file_path_string, short_timer
from src.training_utils import (
    get_wandb_logger,
    count_parameters,
    get_finetuning_datasets,
    get_metrics,
    get_optimizer,
)
from src.likelihood_model import (
    MolformerLikelihoodClassificationHeadCalibrated,
    RobertaLikelihoodClassificationHeadCalibrated,
    RobertaLikelihoodClassificationHeadCustomActivationCalibrated,
    MolELikelihoodPredictionHeadCalibrated,
)
import torch


@dataclass(kw_only=True)
class FinetuneBaseConfig:
    dataset: str
    task_number: int = 0
    splitter_type: str = "scaffold"
    featurizer_type: str | None = None
    nb_epoch: int = 20
    n_tasks: int = 1
    seed: int | None = None
    is_wandb_logger: bool = False
    wandb_kwargs: dict = field(default_factory=dict)
    is_finetune_model_dir_local: bool = False
    checkpoint_interval: int = 0
    max_checkpoints_to_keep: int = 5
    early_stopping: bool = True
    min_epochs_to_train: int = 0
    model_type: str = "chemberta"


@dataclass(kw_only=True)
class InstantiateBasicModelConfig:
    pretrained_model_dir_list: list[str] | None = None
    finetune_model_dir_list: list[str] | None = None
    sequence_classifier_type: Literal["default", "likelihood"] = "default"
    optimizer_type: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 100
    num_labels: Optional[int] = None


@dataclass(kw_only=True)
class FinetuneSingleModelConfig(FinetuneBaseConfig, InstantiateBasicModelConfig):
    finetune_type: str = "full_finetune"
    base_lora_config: dict = field(default_factory=dict)
    """Defined in config file"""
    additional_lora_config: dict = field(default_factory=dict)
    """Defined in finetune function"""
    base_pretrained_model_config: dict = field(default_factory=dict)
    """Defined in config file"""
    additional_pretrained_model_config: dict = field(default_factory=dict)
    """Defined in finetune function"""
    metric_string_list: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class CalibrateSingleModelConfig(FinetuneSingleModelConfig):
    new_finetune_model_dir_list: list[str] | None = None


def _get_finetune_model_save_dir(
    finetune_model_dir_list: list[str] | None,
    is_finetune_model_dir_local: bool,
    seed: int,
):
    if finetune_model_dir_list is None:
        tempdir = tempfile.mkdtemp()
        finetune_model_dir = os.path.join(tempdir, "finetune-model")
    elif not is_finetune_model_dir_local:
        finetune_model_dir = create_file_path_string(
            ["finetuned_models"] + finetune_model_dir_list + [f"seed_{seed}"],
            create_file_path=True,
        )
    else:
        finetune_model_dir = create_file_path_string(
            ["finetuned_models"] + finetune_model_dir_list + [f"seed_{seed}"],
            create_file_path=True,
            local_path=True,
        )
    return finetune_model_dir


class FinetunerBase:
    """
    Run train() and evaluate() to obtain results
    """

    def __init__(self, config: FinetuneBaseConfig):
        self.config = config

        self._initial_setup()

        self.finetune_model: HuggingFaceModel | None = None

    def _initial_setup(self):
        set_seed(self.config.seed)

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ), self.deepchem_task_type = get_finetuning_datasets(
            self.config.dataset,
            self.config.splitter_type,
            self.config.featurizer_type,
            self.config.task_number,
        )

        self.wandb_logger = get_wandb_logger(
            self.config.is_wandb_logger, self.config.wandb_kwargs
        )

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def save(self, model_save_dir_list: list[str]):
        file_name = f"model_{self.config.seed}"

        save_dir = create_file_path_string(model_save_dir_list, local_path=True)

        save_file = os.path.join(save_dir, file_name)

        self.finetune_model.save_checkpoint(model_dir=save_file)

    def set_batch_size(self, batch_size: int):
        self.finetune_model.batch_size = batch_size


class FinetunerSingleModel(FinetunerBase):
    def __init__(self, config: FinetuneSingleModelConfig):
        super().__init__(config)

        self.config = config

        self._create_pretrained_model()
        self._add_finetune_method()

    @property
    def parameter_count(self):
        return count_parameters(self.finetune_model.model)

    def _create_pretrained_model(self):
        self.finetune_model_dir = _get_finetune_model_save_dir(
            finetune_model_dir_list=self.config.finetune_model_dir_list,
            is_finetune_model_dir_local=self.config.is_finetune_model_dir_local,
            seed=self.config.seed,
        )

        if self.config.model_type == "molformer":
            self._create_molformer_model()
        elif self.config.model_type == "molbert":
            self._create_molbert_model()
        elif self.config.model_type == "mole":
            self._create_mole_model()
        else:
            raise ValueError(f"Unrecognised model type: {self.config.model_type}")

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
        )

    def _create_molformer_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.molformer_config_dict = pretrained_model_config
        self.molformer_config = MolformerConfig(**pretrained_model_config)
        if self.config.pretrained_model_dir_list is None:
            load_path = os.path.join(
                create_file_path_string(
                    ["pretrained_molformer", "pytorch_checkpoints"], local_path=True
                ),
                "N-Step-Checkpoint_3_30000.ckpt",
            )
            self.from_pretrained_molformer = True
        else:
            load_path = create_file_path_string(
                self.config.pretrained_model_dir_list, local_path=True
            )
            self.from_pretrained_molformer = False
        self.molformer_load_path = load_path

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
            model_dir=self.finetune_model_dir,
            load_path=self.molbert_load_path,
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
        )

    def _create_molbert_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.molbert_config_dict = pretrained_model_config
        self.molbert_config = MolbertConfig(**pretrained_model_config)
        if self.config.pretrained_model_dir_list is None:
            load_path = None
        else:
            load_path = create_file_path_string(self.config.pretrained_model_dir_list)
        self.molbert_load_path = load_path

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
        )

    def _create_mole_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.mole_config_dict = pretrained_model_config
        self.mole_config = MolEExtraConfig(**pretrained_model_config)
        if self.config.pretrained_model_dir_list is None:
            load_path = None
        else:
            load_path = create_file_path_string(self.config.pretrained_model_dir_list)
        self.mole_load_path = load_path

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
        if self.config.model_type == "molformer":
            for param in self.finetune_model.model.molformer.parameters():
                param.requires_grad = False
        if self.config.model_type == "molbert":
            for param in self.finetune_model.model.molbert.parameters():
                param.requires_grad = False
        if self.config.model_type == "mole":
            for param in self.finetune_model.model.mole.parameters():
                param.requires_grad = False

    def _freeze_early_layers_method(self):
        if self.config.model_type == "molformer":
            for param in self.finetune_model.model.molformer.embeddings.parameters():
                param.requires_grad = False
            for i in range(self.molformer_config.n_layer - 1):
                for param in self.finetune_model.model.molformer.encoder.layers[
                    i
                ].parameters():
                    param.requires_grad = False
        if self.config.model_type == "molbert":
            for param in self.finetune_model.model.molbert.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(self.molbert_config.num_hidden_layers - 1):
                for param in self.finetune_model.model.molbert.bert.encoder.layer[
                    i
                ].parameters():
                    param.requires_grad = False
        if self.config.model_type == "mole":
            for param in self.finetune_model.model.mole.MolE.embeddings.parameters():
                param.requires_grad = False
            for i in range(
                self.finetune_model.model.mole.MolE.config.num_hidden_layers - 1
            ):
                for param in self.finetune_model.model.mole.MolE.encoder.layer[
                    i
                ].parameters():
                    param.requires_grad = False

    def _full_finetune_method(self):
        pass

    def _calibrate_likelihood_head_method(self):
        model_calibrated = False

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


    def _add_finetune_method(self):
        self.FINETUNE_TYPE_TO_FINETUNE_METHOD_MAPPING: Dict[str, Callable[[], None]] = {
            "lora": self._lora_method,
            "classifier_only": self._classifier_only_method,
            "freeze_early_layers": self._freeze_early_layers_method,
            "full_finetune": self._full_finetune_method,
        }

        if (
            self.config.finetune_type
            in self.FINETUNE_TYPE_TO_FINETUNE_METHOD_MAPPING.keys()
        ):
            self.FINETUNE_TYPE_TO_FINETUNE_METHOD_MAPPING[self.config.finetune_type]()
        else:
            raise ValueError(
                f"Finetune type {self.config.finetune_type} not recognised."
            )

    def train(self, dataset_type: str = "train"):
        if dataset_type not in ["train", "val"]:
            raise ValueError(
                f"dataset_type must be 'train' or 'val', got {dataset_type}"
            )
        with short_timer("Train time"):
            self.finetune_model.fit(
                dataset=(
                    self.train_dataset if dataset_type == "train" else self.val_dataset
                ),
                nb_epoch=self.config.nb_epoch,
                test_dataset=self.val_dataset,
                is_epoch_test_logging=True,
                test_log_frequency=20,
                max_test_batches=None,
                checkpoint_interval=self.config.checkpoint_interval,
                max_checkpoints_to_keep=self.config.max_checkpoints_to_keep,
                early_stopping=self.config.early_stopping,
                min_epochs_to_train=self.config.min_epochs_to_train,
            )

    def train_set_epochs(self, nb_epoch: int):
        with short_timer("Train time"):
            self.finetune_model.fit(
                dataset=self.train_dataset,
                nb_epoch=nb_epoch,
                test_dataset=self.val_dataset,
                is_epoch_test_logging=True,
                test_log_frequency=20,
                max_test_batches=None,
                checkpoint_interval=self.config.checkpoint_interval,
                max_checkpoints_to_keep=self.config.max_checkpoints_to_keep,
                early_stopping=self.config.early_stopping,
                min_epochs_to_train=self.config.min_epochs_to_train,
            )

    def evaluate(self, metric_string_list: list[str] = [], dataset_type="test"):
        if len(metric_string_list) == 0:
            if self.deepchem_task_type == "classification":
                metric_string_list = ["accuracy", "roc_auc"]
            if self.deepchem_task_type == "regression":
                metric_string_list = ["rms"]
        metric_list = get_metrics(metric_string_list)

        self.finetune_model.model.eval()

        if dataset_type == "test":
            eval_results = self.finetune_model.evaluate(
                self.test_dataset, metrics=metric_list
            )
        elif dataset_type == "val":
            eval_results = self.finetune_model.evaluate(
                self.val_dataset, metrics=metric_list
            )
        eval_results.update(self.parameter_count)

        return eval_results

    def remove_all_checkpoints_but_final(self):
        checkpoint_paths = self.finetune_model.get_checkpoints()

        for path in checkpoint_paths:
            if not path.endswith("checkpoint1.pt"):
                if os.path.exists(path):
                    os.remove(path)

    def remove_all_checkpoints(self):
        checkpoint_paths = self.finetune_model.get_checkpoints()

        for path in checkpoint_paths:
            if os.path.exists(path):
                os.remove(path)

    def update_finetune_model_dir(self, new_finetune_model_dir_list: list[str]):
        self.finetune_model_dir = _get_finetune_model_save_dir(
            finetune_model_dir_list=new_finetune_model_dir_list,
            is_finetune_model_dir_local=self.config.is_finetune_model_dir_local,
            seed=self.config.seed,
        )

        self.finetune_model.model_dir = self.finetune_model_dir


def finetune_single_model(**kwargs):
    config = FinetuneSingleModelConfig(**kwargs)

    finetuner = FinetunerSingleModel(config)

    print(finetuner.parameter_count)

    finetuner.train()

    finetuner.set_batch_size(100)

    finetuner.finetune_model.model.eval()

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    finetuner.remove_all_checkpoints_but_final()

    return results


def count_single_model(**kwargs):
    config = FinetuneSingleModelConfig(**kwargs)

    finetuner = FinetunerSingleModel(config)

    parameter_count = finetuner.parameter_count

    return parameter_count


def reevaluate_single_model(**kwargs):
    config = FinetuneSingleModelConfig(**kwargs)

    finetuner = FinetunerSingleModel(config)

    finetuner.finetune_model.restore()

    finetuner.set_batch_size(2)

    finetuner.finetune_model.model.eval()

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    return results


def calibrate_likelihood_head_single_model(**kwargs):
    config = CalibrateSingleModelConfig(**kwargs)

    finetuner = FinetunerSingleModel(config)

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
