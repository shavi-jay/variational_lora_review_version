import os

from dataclasses import dataclass, field
from typing import Dict, Callable

from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
import numpy as np

from src.utils import (
    create_file_path_string,
    short_timer,
)
from src.training_utils import (
    get_load_func,
    get_wandb_logger,
    count_parameters,
    get_finetuning_datasets,
    get_metrics,
)
from src.finetune_basic_single import (
    FinetunerBase,
    FinetuneSingleModelConfig,
    _get_finetune_model_save_dir,
)

from src.ensemble_models import BasicEnsembleModelConfig, DeepChemBasicEnsembleModel

from src.model_molformer import MolformerConfig, DEFAULT_MOLFORMER_PATH
from src.model_molbert import MolbertConfig, DEFAULT_MOLBERT_PATH
from src.model_mole import MolEExtraConfig, DEFAULT_MOLE_PATH


@dataclass(kw_only=True)
class FinetuneEnsembleModelConfig(FinetuneSingleModelConfig):
    ensemble_size: int


@dataclass
class FinetuneBasicEnsembleConfig(FinetuneEnsembleModelConfig):
    ensemble_member_config: dict = field(default_factory=dict)


class FinetunerBasicEnsembleModel(FinetunerBase):
    """
    Finetunes ChemBERTa on MoleculeNet for Basic ensemble

    Run train() and evaluate() to obtain results
    """

    def __init__(self, config: FinetuneBasicEnsembleConfig):
        super().__init__(config)

        self.config = config

        self._create_pretrained_model()

        self._add_finetune_method()

        self.finetune_model.model.set_trainable_parameters()

    def _create_pretrained_model(self):
        if self.config.seed is None:
            raise ValueError("Seed must be set for finetuning")
        self.finetune_model_dir = _get_finetune_model_save_dir(
            self.config.finetune_model_dir_list,
            self.config.is_finetune_model_dir_local,
            self.config.seed,
        )

        if self.config.model_type == "molformer":
            self._create_molformer_model()
        elif self.config.model_type == "molbert":
            self._create_molbert_model()
        elif self.config.model_type == "mole":
            self._create_mole_model()
        else:
            raise ValueError(f"Unrecognised model type: {self.config.model_type}")

    def _create_molformer_model(self):
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
            wandb_logger=self.wandb_logger,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            model_dir=self.finetune_model_dir,
            learning_rate=self.config.learning_rate,  # previously 0.0001
        )

        if self.config.pretrained_model_dir_list is None:
            load_path = DEFAULT_MOLFORMER_PATH
            self.finetune_model.load_from_pretrained(
                model_dir=load_path, from_no_finetune=True, from_local_checkpoint=False
            )
        else:
            load_path = create_file_path_string(
                self.config.pretrained_model_dir_list, local_path=True
            )
            self.finetune_model.load_from_pretrained(
                model_dir=load_path, from_no_finetune=False, from_local_checkpoint=True
            )

    def _create_molbert_model(self):
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
            wandb_logger=self.wandb_logger,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            model_dir=self.finetune_model_dir,
            learning_rate=self.config.learning_rate,  # previously 0.0001
        )

        if self.config.pretrained_model_dir_list is None:
            load_path = DEFAULT_MOLBERT_PATH
            self.finetune_model.load_from_pretrained(
                model_dir=load_path, from_no_finetune=True, from_local_checkpoint=False
            )
        else:
            load_path = create_file_path_string(
                self.config.pretrained_model_dir_list, local_path=True
            )
            self.finetune_model.load_from_pretrained(
                model_dir=load_path, from_no_finetune=False, from_local_checkpoint=True
            )

    def _create_mole_model(self):
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
            wandb_logger=self.wandb_logger,
            log_frequency=10,
            n_tasks=self.config.n_tasks,
            model_dir=self.finetune_model_dir,
            learning_rate=self.config.learning_rate,  # previously 0.0001
        )

        if self.config.pretrained_model_dir_list is None:
            load_path = DEFAULT_MOLE_PATH
            self.finetune_model.load_from_pretrained(
                model_dir=load_path, from_no_finetune=True, from_local_checkpoint=False
            )
        else:
            load_path = create_file_path_string(
                self.config.pretrained_model_dir_list, local_path=True
            )
            self.finetune_model.load_from_pretrained(
                model_dir=load_path, from_no_finetune=False, from_local_checkpoint=True
            )

    def _classifier_only_method(self):
        for (
            member_name,
            pretrained_model,
        ) in self.finetune_model.model.pretrained_models.items():
            if self.config.model_type == "molformer":
                for param in pretrained_model.molformer.parameters():
                    param.requires_grad = False
            if self.config.model_type == "molbert":
                for param in pretrained_model.molbert.parameters():
                    param.requires_grad = False
            if self.config.model_type == "mole":
                for param in pretrained_model.mole.parameters():
                    param.requires_grad = False

    def _freeze_early_layers_method(self):
        for (
            member_name,
            pretrained_model,
        ) in self.finetune_model.model.pretrained_models.items():
            if self.config.model_type == "molformer":
                for param in pretrained_model.molformer.embeddings.parameters():
                    param.requires_grad = False
                for i in range(
                    self.finetune_model.model.config.ensemble_member_config.n_layer - 1
                ):
                    for param in pretrained_model.molformer.encoder.layers[
                        i
                    ].parameters():
                        param.requires_grad = False
            if self.config.model_type == "molbert":
                for param in pretrained_model.molbert.bert.embeddings.parameters():
                    param.requires_grad = False
                for i in range(
                    self.finetune_model.model.config.ensemble_member_config.num_hidden_layers
                    - 1
                ):
                    for param in pretrained_model.molbert.bert.encoder.layer[
                        i
                    ].parameters():
                        param.requires_grad = False
            if self.config.model_type == "mole":
                for param in pretrained_model.mole.MolE.embeddings.parameters():
                    param.requires_grad = False
                for i in range(pretrained_model.mole.MolE.config.num_hidden_layers - 1):
                    for param in pretrained_model.mole.MolE.encoder.layer[
                        i
                    ].parameters():
                        param.requires_grad = False

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

    def train(self):
        with short_timer("Train time"):
            self.finetune_model.fit(
                dataset=self.train_dataset,
                nb_epoch=self.config.nb_epoch,
                test_dataset=self.val_dataset,
                is_epoch_test_logging=True,
                test_log_frequency=20,
                max_test_batches=None,
                checkpoint_interval=self.config.checkpoint_interval,
                max_checkpoints_to_keep=self.config.max_checkpoints_to_keep,
                early_stopping=self.config.early_stopping,
            )

    def evaluate(self, metric_string_list: list[str] = ["rms"]):
        self.finetune_model.model.eval()
        with short_timer("Test time"):
            result = self.finetune_model.evaluate(
                test_dataset=self.test_dataset, metric_string_list=metric_string_list
            )
            return result

    def remove_all_checkpoints_but_final(self):
        checkpoint_paths = self.finetune_model.get_checkpoints()

        for path in checkpoint_paths:
            if not path.endswith("checkpoint1.pt"):
                if os.path.exists(path):
                    os.remove(path)

    def check_if_only_final_checkpoint_exists(self):
        checkpoint_paths = self.finetune_model.get_checkpoints()
        if len(checkpoint_paths) == 1:
            if checkpoint_paths[0].endswith("checkpoint1.pt"):
                return True
        return False

    def model_trained_tag(self):
        # Save a file to indicate that the model has been trained
        path = os.path.join(self.finetune_model_dir, "model_trained.txt")
        with open(path, "w") as tag_file:
            tag_file.write("TRAINED")


def finetune_basic_ensemble(
    dataset: str,
    seed: int,
    metric_string_list: list[str] = ["rms"],
    model_save_dir_list: list[str] = None,
    check_if_trained: bool = True,
    **basic_ensemble_finetune_config,
):
    config_dict = {"dataset": dataset, "seed": seed, **basic_ensemble_finetune_config}

    config = FinetuneBasicEnsembleConfig(**config_dict)

    finetuner = FinetunerBasicEnsembleModel(config)

    if check_if_trained:
        if finetuner.check_if_only_final_checkpoint_exists():
            print("Only final checkpoint exists, restoring model.")
            finetuner.finetune_model.restore()
        else:
            finetuner.train()
            finetuner.model_trained_tag()
    else:
        finetuner.train()
        finetuner.model_trained_tag()

    result = finetuner.evaluate(metric_string_list)

    if model_save_dir_list is not None:
        finetuner.save(model_save_dir_list=model_save_dir_list)

    finetuner.remove_all_checkpoints_but_final()

    return result


def reevaluate_basic_ensemble(
    dataset: str,
    seed: int,
    metric_string_list: list[str] = ["rms"],
    model_save_dir_list: list[str] = None,
    **basic_ensemble_finetune_config,
):
    config_dict = {"dataset": dataset, "seed": seed, **basic_ensemble_finetune_config}

    config = FinetuneBasicEnsembleConfig(**config_dict)

    basic_ensemble = FinetunerBasicEnsembleModel(config)

    print("Finetune model directory:")
    print(basic_ensemble.finetune_model.model_dir)

    basic_ensemble.finetune_model.restore()

    result = basic_ensemble.evaluate(metric_string_list)

    return result
