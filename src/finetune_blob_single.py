from dataclasses import dataclass, field

from src.finetune_basic_single import FinetuneSingleModelConfig, FinetunerSingleModel
from src.model_molformer import (
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.model_molbert import (
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
)
from src.model_mole import (
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
)
from src.variational_inference.blob_layer import BLoBConfig, BLoBLoraLinear
from src.variational_model_blob import (
    VariationalMolformerBLoBSingle,
    VariationalMolbertBLoBSingle,
    VariationalMoleBLoBSingle,
)
from src.finetune_mfvi_single import _replace_linear_submodule
from src.training_utils import get_optimizer, count_parameters


@dataclass(kw_only=True)
class FinetuneBLoBModelConfig(FinetuneSingleModelConfig):
    base_blob_config: dict = field(default_factory=dict)
    additional_blob_config: dict = field(default_factory=dict)


class FinetuneBLoBModel(FinetunerSingleModel):
    def __init__(self, config: FinetuneBLoBModelConfig):
        blob_config_dict = {**config.base_blob_config, **config.additional_blob_config}
        self.blob_config = BLoBConfig(**blob_config_dict)

        super().__init__(config)

        self.config = config

        self.add_variational_layers()  # variational layers added AFTER PEFT

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
            mfvi_config=self.blob_config,
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

        self.finetune_model = VariationalMolbertBLoBSingle(
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
            mfvi_config=self.blob_config,
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

        self.finetune_model = VariationalMoleBLoBSingle(
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
            mfvi_config=self.blob_config,
            train_dataset_size=len(self.train_dataset),
        )

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


def get_blob_lora_target_modules(target_modules: list[str]) -> list[str]:
    return [
        f"base_model.model.{m}.lora_A.default" for m in target_modules
    ]


def finetune_blob_model(**kwargs):
    config = FinetuneBLoBModelConfig(**kwargs)

    finetuner = FinetuneBLoBModel(config)

    finetuner.train()

    finetuner.set_batch_size(100)

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    finetuner.remove_all_checkpoints_but_final()

    return results


def count_blob_model(**kwargs):
    config = FinetuneBLoBModelConfig(**kwargs)

    finetuner = FinetuneBLoBModel(config)

    param_count = count_parameters(finetuner.finetune_model.model)

    return param_count


def reevaluate_blob_model(**kwargs):
    config = FinetuneBLoBModelConfig(**kwargs)

    finetuner = FinetuneBLoBModel(config)

    finetuner.finetune_model.restore()

    results = finetuner.evaluate(metric_string_list=config.metric_string_list)

    return results
