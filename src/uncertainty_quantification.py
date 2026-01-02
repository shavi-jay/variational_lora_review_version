from typing import Optional
from dataclasses import dataclass

from src.deepchem_hf_models import HuggingFaceModel

from src.utils import set_seed, create_file_path_string
from src.training_utils import get_finetuning_datasets


@dataclass(kw_only=True)
class UncertaintyQuantificationBaseConfig:
    dataset: str
    task_number: int = 0
    splitter_type: str = "scaffold"
    featurizer_type: str | None = None
    load_model_dir_list: Optional[list[str]] = None
    seed: int = 0
    n_tasks: int = 1
    num_labels: int = 1
    is_load_model_dir_local: bool = True
    model_type: str = "chemberta"


class UncertaintyQuantificationBase:
    """Performs uncertainty quantification on Uncertainty Models"""

    def __init__(self, config: UncertaintyQuantificationBaseConfig):
        self.config = config

        self._initial_setup()

        self.finetune_model: HuggingFaceModel | None = None

        self._load_finetune_model()

    def _initial_setup(self):
        set_seed(self.config.seed)

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ), self.deepchem_task_type = get_finetuning_datasets(
            dataset=self.config.dataset,
            splitter_type=self.config.splitter_type,
            task_number=self.config.task_number,
        )

        if self.config.load_model_dir_list is None:
            self.load_model_dir = None
        else:
            if self.config.is_load_model_dir_local:
                self.load_model_dir = create_file_path_string(
                    self.config.load_model_dir_list + [f"seed_{self.config.seed}"],
                    local_path=True,
                )
            else:
                self.load_model_dir = create_file_path_string(
                    self.config.load_model_dir_list + [f"seed_{self.config.seed}"],
                    local_path=True,
                )

    def _load_finetune_model(self):
        raise NotImplementedError()
