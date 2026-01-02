"""
Clinical Toxicity (clintox) dataset loader.
@author Caleb Geniesse
"""

import os
import pandas as pd
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import (
    TransformerGenerator,
    _MolnetLoader,
)
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

from src.utils import create_file_path_string, path_to_local_data
from src.dataset_tasks import get_dataset_task


ADME_HCLINT_PATH = path_to_local_data(
    "finetuning_datasets/adme/adme-fang-v1-LOG_HLM_CLint"
)


class _ADME_HCLINT_Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "ADME_HCLINT.csv")
        if not os.path.exists(dataset_file):
            dataset_file = ADME_HCLINT_PATH
        loader = dc.data.CSVLoader(
            tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_adme_hclint(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load ADME_HCLINT dataset

    Assessing ADME properties helps understand a drug candidate's interaction
    with the body in terms of absorption, distribution, metabolism, and excretion,
    essential for evaluating its efficacy, safety, and clinical potential.

    https://polarishub.io/datasets/biogen/adme-fang-v1

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles" - SMILES representation of the molecular structure
    - "HCLint" - Microsomal stability (human and rat)
    """

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("adme_hclint", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "admehclint", "processed", task],
            create_file_path=False,
            local_path=True,
        )

    loader = _ADME_HCLINT_Loader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )
    return loader.load_dataset(f"admehclint", reload)


ADME_LM_CLINT_PATH = path_to_local_data(
    "finetuning_datasets/adme/adme-fang-v1-LOG_LM_CLint.csv"
)


class _ADME_LM_CLINT_Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "ADME_LM_CLINT.csv")
        if not os.path.exists(dataset_file):
            dataset_file = ADME_LM_CLINT_PATH
        loader = dc.data.CSVLoader(
            tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_adme_lm_clint(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("adme_lm_clint", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "adme_lm_clint", "processed", task],
            create_file_path=False,
            local_path=True,
        )

    loader = _ADME_LM_CLINT_Loader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )

    return loader.load_dataset(f"adme_lm_clint_{task}", reload)


ADME_PERM_PATH = path_to_local_data(
    "finetuning_datasets/adme/adme-fang-v1-LOG_MDR1-MDCK_ER.csv"
)


class _ADME_PERM_Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "ADME_PERM.csv")
        if not os.path.exists(dataset_file):
            dataset_file = ADME_PERM_PATH
        loader = dc.data.CSVLoader(
            tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_adme_perm(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("adme_perm", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "adme_perm", "processed", task],
            create_file_path=False,
            local_path=True,
        )

    loader = _ADME_PERM_Loader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )

    return loader.load_dataset(f"adme_perm_{task}", reload)
