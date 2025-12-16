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
from deepchem.data import Dataset, DiskDataset
from deepchem.molnet.load_function.clintox_datasets import _ClintoxLoader
from deepchem.molnet.load_function.toxcast_datasets import _ToxcastLoader
from deepchem.molnet.load_function.bbbp_datasets import _BBBPLoader
from deepchem.molnet.load_function.tox21_datasets import _Tox21Loader
from typing import List, Optional, Tuple, Union

from src.utils import create_file_path_string, path_to_local_data
from src.dataset_tasks import get_dataset_task, ADME_HCLINT_TASKS


def load_clintox(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load ClinTox dataset

    The ClinTox dataset compares drugs approved by the FDA and
    drugs that have failed clinical trials for toxicity reasons.
    The dataset includes two classification tasks for 1491 drug
    compounds with known chemical structures:

    #. clinical trial toxicity (or absence of toxicity)
    #. FDA approval status.

    List of FDA-approved drugs are compiled from the SWEETLEAD
    database, and list of drugs that failed clinical trials for
    toxicity reasons are compiled from the Aggregate Analysis of
    ClinicalTrials.gov(AACT) database.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles" - SMILES representation of the molecular structure
    - "FDA_APPROVED" - FDA approval status
    - "CT_TOX" - Clinical trial results

    """

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("clintox", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "clintox", "processed"],
            create_file_path=False,
            local_path=True,
        )

    loader = _ClintoxLoader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )
    return loader.load_dataset(f"clintox_{task}", reload)


def load_toxcast(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Toxcast dataset

    ToxCast is an extended data collection from the same
    initiative as Tox21, providing toxicology data for a large
    library of compounds based on in vitro high-throughput
    screening. The processed collection includes qualitative
    results of over 600 experiments on 8k compounds.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles": SMILES representation of the molecular structure
    - "ACEA_T47D_80hr_Negative" ~ "Tanguay_ZF_120hpf_YSE_up": Bioassays results.
      Please refer to the section "high-throughput assay information" at
      https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data
      for details.
    """

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("toxcast", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "toxcast", "processed", task],
            create_file_path=False,
            local_path=True,
        )

    loader = _ToxcastLoader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )
    return loader.load_dataset(f"toxcast_{task}", reload)


def load_bbbp(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBP dataset

    The blood-brain barrier penetration (BBBP) dataset is designed for the
    modeling and prediction of barrier permeability. As a membrane separating
    circulating blood and brain extracellular fluid, the blood-brain barrier
    blocks most drugs, hormones and neurotransmitters. Thus penetration of the
    barrier forms a long-standing issue in development of drugs targeting
    central nervous system.

    This dataset includes binary labels for over 2000 compounds on their
    permeability properties.

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "name" - Name of the compound
    - "smiles" - SMILES representation of the molecular structure
    - "p_np" - Binary labels for penetration/non-penetration

    """

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("bbbp", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "bbbp", "processed"],
            create_file_path=False,
            local_path=True,
        )

    loader = _BBBPLoader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )
    return loader.load_dataset(f"bbbp", reload)


def load_tox21(
    featurizer: Union[dc.feat.Featurizer, str] = "ECFP",
    splitter: Union[dc.splits.Splitter, str, None] = "scaffold",
    transformers: List[Union[TransformerGenerator, str]] = ["balancing"],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Tox21 dataset

    The "Toxicology in the 21st Century" (Tox21) initiative created a public
    database measuring toxicity of compounds, which has been used in the 2014
    Tox21 Data Challenge. This dataset contains qualitative toxicity measurements
    for 8k compounds on 12 different targets, including nuclear receptors and
    stress response pathways.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles" - SMILES representation of the molecular structure
    - "NR-XXX" - Nuclear receptor signaling bioassays results
    - "SR-XXX" - Stress response bioassays results

    please refer to https://tripod.nih.gov/tox21/challenge/data.jsp for details.

    """

    task_number = kwargs.get("task_number", 0)

    if task_number is None:
        task_number = 0

    task = get_dataset_task("tox21", task_number)

    if data_dir is None:
        data_dir = create_file_path_string(
            ["finetuning_datasets", "tox21", "processed", task],
            create_file_path=False,
            local_path=True,
        )

    loader = _Tox21Loader(
        featurizer, splitter, transformers, [task], data_dir, save_dir, **kwargs
    )
    return loader.load_dataset(f"tox21_{task}", reload)


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
