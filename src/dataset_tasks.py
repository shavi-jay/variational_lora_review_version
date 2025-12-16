from deepchem.molnet.load_function.bace_datasets import (
    BACE_CLASSIFICATION_TASKS,
    BACE_REGRESSION_TASKS,
)
from deepchem.molnet.load_function.delaney_datasets import DELANEY_TASKS
from deepchem.molnet.load_function.lipo_datasets import LIPO_TASKS
from deepchem.molnet.load_function.freesolv_dataset import FREESOLV_TASKS

DATASET_TO_TASK_TYPE = {
    "delaney": "regression",
    "lipo": "regression",
    "bace": "regression",
    "freesolv": "regression",
    "qm7": "regression",
    "qm8": "regression",
    "adme_hclint": "regression",
    "adme_lm_clint": "regression",
    "adme_perm": "regression",
    "bace_classification": "classification",
    "bbbp": "classification",
    "clintox": "classification",
    "toxcast": "classification",
    "tox21": "classification",
}

# Tasks

CLINTOX_TASKS = ["FDA_APPROVED", "CT_TOX"]

TOXCAST_TASKS = [
    "APR_HepG2_CellLoss_72h_dn",
    "ATG_NRF2_ARE_CIS_up",
    "ATG_PXRE_CIS_up",
    "BSK_3C_HLADR_down",
    "BSK_3C_Proliferation_down",
    "BSK_3C_SRB_down",
    "BSK_3C_Vis_down",
    "BSK_4H_Eotaxin3_down",
    "BSK_CASM3C_Proliferation_down",
    "BSK_LPS_VCAM1_down",
    "BSK_SAg_CD38_down",
    "BSK_SAg_CD40_down",
    "BSK_SAg_Proliferation_down",
    "BSK_hDFCGF_CollagenIII_down",
    "BSK_hDFCGF_Proliferation_down",
    "CEETOX_H295R_11DCORT_dn",
    "CEETOX_H295R_ANDR_dn",
    "TOX21_ARE_BLA_agonist_ratio",
    "TOX21_TR_LUC_GH3_Antagonist",
]

BBBP_TASKS = ["p_np"]

TOX21_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

ADME_HCLINT_TASKS = ["LOG_HLM_CLint"]

ADME_LM_CLINT_TASKS = ["LOG_HLM_CLint", "LOG_RLM_CLint"]

ADME_PERM_TASKS = ["LOG_MDR1-MDCK_ER", "LOG_HLM_CLint", "LOG_RLM_CLint"]

DATASET_TO_TASK = {
    "bace_classification": BACE_CLASSIFICATION_TASKS,
    "clintox": CLINTOX_TASKS,
    "toxcast": TOXCAST_TASKS,
    "bbbp": BBBP_TASKS,
    "tox21": TOX21_TASKS,
    "bace": BACE_REGRESSION_TASKS,
    "lipo": LIPO_TASKS,
    "freesolv": FREESOLV_TASKS,
    "delaney": DELANEY_TASKS,
    "adme_hclint": ADME_HCLINT_TASKS,
    "adme_lm_clint": ADME_LM_CLINT_TASKS,
    "adme_perm": ADME_PERM_TASKS,
}


def get_dataset_task(dataset: str, task_number: int | None):
    if task_number is None:
        return None
    else:
        try:
            return DATASET_TO_TASK[dataset][task_number]
        except:
            if dataset not in DATASET_TO_TASK.keys():
                raise ValueError(f"Dataset {dataset} does not have multiple tasks")
            if not isinstance(task_number, int):
                raise ValueError("Task number must be int")
            if task_number >= len(DATASET_TO_TASK[dataset]):
                raise IndexError(f"Task {task_number} does not exist")


def get_number_of_tasks(dataset: str):
    try:
        return len(DATASET_TO_TASK[dataset])
    except:
        if dataset in DATASET_TO_TASK_TYPE.keys():
            return 0
        else:
            raise ValueError(f"Dataset {dataset} does not exist")
