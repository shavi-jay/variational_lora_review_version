import os
from typing import Optional, Union, Iterable, Tuple, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc
from deepchem.data import Dataset
from deepchem.metrics import (
    Metric,
    mae_score,
    rms_score,
    accuracy_score,
    roc_auc_score,
    r2_score,
)
from deepchem.feat import RawFeaturizer, DummyFeaturizer, Featurizer
from deepchem.splits import Splitter
from deepchem.models.optimizers import Optimizer, Adam, LearningRateSchedule
from deepchem.molnet import load_delaney, load_bace_regression, load_lipo

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from src.molformer.molformer_tokenizer import MolTranBertTokenizer
from src.molbert.tokenizer import MolbertTokenizer
from src.mole.mole_tokenizer import MolETokenizer
from src.splitter import PseudoScaffoldSplitter

from src.optimizer import Lamb
from src.utils import create_file_path_string
from src.dataset_loaders import (
    load_adme_hclint,
    load_adme_lm_clint,
    load_adme_perm,
)
from src.dataset_tasks import DATASET_TO_TASK_TYPE

DATASET_TO_LOAD_FUNCTION = {
    "delaney": load_delaney,
    "lipo": load_lipo,
    "bace": load_bace_regression,
    "freesolv": dc.deepchem.molnet.load_freesolv,
    "qm7": dc.deepchem.molnet.load_qm7,
    "qm8": dc.deepchem.molnet.load_qm8,
    "adme_hclint": load_adme_hclint,
    "adme_lm_clint": load_adme_lm_clint,
    "adme_perm": load_adme_perm,
    "bace_classification": dc.deepchem.molnet.load_bace_classification,
}


class EarlyStopping:
    """
    Helper class for implementing early stopping. Monitors checkpoints and chooses best checkpoint
    from the max checkpoints.
    """

    def __init__(
        self,
        max_checkpoints_to_keep: int,
        early_stopping: bool = True,
        min_epochs_to_train: int = 0,
    ):
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.early_stopping = early_stopping
        self.min_epochs_to_train = min_epochs_to_train
        self.first_checkpoint = True
        self.optimal_checkpoint = 1
        self.stopped = False

    def should_early_stop(self, test_loss: float, epoch: Optional[int] = None):
        if not self.early_stopping:
            return False
        if epoch is None and self.min_epochs_to_train != 0:
            raise ValueError("If min_epochs_to_train > 0, then epoch must be specified")
        if epoch is not None and epoch < self.min_epochs_to_train:
            return False
        if self.first_checkpoint:
            self.optimal_loss = test_loss
            self.optimal_checkpoint = 1
            self.first_checkpoint = False
        else:
            if test_loss < self.optimal_loss:
                self.optimal_checkpoint = 1
                self.optimal_loss = test_loss
            elif self.optimal_checkpoint >= self.max_checkpoints_to_keep:
                self.stopped = True
                return True
            else:
                self.optimal_checkpoint += 1

        return False


def get_wandb_logger(is_wandb_logger: bool = False, wandb_kwargs: dict = {}):
    if is_wandb_logger:
        wandb_logger = dc.deepchem.models.WandbLogger(**wandb_kwargs)
    else:
        wandb_logger = None
    return wandb_logger


def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable%": round(trainable_params * 100 / total_params, 2),
    }


def get_splitter(splitter_type: str) -> str | Splitter:
    MAP_TO_SPLITTER = {
        "scaffold": "scaffold",
        "random": "random",
        "pseudo_scaffold": PseudoScaffoldSplitter(),
    }

    try:
        return MAP_TO_SPLITTER[splitter_type]
    except:
        if splitter_type not in MAP_TO_SPLITTER.keys():
            raise ValueError(f"Splitter {splitter_type} not recognised")
        return


def get_featurizer(featurizer_type: str) -> Featurizer:
    MAP_TO_FEATURIZER = {
        "dummy": DummyFeaturizer(),
        "raw_no_smiles": RawFeaturizer(smiles=False),
        "raw_smiles": RawFeaturizer(smiles=True),
    }

    try:
        return MAP_TO_FEATURIZER[featurizer_type]
    except:
        if featurizer_type not in MAP_TO_FEATURIZER.keys():
            raise ValueError(f"Featurizer {featurizer_type} not recognised")
        return


def get_load_func(dataset: str):
    try:
        return DATASET_TO_LOAD_FUNCTION[dataset], DATASET_TO_TASK_TYPE[dataset]
    except:
        if dataset not in DATASET_TO_LOAD_FUNCTION.keys():
            raise ValueError(f"Dataset {dataset} not recognised")
        raise ValueError(f"Error getting load function for dataset {dataset}")


def get_finetuning_datasets(
    dataset: str,
    splitter_type: str | Splitter = "scaffold",
    featurizer_type: str | None = None,
    task_number: int | None = None,
):
    load_func, task_type = get_load_func(dataset)

    if featurizer_type is None:
        featurizer = RawFeaturizer(smiles=True)
    else:
        featurizer = get_featurizer(featurizer_type)

    tasks, (train_dataset, val_dataset, test_dataset), transformers = load_func(
        featurizer=featurizer,
        splitter=get_splitter(splitter_type),
        task_number=task_number,
    )

    return (train_dataset, val_dataset, test_dataset), task_type


def get_metrics(metric_string_list: list[str]):
    """
    Converts list of metrics to deepchem.metrics list.

    Metric types: "rmse", "mae", "acc"

    """
    metric_list = []

    if "rms" in metric_string_list:
        metric_list.append(Metric(rms_score))
    if "r2" in metric_string_list:
        metric_list.append(Metric(r2_score))
    if "mae" in metric_string_list:
        metric_list.append(Metric(mae_score))
    if "accuracy" in metric_string_list:
        metric_list.append(Metric(accuracy_score))
    if "roc_auc" in metric_string_list:
        metric_list.append(Metric(roc_auc_score))

    if len(metric_list) == 0:
        raise ValueError("No metric identified")

    return metric_list


def create_tokenizer(model_type: str) -> PreTrainedTokenizer:
    if model_type == "chemberta":
        tokenizer_path = "DeepChem/ChemBERTa-77M-MTR"
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    elif model_type == "molformer":
        vocab_file_path = os.path.join(
            create_file_path_string(["molformer"]), "bert_vocab.txt"
        )
        tokenizer = MolTranBertTokenizer(vocab_file_path)
    elif model_type == "molbert":
        tokenizer = MolbertTokenizer()
    elif model_type == "mole":
        tokenizer = MolETokenizer()
    else:
        raise ValueError(f"Unrecognised model type: {model_type}")
    return tokenizer


def get_activation_function(activation_func: str) -> nn.Module:
    ACTIVATION_FUNC_TO_MODULE = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
    }

    try:
        return ACTIVATION_FUNC_TO_MODULE[activation_func]
    except:
        if activation_func not in ACTIVATION_FUNC_TO_MODULE.keys():
            raise ValueError(f"Activation function {activation_func} not recognised")
        return


def get_optimizer(
    optimizer_type: str, learning_rate: Union[float, LearningRateSchedule]
) -> Optimizer:
    OPTIMIZER_MAP = {
        "adam": Adam(learning_rate=learning_rate),
        "lamb": Lamb(learning_rate=learning_rate),
    }

    try:
        return OPTIMIZER_MAP[optimizer_type]
    except:
        if optimizer_type not in OPTIMIZER_MAP.keys():
            raise ValueError(f"Optimizer {optimizer_type} not recognised")
        return


def classification_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int = 2,
):
    y_pred = torch.argmax(logits.view(-1, num_labels), dim=1)
    batch_size = y_pred.shape[0]

    return torch.sum(y_pred != labels.view(-1)) / batch_size


def default_generator_with_batches(
    dataset: Dataset,
    epochs: int = 1,
    batch_size: int = 100,
    mode: str = "fit",
    deterministic: bool = True,
    pad_batches: bool = True,
) -> Iterable[Tuple[List, List, List]]:
    """Create a generator that iterates batches for a dataset.

    Subclasses may override this method to customize how model inputs are
    generated from the data.

    Parameters
    ----------
    dataset: Dataset
      the data to iterate
    epochs: int
      the number of times to iterate over the full dataset
    mode: str
      allowed values are 'fit' (called during training), 'predict' (called
      during prediction), and 'uncertainty' (called during uncertainty
      prediction)
    deterministic: bool
      whether to iterate over the dataset in order, or randomly shuffle the
      data for each epoch
    pad_batches: bool
      whether to pad each batch up to this model's preferred batch size

    Returns
    -------
    a generator that iterates batches, each represented as a tuple of lists:
    ([inputs], [outputs], [weights])
    """
    for epoch in range(epochs):
        for X_b, y_b, w_b, ids_b in dataset.iterbatches(
            batch_size=batch_size, deterministic=deterministic, pad_batches=pad_batches
        ):
            yield ([X_b], [y_b], [w_b])
