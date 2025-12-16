import deepchem as dc

import os
from pathlib import Path
import yaml

from contextlib import contextmanager
import time

import numpy as np
import random
import torch
import pandas as pd

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from src.dataset_tasks import get_number_of_tasks, get_dataset_task

def create_file_path_string(
    file_dir_list: list[str], create_file_path: bool = False, local_path: bool = False
):
    """
    Creates a file path from a list of folders and file name.
    If create_file_bath is True, then a folder will be created with the file path.
    If local_path is True, file will be created in the local directory
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))

    if not local_path:
        file_path = os.path.join(abs_path, *file_dir_list)
    else:
        full_dir_list = ["local_data"] + file_dir_list
        file_path = os.path.join(Path(abs_path).parents[0], *full_dir_list)

    if create_file_path and not os.path.exists(file_path):
        os.makedirs(file_path)

    return file_path

def set_seed(seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(config_load_paths: list[str], config_file_name: str):
    path_to_folder = create_file_path_string(config_load_paths)
    path_to_file = os.path.join(path_to_folder, config_file_name)

    with open(path_to_file, "r") as file:
        config = yaml.safe_load(file)

    return config

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray | None:
    if tensor is None:
        return None
    else:
        return tensor.cpu().detach().numpy()


def path_to_local_data(path: str) -> str:
    abs_path = os.path.abspath(os.path.dirname(__file__))
    local_data_path = os.path.join(Path(abs_path).parents[0], "local_data", path)
    return local_data_path
