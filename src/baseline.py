from typing import Union

import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
)
from xgboost import XGBClassifier, XGBRegressor
from ngboost import NGBClassifier, NGBRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier

from deepchem.data import Dataset
from deepchem.feat import Featurizer, RDKitDescriptors, MACCSKeysFingerprint

from src.training_utils import get_load_func
from src.fingerprint_featurizer import (
    CircularFingerprint,
    ConcatenatedRDKitDescriptorsCircularFingerprint,
)

FEATURIZER_TO_FEATURIZER_MODEL = {
    "RDKit": RDKitDescriptors(),
    "RDKit2": RDKitDescriptors(ipc_avg=False),  # fixing a bug in deepchem
    "ECFP": CircularFingerprint(),
    "ECFP_counts": CircularFingerprint(is_counts_based=True),
    "ECFP_binary": CircularFingerprint(is_counts_based=False),
    "RDKit_ECFP_counts": ConcatenatedRDKitDescriptorsCircularFingerprint(
        is_counts_based=True, ipc_avg=False
    ),
    "RDKit_ECFP_binary": ConcatenatedRDKitDescriptorsCircularFingerprint(
        is_counts_based=False, ipc_avg=False
    ),
}


def baseline(
    dataset: str,
    seed: int = 0,
    task_number: int = 0,
    featurizer: str = "ECFP",
    splitter_type="scaffold",
    metric_string_list=["roc_auc"],
    model_type="random_forest",
    model_args: dict = {},
):
    load_func, task_type = get_load_func(dataset)

    featurizer_model = None
    if featurizer in FEATURIZER_TO_FEATURIZER_MODEL:
        featurizer_model = FEATURIZER_TO_FEATURIZER_MODEL[featurizer]
    if featurizer_model is None:
        raise ValueError(f"Featurizer {featurizer} is not supported.")
    tasks, (train_dataset, val_dataset, test_dataset), transformers = load_func(
        splitter=splitter_type, task_number=task_number, featurizer=featurizer_model
    )

    model = None
    if task_type == "regression":
        if model_type == "random_forest":
            model = RandomForestRegressor(random_state=seed, **model_args)
        elif model_type == "xgboost":
            model = XGBRegressor(random_state=seed, subsample=0.8, **model_args)
        elif model_type == "ngboost":
            model = NGBRegressor(random_state=seed, **model_args)
    elif task_type == "classification":
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=seed, **model_args)
        elif model_type == "xgboost":
            model = XGBClassifier(random_state=seed, subsample=0.8, **model_args)
        elif model_type == "ngboost":
            model = NGBClassifier(random_state=seed, **model_args)
    else:
        raise ValueError(f"Task type {task_type} is not supported.")
    if model is None:
        raise ValueError(f"Model type {model_type} is not supported.")

    shape = train_dataset.X.shape

    model.fit(train_dataset.X.reshape(shape[0], -1), train_dataset.y.squeeze())

    scores = evaluate_metrics(
        dataset=test_dataset, model=model, metric_string_list=metric_string_list
    )

    return scores


def evaluate_metrics(
    dataset: Dataset,
    model: Union[
        RandomForestRegressor,
        RandomForestClassifier,
        XGBRegressor,
        XGBClassifier,
        NGBRegressor,
        NGBClassifier,
    ],
    metric_string_list: list[str] = ["roc_auc"],
):
    scores = {}

    for metric in metric_string_list:
        shape = dataset.X.shape

        score_name = f"{metric}_score"
        if metric == "rms":
            y_pred = model.predict(dataset.X.reshape(shape[0], -1))
            scores.update({score_name: np.sqrt(mean_squared_error(dataset.y, y_pred))})
        elif metric == "mae":
            y_pred = model.predict(dataset.X)
            scores.update({score_name: mean_absolute_error(dataset.y, y_pred)})
        elif metric == "roc_auc":
            if isinstance(model, (RandomForestRegressor, XGBRegressor, NGBRegressor)):
                raise ValueError("ROC AUC can only be calculated for classification models")
            y_score = model.predict_proba(dataset.X)[:, 1]
            scores.update({score_name: roc_auc_score(dataset.y, y_score)})
        elif metric == "accuracy":
            y_pred = model.predict(dataset.X)
            scores.update({score_name: accuracy_score(dataset.y, y_pred)})

    return scores
