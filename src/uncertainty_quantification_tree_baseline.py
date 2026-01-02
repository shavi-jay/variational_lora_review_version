from typing import Union
from dataclasses import dataclass, field
import numpy as np
import torch

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from src.baseline import FEATURIZER_TO_FEATURIZER_MODEL
from src.utils import set_seed

from src.uncertainty_quantification_regression import (
    UncertaintyQuantificationRegressionBase,
    UncertaintyRegressionPredictionOutput,
    UncertaintyRegressionPredictionOutputNumpy,
)
from src.uncertainty_quantification import (
    UncertaintyQuantificationBase,
    UncertaintyQuantificationBaseConfig,
)

from xgboost import XGBClassifier, XGBRegressor, XGBModel
from ngboost import NGBClassifier, NGBRegressor, NGBoost
from sklearn.neural_network import MLPRegressor, MLPClassifier

from deepchem.data import Dataset
from deepchem.feat import Featurizer, RDKitDescriptors, MACCSKeysFingerprint

from src.training_utils import get_load_func


@dataclass(kw_only=True)
class UncertaintyQuantificationBaselineConfig(UncertaintyQuantificationBaseConfig):
    """Configuration for Uncertainty Quantification for Tree-based Baseline Models"""

    model_args: dict = field(default_factory=dict)


class UncertaintyQuantificationBaseline(UncertaintyQuantificationRegressionBase):
    """Uncertainty Quantification for Tree-based Baseline Models"""

    def __init__(self, config: UncertaintyQuantificationBaselineConfig):
        super().__init__(config)

        self.config: UncertaintyQuantificationBaselineConfig

    def _initial_setup(self):
        set_seed(self.config.seed)

        load_func_output = get_load_func(self.config.dataset)
        if load_func_output is not None:
            load_func, self.deepchem_task_type = load_func_output
        else:
            raise ValueError(f"Dataset {self.config.dataset} is not supported.")

        featurizer = None
        if self.config.featurizer_type in FEATURIZER_TO_FEATURIZER_MODEL:
            featurizer = FEATURIZER_TO_FEATURIZER_MODEL[self.config.featurizer_type]
        if featurizer is None:
            raise ValueError(
                f"Featurizer {self.config.featurizer_type} is not supported."
            )
        (
            tasks,
            (self.train_dataset, self.val_dataset, self.test_dataset),
            transformers,
        ) = load_func(
            splitter=self.config.splitter_type,
            task_number=self.config.task_number,
            featurizer=featurizer,
        )

    def _load_finetune_model(self):
        model = None
        if self.deepchem_task_type == "regression":
            if self.config.model_type == "random_forest":
                model = RandomForestRegressor(
                    random_state=self.config.seed, **self.config.model_args
                )
            if self.config.model_type == "ngboost":
                model = NGBRegressor(
                    random_state=self.config.seed, **self.config.model_args
                )
        elif self.deepchem_task_type == "classification":
            if self.config.model_type == "random_forest":
                model = RandomForestClassifier(
                    random_state=self.config.seed, **self.config.model_args
                )
            if self.config.model_type == "ngboost":
                model = NGBClassifier(
                    random_state=self.config.seed, **self.config.model_args
                )
        else:
            raise ValueError(f"Task type {self.deepchem_task_type} is not supported.")
        if model is None:
            raise ValueError(f"Model type {self.config.model_type} is not supported.")

        shape = self.train_dataset.X.shape

        print(
            f"Training {self.config.model_type} model on dataset {self.config.dataset}"
        )
        model.fit(
            self.train_dataset.X.reshape(shape[0], -1), self.train_dataset.y.squeeze()
        )
        print(f"Model trained successfully.")
        self.tree_model = model

    def predict_mean_and_std_on_dataset(
        self,
        test_dataset: Dataset,
        compute_std: bool = False,
        output_labels: bool = False,
        to_numpy: bool = False,
        predict_mean_and_std_kwargs={},
    ) -> (
        UncertaintyRegressionPredictionOutput
        | UncertaintyRegressionPredictionOutputNumpy
    ):

        shape = test_dataset.X.shape

        if self.config.model_type == "random_forest" and isinstance(
            self.tree_model, (RandomForestRegressor, RandomForestClassifier)
        ):
            model_predictions = np.array(
                [
                    tree.predict(test_dataset.X.reshape(shape[0], -1))
                    for tree in self.tree_model.estimators_
                ]
            )
            mean_predictions = np.mean(model_predictions, axis=0)
            if compute_std:
                std_predictions = np.std(model_predictions, axis=0)
            else:
                std_predictions = None
        elif self.config.model_type == "ngboost" and isinstance(
            self.tree_model, NGBoost
        ):
            model_predictions = self.tree_model.pred_dist(
                test_dataset.X.reshape(shape[0], -1)
            )
            mean_predictions = model_predictions.loc
            if compute_std:
                std_predictions = model_predictions.scale
            else:
                std_predictions = None
        if output_labels:
            label = test_dataset.y.squeeze()
        else:
            label = None
        if to_numpy:
            return UncertaintyRegressionPredictionOutputNumpy(
                mean=mean_predictions, std=std_predictions, label=label
            )
        else:
            return UncertaintyRegressionPredictionOutput(
                mean=mean_predictions,
                std=std_predictions,
                label=(
                    torch.tensor(label, dtype=torch.float32)
                    if label is not None
                    else None
                ),
            )


def baseline_tree_uq(
    dataset: str,
    seed: int = 0,
    task_number: int = 0,
    featurizer: str = "ECFP",
    splitter_type="scaffold",
    metric_string_list=["ece"],
    model_type="random_forest",
    model_args: dict = {},
):
    config = UncertaintyQuantificationBaselineConfig(
        dataset=dataset,
        task_number=task_number,
        splitter_type=splitter_type,
        featurizer_type=featurizer,
        seed=seed,
        model_type=model_type,
        model_args=model_args,
    )

    uq_model = UncertaintyQuantificationBaseline(config)

    scores = {}

    if "rms" in metric_string_list:
        scores["rms_score"] = uq_model.root_mean_squared_error(
            dataset=uq_model.test_dataset
        )
    if "ece" in metric_string_list:
        scores["ece_score"] = uq_model.regression_expected_calibration_error(
            dataset=uq_model.test_dataset
        )
    if "nll" in metric_string_list:
        scores["nll_score"] = uq_model.gaussian_negative_log_likelihood(
            dataset=uq_model.test_dataset
        )

    return scores


if __name__ == "__main__":
    config = UncertaintyQuantificationBaseConfig(
        dataset="adme_perm",
        task_number=0,
        splitter_type="scaffold",
        featurizer_type="ECFP",
        seed=1,
        model_type="ngboost",
    )

    uq_model = UncertaintyQuantificationBaseline(config)

    print("Expected Calibration Error (ECE):")
    print(uq_model.regression_expected_calibration_error(dataset=uq_model.test_dataset))

    print("Negative Log Likelihood (NLL):")
    print(uq_model.gaussian_negative_log_likelihood(dataset=uq_model.test_dataset))
