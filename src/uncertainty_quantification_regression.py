from typing import Optional, Iterable, Literal
from dataclasses import dataclass, field

import torch
import torch.nn as nn

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, t

import matplotlib.pyplot as plt
import seaborn as sns

from deepchem.data import Dataset
from deepchem.utils.typing import OneOrMany

from src.deepchem_hf_models import HuggingFaceModel

from src.utils import (
    tensor_to_numpy,
    set_seed,
    create_file_path_string,
    load_yaml_config,
)
from src.training_utils import get_finetuning_datasets

from src.uncertainty_quantification import (
    UncertaintyQuantificationBase,
    UncertaintyQuantificationBaseConfig,
)


@dataclass
class UncertaintyRegressionPredictionOutput:
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None


@dataclass
class UncertaintyRegressionPredictionOutputNumpy:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    label: Optional[np.ndarray] = None


class UncertaintyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_pretrained_model_set = False
        self.pretrained_model: nn.Module | None = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def predict_mean_and_std(
        self, *args, **kwargs
    ) -> UncertaintyRegressionPredictionOutput:
        raise NotImplementedError()

    def from_pretrained_local(self, pretrained_model_path: str):
        if self.pretrained_model is not None:
            data = torch.load(pretrained_model_path, map_location=self.device)
            self.pretrained_model.load_state_dict(
                data["model_state_dict"], strict=False
            )
            self.is_pretrained_model_set = True
        else:
            raise RuntimeError("No model instantiated. Can't load model.")


###


class UncertaintyQuantificationRegressionBase(UncertaintyQuantificationBase):
    """Base class for uncertainty quantification on regression tasks"""

    @torch.no_grad
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
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    @torch.no_grad
    def generate_student_t_lower_tail(mean_vector, std_vector, quantile: float):
        # return mean_vector + norm.ppf(quantile) * std_vector
        return mean_vector + t.ppf(quantile, 5) * std_vector

    @staticmethod
    @torch.no_grad
    def generate_centred_gaussian_confidence_interval(
        mean_vector, std_vector, confidence_prob: float
    ):
        lower_quantile_prob = 0.5 - confidence_prob / 2.0
        upper_quantile_prob = 0.5 + confidence_prob / 2.0

        lower_quantile = mean_vector + t.ppf(lower_quantile_prob, 5) * std_vector
        upper_quantile = mean_vector + t.ppf(upper_quantile_prob, 5) * std_vector

        return lower_quantile, upper_quantile

    @staticmethod
    @torch.no_grad
    def generate_gaussian_lower_tail(mean_vector, std_vector, quantile: float):
        lower_tail = mean_vector + norm.ppf(quantile) * std_vector
        return lower_tail

    def proportion_in_centred_confidence_interval(
        self, dataset: Dataset, quantile: float
    ):
        uncertainty_prediction_output = self.predict_mean_and_std_on_dataset(
            dataset, compute_std=True, output_labels=True, to_numpy=True
        )

        (
            lower_quantile,
            upper_quantile,
        ) = self.generate_centred_gaussian_confidence_interval(
            uncertainty_prediction_output.mean,
            uncertainty_prediction_output.std,
            quantile,
        )

        is_data_in_confidence_interval = (
            uncertainty_prediction_output.label < upper_quantile
        ) & (uncertainty_prediction_output.label > lower_quantile)

        return np.mean(is_data_in_confidence_interval)

    def proportion_in_lower_tail(self, dataset: Dataset, quantile: float):
        uncertainty_prediction_output = self.predict_mean_and_std_on_dataset(
            dataset, compute_std=True, output_labels=True, to_numpy=True
        )

        lower_tail = self.generate_gaussian_lower_tail(
            uncertainty_prediction_output.mean,
            uncertainty_prediction_output.std,
            quantile,
        )

        return np.mean(uncertainty_prediction_output.label < lower_tail)

    def proportion_in_confidence_interval(
        self, dataset: Dataset, quantile: float, interval_type: str = "centred"
    ):
        if interval_type == "centred":
            return self.proportion_in_centred_confidence_interval(dataset, quantile)
        elif interval_type == "lower_tail":
            return self.proportion_in_lower_tail(dataset, quantile)

    def proportion_in_confidence_interval_range(
        self,
        dataset: Dataset,
        quantile_list: Iterable[float],
        interval_type: Literal["lower_tail", "centred"] = "centred",
    ):
        return np.array(
            [
                self.proportion_in_confidence_interval(dataset, quantile, interval_type)
                for quantile in quantile_list
            ]
        )

    def regression_expected_calibration_error(
        self, dataset: Dataset, step_size: float = 0.05, ece_type="ece_centred"
    ):
        print("Calculating Expected Calibration Error (ECE)...")
        if ece_type == "ece_centred":
            interval_type = "centred"
        elif ece_type == "ece_lower_tail":
            interval_type = "lower_tail"
        else:
            raise ValueError(
                "ece_type must be either 'ece_centred' or 'ece_lower_tail'"
            )

        if self.deepchem_task_type == "regression":
            quantile_array = np.arange(step_size, 1, step_size)
            accuracy_array = self.proportion_in_confidence_interval_range(
                dataset, quantile_array, interval_type
            )
            return np.mean(np.abs(quantile_array - accuracy_array))
        else:
            return UserWarning("This metric is used for regression tasks")

    def gaussian_negative_log_likelihood(self, dataset: Dataset):
        uncertainty_prediction_output = self.predict_mean_and_std_on_dataset(
            dataset, compute_std=True, output_labels=True, to_numpy=True
        )

        print("Calculating Gaussian Negative Log Likelihood (NLL)...")
        mean_output = uncertainty_prediction_output.mean
        std_output = uncertainty_prediction_output.std
        labels = uncertainty_prediction_output.label

        if std_output is None:
            raise ValueError(
                "Standard deviation output is required for NLL calculation."
            )
        if mean_output is None:
            raise ValueError("Mean output is required for NLL calculation.")
        if labels is None:
            raise ValueError("Labels are required for NLL calculation.")
        nll = np.mean(
            0.5 * np.pow((mean_output - labels) / std_output, 2) + np.log(std_output)
        )

        return nll.item()

    def root_mean_squared_error(self, dataset: Dataset):
        uncertainty_prediction_output = self.predict_mean_and_std_on_dataset(
            dataset, compute_std=True, output_labels=True, to_numpy=True
        )

        print("Calculating Root Mean Square Error (RMSE)...")
        mean_output = uncertainty_prediction_output.mean
        labels = uncertainty_prediction_output.label

        if mean_output is None:
            raise ValueError("Mean output is required for RMSE calculation.")
        if labels is None:
            raise ValueError("Labels are required for RMSE calculation.")

        rmse = np.sqrt(np.mean((mean_output - labels) ** 2))

        return rmse.item()


class UncertaintyQuantificationRegressionHF(UncertaintyQuantificationRegressionBase):
    """Performs uncertainty quantification on Uncertainty Models"""

    @torch.no_grad
    def predict_mean_and_std_on_batch(
        self,
        inputs,
        compute_std: bool = False,
        **kwargs,
    ) -> UncertaintyRegressionPredictionOutput:
        return self.finetune_model.model.predict_mean_and_std(
            **inputs, compute_std=compute_std
        )

    @torch.no_grad
    def predict_mean_and_std_on_dataset(
        self,
        test_dataset: Dataset,
        compute_std: bool = False,
        output_labels: bool = False,
        to_numpy: bool = False,
        predict_mean_and_std_kwargs={},
    ):
        if self.finetune_model is None:
            raise RuntimeError("No finetune model loaded. Can't perform prediction.")
        generator = self.finetune_model.default_generator(
            test_dataset, epochs=1, deterministic=True, pad_batches=False
        )

        mean_values = []

        if compute_std:
            std_values = []
        if output_labels:
            label_values = []

        for batch in generator:
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self.finetune_model._prepare_batch(batch)

            output = self.predict_mean_and_std_on_batch(
                inputs=inputs, compute_std=compute_std, **predict_mean_and_std_kwargs
            )

            mean_values.append(output.mean)

            if compute_std:
                std_values.append(output.std)
            if output_labels:
                label_values.append(labels.squeeze())

        mean = torch.hstack(mean_values)
        std = torch.hstack(std_values) if compute_std else None
        true_label = torch.hstack(label_values) if output_labels else None

        if to_numpy:
            return UncertaintyRegressionPredictionOutputNumpy(
                mean=tensor_to_numpy(mean),
                std=tensor_to_numpy(std),
                label=tensor_to_numpy(true_label),
            )
        else:
            return UncertaintyRegressionPredictionOutput(
                mean=mean, std=std, label=true_label
            )


def generate_reliability_plot_regression(
    config_dir: list[str],
    config_file: str,
    config_class,
    uncertainty_quantification_class,
    dataset_list: list[str],
    seed_range: list[int],
    load_model_dir_list_list: list[list[str]],
    save_path: list[str],
    fig_name: str,
    save_file_name: str,
    interval_type: str = "centred",
    step_size: float = 0.05,
    additional_uncertainty_quantification_kwargs: dict = {},
):
    x_range = np.arange(step_size, 1, step_size).tolist()

    dataset_values = []
    seed_values = []
    accuracy_values = []
    confidence_values = []

    for dataset, load_model_dir_list in zip(dataset_list, load_model_dir_list_list):
        for seed in seed_range:
            yaml_config = load_yaml_config(config_dir, config_file)

            config_dict = {
                **yaml_config,
                "dataset": dataset,
                "seed": seed,
                "load_model_dir_list": load_model_dir_list,
                **additional_uncertainty_quantification_kwargs,
            }

            config: UncertaintyQuantificationBaseConfig = config_class(**config_dict)

            uncertainty_quantifier: UncertaintyQuantificationRegressionHF = (
                uncertainty_quantification_class(config)
            )

            accuracy = uncertainty_quantifier.proportion_in_confidence_interval_range(
                dataset=uncertainty_quantifier.test_dataset,
                quantile_list=x_range,
                interval_type=interval_type,
            )

            dataset_values += [dataset] * len(x_range)
            seed_values += [seed] * len(x_range)
            accuracy_values += accuracy.tolist()
            confidence_values += x_range

    reliability_df = pd.DataFrame(
        {
            "confidence": confidence_values,
            "accuracy": accuracy_values,
            "dataset": dataset_values,
            "seed": seed_values,
        }
    )

    fig, ax = plt.subplots()

    sns.lineplot(
        data=reliability_df, x="confidence", y="accuracy", hue="dataset", ax=ax
    )
    ax.plot(x_range, x_range, linestyle="dashed", color="black", alpha=0.1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(fig_name)

    save_abs_path = create_file_path_string(save_path, True)

    save_fig_path = os.path.join(save_abs_path, f"{save_file_name}_fig.png")
    fig.savefig(save_fig_path)
    plt.close(fig)
