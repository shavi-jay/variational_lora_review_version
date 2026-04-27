from typing import Optional, Dict, Callable, Literal
from dataclasses import dataclass, field
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader

from peft import get_peft_model, LoraConfig, TaskType
from laplace import Laplace

from deepchem.data import Dataset
from deepchem.utils.typing import OneOrMany

from src.dataset_tasks import get_dataset_task
from src.utils import (
    tensor_to_numpy,
    set_seed,
    create_file_path_string,
    load_yaml_config,
)
from src.training_utils import get_finetuning_datasets
from src.deepchem_hf_models import HuggingFaceModel
from src.model_chemberta import Chemberta
from src.model_molformer import (
    MolformerDeepchem,
    MolformerConfig,
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.model_molbert import (
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
    MolbertDeepchem,
    MolbertConfig,
)
from src.model_mole import (
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
    MolEDeepchem,
    MolEExtraConfig,
)

from src.uncertainty_quantification import (
    UncertaintyQuantificationBaseConfig,
    UncertaintyQuantificationBase,
)
from src.uncertainty_quantification_regression import (
    UncertaintyQuantificationRegressionHF,
    UncertaintyRegressionPredictionOutput,
    UncertaintyRegressionPredictionOutputNumpy,
)


class _LogitsOnlyWrapper(nn.Module):
    """Unwraps LikelihoodSequenceClassifierOutput so Laplace sees a plain tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        return logits


class DeepChemDatasetAdapter(TorchDataset):
    def __init__(self, deepchem_dataset: Dataset, tokenizer, task_name: str):
        self.deepchem_dataset = deepchem_dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.batch_list = list(
            deepchem_dataset.iterbatches(
                batch_size=1, deterministic=True, pad_batches=False
            )
        )

    def __len__(self):
        return len(self.deepchem_dataset)

    def __getitem__(self, idx):
        X_b, y_b, w_b, ids_b = self.batch_list[idx]
        return X_b[0], float(y_b[0][0])


def _make_collate_fn(tokenizer):
    def collate_fn(batch):
        smiles_list, labels = zip(*batch)
        tokens = tokenizer(
            list(smiles_list), padding=True, return_tensors="pt", truncation=True
        )
        tokens["labels"] = torch.tensor(labels, dtype=torch.float).unsqueeze(-1)
        return tokens

    return collate_fn


def _deepchem_to_dataloader(
    dataset: Dataset, tokenizer, batch_size: int = 32, task_name: str = "regression"
) -> DataLoader:
    adapter = DeepChemDatasetAdapter(dataset, tokenizer, task_name)
    return DataLoader(
        adapter,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_collate_fn(tokenizer),
    )


@dataclass(kw_only=True)
class UncertaintyQuantificationLaplaceLoraPEFTConfig(
    UncertaintyQuantificationBaseConfig
):
    batch_size: int = 100
    finetune_type: str = "lora"
    base_lora_config: dict = field(default_factory=dict)
    additional_lora_config: dict = field(default_factory=dict)
    base_pretrained_model_config: dict = field(default_factory=dict)
    additional_pretrained_model_config: dict = field(default_factory=dict)
    laplace_hessian_structure: str = "kron"
    laplace_subset_of_weights: str = "all"
    laplace_prior_precision: float = 1.0
    laplace_sigma_noise: float = 1.0
    laplace_optimize_prior: bool = True
    laplace_manual_jacobians: bool = False
    sequence_classifier_type: Literal["default", "likelihood"] = "default"


class UncertaintyQuantificationLaplaceLoraSingle(UncertaintyQuantificationBase):
    def __init__(self, config: UncertaintyQuantificationLaplaceLoraPEFTConfig):
        self.la = None
        super().__init__(config)
        self.config: UncertaintyQuantificationLaplaceLoraPEFTConfig

    def _load_finetune_model(self):
        if self.config.model_type == "chemberta":
            self._load_chemberta_model()
        elif self.config.model_type == "molformer":
            self._load_molformer_model()
        elif self.config.model_type == "molbert":
            self._load_molbert_model()
        elif self.config.model_type == "mole":
            self._load_mole_model()
        else:
            raise ValueError(f"Unrecognised model type: {self.config.model_type}")

        self._add_finetune_method()

        print(self.finetune_model.model_dir)

        self.finetune_model.restore()

        self._fit_laplace()

    def _instantiate_chemberta(self):
        self.finetune_model = Chemberta(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            config=self.chemberta_config,
            tokenizer_path=self.tokenizer_path,
            n_tasks=self.config.n_tasks,
            batch_size=self.config.batch_size,
        )

    def _load_chemberta_model(self):
        self.tokenizer_path = "DeepChem/ChemBERTa-77M-MTR"

        config_filename = create_file_path_string(
            ["configs", "chemberta_mlm_config.json"]
        )
        with open(config_filename) as f_in:
            self.chemberta_config: dict = json.load(f_in)

        self._instantiate_chemberta()

    def _instantiate_molformer(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolformerForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolformerForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

        self.finetune_model = MolformerDeepchem(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            load_path=self.molformer_load_path,
            from_pretrained_molformer=self.from_pretrained_molformer,
            n_tasks=self.config.n_tasks,
            config=self.molformer_config_dict,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
        )

    def _load_molformer_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.molformer_config = MolformerConfig(**pretrained_model_config)
        self.molformer_config_dict = pretrained_model_config

        self.molformer_load_path = os.path.join(
            create_file_path_string(
                ["pretrained_molformer", "pytorch_checkpoints"], local_path=True
            ),
            "N-Step-Checkpoint_3_30000.ckpt",
        )
        self.from_pretrained_molformer = True

        self._instantiate_molformer()

    def _instantiate_molbert(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolbertForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolbertForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

        self.finetune_model = MolbertDeepchem(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.molbert_config_dict,
            batch_size=self.config.batch_size,
            sequence_classifier=sequence_classifier,
            num_labels=self.config.num_labels,
        )

    def _load_molbert_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.molbert_config = MolbertConfig(**pretrained_model_config)
        self.molbert_config_dict = pretrained_model_config

        self._instantiate_molbert()

    def _instantiate_mole(self):
        if self.config.sequence_classifier_type == "default":
            sequence_classifier = MolEForSequenceClassification
        elif self.config.sequence_classifier_type == "likelihood":
            sequence_classifier = MolEForSequenceClassificationLikelihoodLoss
        else:
            raise ValueError(
                f"Unrecognised sequence classifier type: {self.config.sequence_classifier_type}"
            )

        self.finetune_model = MolEDeepchem(
            task=self.deepchem_task_type,
            model_dir=self.load_model_dir,
            n_tasks=self.config.n_tasks,
            config=self.mole_config_dict,
            batch_size=self.config.batch_size,
            num_labels=self.config.num_labels,
            sequence_classifier=sequence_classifier,
        )

    def _load_mole_model(self):
        pretrained_model_config = {
            **self.config.base_pretrained_model_config,
            **self.config.additional_pretrained_model_config,
        }
        self.mole_config_dict = pretrained_model_config
        self.mole_config = MolEExtraConfig(**pretrained_model_config)

        self._instantiate_mole()

    def _lora_method(self):
        task_type_string = self.config.base_lora_config.get("task_type", "SEQ_CLS")
        if task_type_string == "SEQ_CLS":
            task_type = TaskType.SEQ_CLS
        elif task_type_string == "SEQ_2_SEQ_LM":
            task_type = TaskType.SEQ_2_SEQ_LM
        else:
            raise ValueError("task_type not recognised")

        lora_config = {
            **self.config.base_lora_config,
            **self.config.additional_lora_config,
            "task_type": task_type,
        }
        peft_config = LoraConfig(**lora_config)

        self.finetune_model.model = get_peft_model(
            self.finetune_model.model, peft_config
        )

    def _classifier_only_method(self):
        raise RuntimeError(
            "Classifier only finetuning not implemented for Laplace-LoRA."
        )

    def _freeze_early_layers_method(self):
        raise RuntimeError(
            "Freeze early layers finetuning not implemented for Laplace-LoRA."
        )

    def _full_finetune_method(self):
        raise RuntimeError("Full finetuning not implemented for Laplace-LoRA.")

    def _add_finetune_method(self):
        self.FINETUNE_TYPE_TO_FINETUNE_METHOD_MAPPING: Dict[str, Callable[[], None]] = {
            "lora": self._lora_method,
            "classifier_only": self._classifier_only_method,
            "freeze_early_layers": self._freeze_early_layers_method,
            "full_finetune": self._full_finetune_method,
        }

        if (
            self.config.finetune_type
            in self.FINETUNE_TYPE_TO_FINETUNE_METHOD_MAPPING.keys()
        ):
            self.FINETUNE_TYPE_TO_FINETUNE_METHOD_MAPPING[self.config.finetune_type]()
        else:
            raise ValueError(
                f"Finetune type {self.config.finetune_type} not recognised."
            )

    def _fit_laplace(self):
        print("Fitting Laplace approximation...")

        for name, param in self.finetune_model.model.named_parameters():
            if param.requires_grad and "lora_" not in name:
                param.requires_grad_(False)

        train_dataloader = _deepchem_to_dataloader(
            self.train_dataset,
            self.finetune_model.tokenizer,
            batch_size=self.config.batch_size,
        )

        self.la = Laplace(
            _LogitsOnlyWrapper(self.finetune_model.model),
            likelihood="regression",
            subset_of_weights=self.config.laplace_subset_of_weights,
            hessian_structure=self.config.laplace_hessian_structure,
            prior_precision=self.config.laplace_prior_precision,
            sigma_noise=self.config.laplace_sigma_noise,
        )

        self.la.fit(train_dataloader)
        print("Laplace approximation fitted.")

        if self.config.laplace_optimize_prior:
            self.optimize_laplace_prior()

    def optimize_laplace_prior(
        self, method: str = "marglik", n_steps: int = 100, lr: float = 0.1
    ):
        if self.la is None:
            raise ValueError(
                "Laplace object not initialized. Call _load_finetune_model first."
            )

        print(f"Optimizing Laplace prior precision with {method}...")
        prior_prec = self.la.optimize_prior_precision(
            method=method,
            pred_type="glm",
            n_steps=n_steps,
            lr=lr,
        )
        print(f"Optimized prior precision: {prior_prec}")
        return prior_prec

    def _jacobians_manual(self, inputs):
        """Per-sample Jacobians via a loop, bypassing ASDL's batch_gradient.

        Needed for MolE because query_proj is called twice per forward (content path
        + relative-position path in disentangled_attention_bias), which causes ASDL's
        pre-forward hook to overwrite in_data with the smaller rel_embeddings shape.
        """
        params = [p for p in self.la.model.parameters() if p.requires_grad]
        f = self.la.model(**inputs)  # [batch, 1]
        batch_size = f.shape[0]
        Js = []
        for i in range(batch_size):
            grads = torch.autograd.grad(
                f[i, 0],
                params,
                retain_graph=(i < batch_size - 1),
                create_graph=False,
            )
            Js.append(torch.cat([g.flatten() for g in grads]))
        return torch.stack(Js).unsqueeze(1), f  # [batch, 1, n_params], [batch, 1]

    def predict_mean_and_std_on_batch(
        self, inputs, compute_std: bool = False, **kwargs
    ):
        self.finetune_model.model.eval()

        if not compute_std:
            with torch.no_grad():
                f_mu = self.la.model(**inputs)
            pred_mean = f_mu.squeeze(-1).detach()
            return UncertaintyRegressionPredictionOutput(mean=pred_mean, std=None)

        with torch.enable_grad():
            if self.config.laplace_manual_jacobians:
                Js, f_mu = self._jacobians_manual(inputs)
            else:
                Js, f_mu = self.la.backend.jacobians(inputs)
            f_var = self.la.functional_variance(Js)

        if f_mu.shape[-1] != 1:
            raise NotImplementedError(
                f"Multi-output regression not supported (n_outputs={f_mu.shape[-1]})"
            )

        pred_mean = f_mu.squeeze(-1).detach()
        pred_std = (
            f_var.squeeze(-1).squeeze(-1).detach() + self.la.sigma_noise**2
        ).sqrt()

        return UncertaintyRegressionPredictionOutput(mean=pred_mean, std=pred_std)

    def predict_mean_and_std_on_dataset(
        self,
        test_dataset,
        compute_std: bool = False,
        output_labels: bool = False,
        to_numpy: bool = False,
        predict_mean_and_std_kwargs={},
    ):
        dataloader = _deepchem_to_dataloader(
            test_dataset,
            self.finetune_model.tokenizer,
            batch_size=self.config.batch_size,
        )

        mean_values, std_values, label_values = [], [], []

        for batch in dataloader:
            device = next(self.finetune_model.model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            output = self.predict_mean_and_std_on_batch(
                inputs=batch, compute_std=compute_std, **predict_mean_and_std_kwargs
            )
            mean_values.append(output.mean)
            if compute_std:
                std_values.append(output.std)
            if output_labels:
                label_values.append(labels)

        mean = torch.cat(mean_values)
        std = torch.cat(std_values) if compute_std else None
        true_label = torch.cat(label_values) if output_labels else None

        if to_numpy:
            return UncertaintyRegressionPredictionOutputNumpy(
                mean=tensor_to_numpy(mean),
                std=tensor_to_numpy(std),
                label=tensor_to_numpy(true_label),
            )
        return UncertaintyRegressionPredictionOutput(
            mean=mean, std=std, label=true_label
        )


class UncertaintyQuantificationLaplaceLoraRegression(
    UncertaintyQuantificationLaplaceLoraSingle, UncertaintyQuantificationRegressionHF
):
    pass


def uncertainty_quantification_laplace_lora(
    metric_string_list: list[str] = ["ece"],
    problem_type: str = "regression",
    dataset_type: str = "test",
    **kwargs,
):
    """
    Convenience function to compute uncertainty quantification metrics using Laplace-LoRA.

    Args:
        metric_string_list: Metrics to compute (ece, nll, rmse)
        problem_type: 'regression' or 'classification' (only regression supported)
        dataset_type: 'test' or 'val' dataset to evaluate on
        **kwargs: Config parameters for UncertaintyQuantificationLaplaceLoraPEFTConfig

    Returns:
        Dictionary with computed metric scores
    """
    config = UncertaintyQuantificationLaplaceLoraPEFTConfig(**kwargs)

    if problem_type == "regression":
        uncertainty_quantifier = UncertaintyQuantificationLaplaceLoraRegression(config)
        scores = {}

        if "ece" in metric_string_list:
            ece = uncertainty_quantifier.regression_expected_calibration_error(
                dataset=(
                    uncertainty_quantifier.test_dataset
                    if dataset_type == "test"
                    else uncertainty_quantifier.val_dataset
                )
            )
            scores.update({"ece_score": ece})

        if "nll" in metric_string_list:
            nll = uncertainty_quantifier.gaussian_negative_log_likelihood(
                dataset=(
                    uncertainty_quantifier.test_dataset
                    if dataset_type == "test"
                    else uncertainty_quantifier.val_dataset
                )
            )
            scores.update({"nll_score": nll})

        if "rms" in metric_string_list:
            rms = uncertainty_quantifier.root_mean_squared_error(
                dataset=(
                    uncertainty_quantifier.test_dataset
                    if dataset_type == "test"
                    else uncertainty_quantifier.val_dataset
                )
            )
            scores.update({"rms_score": rms})

        return scores
    else:
        raise ValueError(f"Problem type {problem_type} not recognised.")
