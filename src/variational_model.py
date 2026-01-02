from dataclasses import dataclass, field

import torch

from typing import Optional

import numpy as np

from src.deepchem_hf_models import HuggingFaceModel

from src.model_molformer import MolformerDeepchem
from src.model_molbert import MolbertDeepchem
from src.model_mole import MolEDeepchem

from src.utils import return_short_time


@dataclass
class MFVIConfig:
    eps: float = (1e-5,)
    beta: float = 1 / 1000
    max_likelihood_epochs: int = 0
    beta_annealing_epochs: int = 0
    target_all_modules: bool = False
    mfvi_target_modules: list[str] = field(default_factory=list)
    time_kl: bool = False
    samples_per_prediction: int = 10


class VariationalModel(HuggingFaceModel):
    def __init__(
        self,
        mfvi_config: MFVIConfig,
        train_dataset_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mfvi_config = mfvi_config
        self.train_dataset_size = train_dataset_size
        if self.mfvi_config.time_kl:
            self.total_kl_time = 0
            self.kl_computations = 0

    @property
    def mfvi_beta(self):
        if self._global_epoch <= self.mfvi_config.max_likelihood_epochs:
            return 0.0
        if self.mfvi_config.beta_annealing_epochs == 0:
            annealing_factor = 1.0
        else:
            annealing_factor = (
                self._global_epoch - self.mfvi_config.max_likelihood_epochs + 1.0
            ) / self.mfvi_config.beta_annealing_epochs
        return min(annealing_factor, 1.0) * self.mfvi_config.beta

    def loss_function(
        self,
        nll: torch.Tensor,
        no_kl_divergence: bool = False,
        print_str: Optional[str] = None,
        **kwargs,
    ):
        scaling_factor = (
            float(self.train_dataset_size) / float(self.batch_size)
            if self.train_dataset_size is not None
            else 1
        )
        if no_kl_divergence:
            return nll * scaling_factor
        else:
            with return_short_time() as timer:
                if self.mfvi_beta != 0.0:
                    kl_divergence = self.mfvi_beta * self.total_kl_divergence()
                else:
                    kl_divergence = torch.tensor(0.0, device=nll.device)
            if self.mfvi_config.time_kl:
                self.total_kl_time += timer.time
                self.kl_computations += 1
            with torch.no_grad():
                if print_str is not None and self.get_global_step() % 10 == 0:
                    print(
                        f"{print_str}; Likelihood: {np.round(nll.cpu().detach().numpy(), decimals=4)}; KL: {np.round(kl_divergence.cpu().detach().numpy(),decimals=4)}"
                    )
                if self.wandb_logger is not None:
                    if self.get_global_step() % 10:
                        self.wandb_logger.log_data(
                            {"beta": self.mfvi_beta}, step=self.get_global_step()
                        )
            return nll * scaling_factor + kl_divergence

    def _test_loss_function(
        self,
        batch_loss: torch.Tensor,
        is_checkpoint_evaluation: bool = False,
        **kwargs,
    ):
        return self.loss_function(
            nll=batch_loss,
            no_kl_divergence=is_checkpoint_evaluation,
            **kwargs,
        )

    def total_kl_divergence(self):
        kl = 0.0
        num_parameters = 0.0
        for module in self.model.modules():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence()
                num_parameters += module.num_parameters
        return kl / num_parameters

    @property
    def mean_kl_time(self):
        if self.mfvi_config.time_kl:
            return self.total_kl_time / self.kl_computations
        else:
            return "KL not timed"

    def _predict_output_values(self, inputs, **kwargs):
        with torch.no_grad():
            self.model.eval()
            output_values = self.model(**inputs).get("logits")

            if self.mfvi_config.samples_per_prediction > 1:
                for _ in range(self.mfvi_config.samples_per_prediction - 1):
                    output_values += self.model(**inputs).get("logits")

            return output_values / (float(self.mfvi_config.samples_per_prediction))


class VariationalMolformerSingle(MolformerDeepchem, VariationalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class VariationalMolbertSingle(MolbertDeepchem, VariationalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class VariationalMoleSingle(MolEDeepchem, VariationalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
