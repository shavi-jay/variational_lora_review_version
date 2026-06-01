"""
BLoB (Bayesian Low-Rank Adaptation by Backpropagation) layer.
Reference: Wang et al., NeurIPS 2024. Upstream: github.com/Wang-ML-Lab/bayesian-peft
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.variational_model import MFVIConfig
from src.variational_inference.variational_layer import kl_divergence_explicit


@dataclass
class BLoBConfig(MFVIConfig):
    prior_std: float = 0.2
    init_eps: float = 0.05
    use_flipout: bool = True
    kl_reweighting: bool = True
    inference_no_sample: bool = False
    mean_kl: bool = False  # BLoB sums raw KL (not divided by num_parameters)


class BLoBLoraLinear(nn.Module):
    """
    Variational replacement for lora_A.default (asymmetric Bayesianization, paper §3.1).

    Parameterizes q(A) = N(M, Ω) where Ω = G² (quadratic std, paper §3.3).
    Forward returns M @ x + flipout_perturbation(x) so that after lora_B the net
    effect matches Eqn. 12.  Set self.sampling = False for BLoB(N=0) inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_init: Optional[torch.Tensor] = None,
        prior_std: float = 0.2,
        init_eps: float = 0.05,
        use_flipout: bool = True,
        EPS: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_flipout = use_flipout
        self.sampling = True

        self.weight_mean = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        # G parameter; weight_std property returns G² (quadratic parameterization)
        self._std_param = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if weight_init is not None:
            self.weight_mean.data = weight_init.clone()
        else:
            nn.init.kaiming_uniform_(self.weight_mean, a=np.sqrt(5))

        # G ~ U(ε/√2, ε) as in upstream BLoB init
        nn.init.uniform_(self._std_param, init_eps / (2 ** 0.5), init_eps)

        self.register_buffer("EPS", torch.tensor(EPS))
        self.register_buffer("prior_weight_mean", torch.zeros_like(self.weight_mean))
        self.register_buffer(
            "prior_weight_std", torch.full_like(self.weight_mean, prior_std)
        )

    @property
    def weight_std(self):
        return torch.clamp(self._std_param ** 2, min=self.EPS)

    @property
    def weight(self):
        return self.weight_mean

    @property
    def num_parameters(self):
        return self.weight_mean.numel() + self._std_param.numel()

    def kl_divergence(self):
        return kl_divergence_explicit(
            self.weight_mean,
            self.weight_std,
            self.prior_weight_mean,
            self.prior_weight_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.sampling:
            return F.linear(x, self.weight_mean)

        if not self.use_flipout:
            w = self.weight_mean + self.weight_std * torch.randn_like(self.weight_mean)
            return F.linear(x, w)

        # LoRA Flipout (paper §3.4): mean path + per-example perturbation
        mean_out = F.linear(x, self.weight_mean)
        lora_noise_a = self.weight_std * torch.randn_like(self.weight_mean)

        # r_A, s_A: per-example sign matrices in {-1, +1}
        r_A = torch.sign(torch.rand_like(x) - 0.5)
        s_A = torch.sign(torch.rand_like(mean_out) - 0.5)

        if x.dim() == 2:
            # (batch, in_features)
            noise = ((x * r_A) @ lora_noise_a.T) * s_A
        else:
            # (batch, seq, in_features)
            noise = torch.matmul(x * r_A, lora_noise_a.T) * s_A

        return mean_out + noise
