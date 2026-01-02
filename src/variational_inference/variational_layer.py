"""
Adapting MFVILinear module in ProbAI 2022 tutorial
"""

from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class MFVILinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        prior_weight_std=0.01,
        prior_bias_std=0.01,
        init_std=0.0005,
        sqrt_width_scaling=False,
        device=None,
        dtype=None,
        EPS=1e-5,
        weight_init: Optional[torch.Tensor] = None,
        bias_init: Optional[torch.Tensor] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MFVILinear, self).__init__()
        self.in_features = in_features  # dimension of network layer input
        self.out_features = out_features  # dimension of network layer output
        self.bias = bias  # is bias term in layer

        # define the trainable variational parameters for q distribtuion
        # first define and initialise the mean parameters
        self.weight_mean = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias_mean", None)
        self._weight_std_param = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self._bias_std_param = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("_bias_std_param", None)
        self.reset_parameters(init_std)
        if weight_init is not None:
            if weight_init.shape == self.weight_mean.shape:
                self.weight_mean.data = weight_init
        if bias and bias_init is not None:
            if bias_init.shape == self.bias_mean.shape:
                self.bias_mean.data = bias_init
        self.EPS = nn.Parameter(torch.Tensor(EPS, device=device), requires_grad=False)

        # define the prior parameters (for prior p, assume the mean is 0)
        prior_mean = 0.0
        if sqrt_width_scaling:  # prior variance scales as 1/dim_in
            prior_weight_std /= self.in_features**0.5

        # prior parameters are registered as constants
        self.register_buffer(
            "prior_weight_mean", torch.full_like(self.weight_mean, prior_mean)
        )
        self.register_buffer(
            "prior_weight_std",
            torch.full_like(self._weight_std_param, prior_weight_std),
        )
        self.register_buffer(
            "prior_bias_mean",
            torch.full_like(self.bias_mean, prior_mean) if bias else None,
        )
        self.register_buffer(
            "prior_bias_std",
            torch.full_like(self._bias_std_param, prior_bias_std) if bias else None,
        )

    def extra_repr(self):
        s = "dim_in={}, dim_in={}, bias={}".format(
            self.in_features, self.out_features, self.bias
        )
        weight_std = self.prior_weight_std.data.flatten()[0]
        if torch.allclose(weight_std, self.prior_weight_std):
            s += f", weight prior std={weight_std.item():.2f}"
        if self.bias:
            bias_std = self.prior_bias_std.flatten()[0]
            if torch.allclose(bias_std, self.prior_bias_std):
                s += f", bias prior std={bias_std.item():.2f}"
        return s

    def reset_parameters(self, init_std=0.05):
        _init_std_param = np.log(init_std)
        nn.init.kaiming_uniform_(self.weight_mean, a=np.sqrt(5))
        self._weight_std_param.data = torch.full_like(self.weight_mean, _init_std_param)
        bound = self.in_features**-0.5
        if self.bias:
            nn.init.uniform_(self.bias_mean, -bound, bound)
            self._bias_std_param.data = torch.full_like(self.bias_mean, _init_std_param)

    # define the q distribution standard deviations with property decorator
    @property
    def weight_std(self):
        return torch.clamp(torch.exp(self._weight_std_param), min=self.EPS)

    @property
    def bias_std(self):
        if self.bias:
            return torch.clamp(torch.exp(self._bias_std_param), min=self.EPS)
        else:
            return None

    @property
    def weight(self):
        return self.weight_mean

    @property
    def num_parameters(self):
        if self.bias:
            return self.weight_mean.numel() + self.bias_mean.numel()
        else:
            return self.weight_mean.numel()

    # KL divergence KL[q||p] between two Gaussians
    def kl_divergence(self):
        return self.kl_divergence_new()

    # forward pass with Monte Carlo (MC) sampling
    def forward(self, input):
        weight = self._normal_sample(self.weight_mean, self.weight_std)
        bias = self._normal_sample(self.bias_mean, self.bias_std) if self.bias else None
        return F.linear(input, weight, bias)

    def _normal_sample(self, mean, std):
        return mean + torch.randn_like(mean) * std

    def kl_divergence_original(self):
        q_weight = dist.Normal(self.weight_mean, self.weight_std)
        p_weight = dist.Normal(self.prior_weight_mean, self.prior_weight_std)
        kl = dist.kl_divergence(q_weight, p_weight).sum()
        if self.bias:
            q_bias = dist.Normal(self.bias_mean, self.bias_std)
            p_bias = dist.Normal(self.prior_bias_mean, self.prior_bias_std)
            kl += dist.kl_divergence(q_bias, p_bias).sum()
        return kl

    def kl_divergence_new(self):
        kl = kl_divergence_explicit(
            self.weight_mean,
            self.weight_std,
            self.prior_weight_mean,
            self.prior_weight_std,
        )
        if self.bias:
            kl += kl_divergence_explicit(
                self.bias_mean, self.bias_std, self.prior_bias_mean, self.prior_bias_std
            )
        return kl


def kl_divergence_explicit(
    p_mean: torch.Tensor, p_std: torch.Tensor, q_mean: torch.Tensor, q_std: torch.Tensor
):
    """Compute KL(p||q) for p,q univariate Gaussian distributions"""
    log_term = torch.log(q_std) - torch.log(p_std)
    frac_term = (
        0.5
        * (torch.pow(p_mean - q_mean, 2) + torch.pow(p_std, 2))
        / torch.pow(q_std, 2)
    )
    kl = log_term + frac_term - 0.5
    return torch.sum(kl)
