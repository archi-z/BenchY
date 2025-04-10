"""
Implementation of commonly used policies that can be shared across agents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    MultivariateNormal,
    TransformedDistribution,
    TanhTransform
)

from scale_rl.networks.utils import orthogonal_init_


class NormalTanhPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        dtype: torch.dtype,
        kernel_init_scale=1.0,
        state_dependent_std=True,
        log_std_min=-10.0,
        log_std_max=2.0
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.dtype = dtype

        self.mean_layer = nn.Linear(hidden_dim, action_dim, dtype=dtype)
        orthogonal_init_(self.mean_layer, gain=kernel_init_scale)

        if state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dim, action_dim, dtype=dtype)
            orthogonal_init_(self.log_std_layer, gain=kernel_init_scale)

        else:
            self.log_std_param = nn.Parameter(torch.zeros(action_dim, dtype=dtype))

    def forward(
        self,
        features: torch.Tensor,
        temperature=1.0
    ) -> Distribution:
        means = self.mean_layer(features)

        if self.state_dependent_std:
            log_stds = self.log_std_layer(features)
        else:
            log_stds = self.log_std_param

        # suggested by Ilya for stability
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + F.tanh(log_stds)
        )
        std = log_stds.exp() * temperature

        dist = MultivariateNormal(
            loc=means,
            scale_tril=torch.diag_embed(std),
            validate_args=False
        )
        dist = TransformedDistribution(dist, TanhTransform(cache_size=1))

        return dist


class TanhPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        dtype: torch.dtype,
        kernel_init_scale=1.0
    ):
        super().__init__()

        self.fc = nn.Linear(hidden_dim, action_dim, dtype=dtype)

        orthogonal_init_(self.fc, gain=kernel_init_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        actions = self.fc(features)
        return F.tanh(actions)
