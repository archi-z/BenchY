"""
Implementation of commonly used policies that can be shared across agents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from scale_rl.networks.utils import orthogonal_init_


class NormalTanhPolicy(nn.Module):
    pass


class TanhPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        kernel_init_scale=1.0
    ):
        super().__init__()

        self.fc = nn.Linear(hidden_dim, action_dim)

        orthogonal_init_(self.fc, gain=kernel_init_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        actions = self.fc(features)
        return F.tanh(actions)
