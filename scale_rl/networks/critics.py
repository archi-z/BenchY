"""
Implementation of commonly used critics that can be shared across agents.
"""
import torch
import torch.nn as nn

from scale_rl.networks.utils import orthogonal_init_


class LinearCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dtype: torch.dtype,
        kernel_init_scale=1.0
    ):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, 1, dtype=dtype)

        orthogonal_init_(self.fc, kernel_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

