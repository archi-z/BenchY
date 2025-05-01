import math

import torch
import torch.nn as nn

from scale_rl.networks.utils import he_normal_init_, orthogonal_init_


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        activ='ReLU'
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.activ = getattr(nn, activ)()

        orthogonal_init_(self.fc1, gain=math.sqrt(2))
        orthogonal_init_(self.fc2, gain=math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.activ(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype,
        activ='ReLU'
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*4, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim, dtype=dtype)
        self.activ = getattr(nn, activ)()

        he_normal_init_(self.fc1)
        he_normal_init_(self.fc2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        
        return res + x


class SimbaBlock(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        activ='ReLU'
    ):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dtype=dtype, activ=activ) for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)

        orthogonal_init_(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.layer_norm(x)
        
        return x
