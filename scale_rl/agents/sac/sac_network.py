import math

import torch
import torch.nn as nn
from torch.distributions import Distribution

from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.layers import MLPBlock, ResidualBlock
from scale_rl.networks.policies import NormalTanhPolicy
from scale_rl.networks.utils import orthogonal_init_


class SACEncoder(nn.Module):
    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype
    ):
        super().__init__()
        self.block_type = block_type

        if block_type == "mlp":
            self.mlp = MLPBlock(
                input_dim,
                hidden_dim,
                dtype=dtype
            )

        elif block_type == "residual":
            self.fc = nn.Linear(input_dim, hidden_dim, dtype=dtype)
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dtype=dtype) for _ in range(num_blocks)
            ])
            self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)

            orthogonal_init_(self.fc)

        else:
            raise NotImplementedError(f"Unsupported block_type: {self.block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == "mlp":
            x = self.mlp(x)

        elif self.block_type == "residual":
            x = self.fc(x)
            for block in self.residual_blocks:
                x = block(x)
            x = self.layer_norm(x)
        
        return x


class SACActor(nn.Module):
    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        dtype: torch.dtype
    ):
        super().__init__()
        self.dtype = dtype

        self.encoder = SACEncoder(
            block_type=block_type,
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype
        )
        self.predictor=NormalTanhPolicy(hidden_dim, action_dim, dtype=dtype)

    def forward(
        self,
        observations: torch.Tensor,
        temperature=1.0
    ) -> Distribution:
        observations = observations.to(dtype=self.dtype)
        z = self.encoder(observations)
        dist = self.predictor(z, temperature)

        return dist


class SACCritic(nn.Module):
    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype
    ):
        super().__init__()
        self.dtype = dtype
        
        self.encoder = SACEncoder(
            block_type=block_type,
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype
        )
        self.predictor = LinearCritic(
            input_dim=hidden_dim,
            dtype=dtype
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((observations, actions), dim=1)
        features = features.to(dtype=self.dtype)
        z = self.encoder(features)
        q = self.predictor(z)

        return q


class SACClippedDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """
    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        num_qs=2
    ):
        super().__init__()

        self.critics = nn.ModuleList([
            SACCritic(
                block_type=block_type,
                num_blocks=num_blocks,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dtype=dtype
            ) for _ in range(num_qs)
        ])

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        qs = [critic(observations, actions) for critic in self.critics]
        qs = torch.stack(qs, dim=0)

        return qs


class SACTemperature(nn.Module):
    def __init__(
        self,
        initial_value=1.0
    ):
        super().__init__()

        self.log_temp = nn.Parameter(
            torch.full((), math.log(initial_value))
        )

    def forward(self) -> torch.Tensor:
        return self.log_temp.exp()