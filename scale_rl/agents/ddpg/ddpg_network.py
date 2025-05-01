import torch
import torch.nn as nn

from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.policies import TanhPolicy
from scale_rl.networks.layers import SimbaBlock


class DDPGActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        dtype: torch.dtype,
        activ='ReLU'
    ):
        super().__init__()
        self.dtype = dtype

        self.encoder = SimbaBlock(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            activ=activ
        )
        self.predictor=TanhPolicy(hidden_dim, action_dim, dtype=dtype)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(dtype=self.dtype)
        z = self.encoder(observations)
        action = self.predictor(z)

        return action


class DDPGCritic(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        activ='ELU'
    ):
        super().__init__()
        self.dtype = dtype

        self.encoder = SimbaBlock(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            activ=activ
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


class DDPGClippedDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        activ='ELU',
        num_qs=2
    ):
        super().__init__()
        
        self.critics = nn.ModuleList([
            DDPGCritic(
                num_blocks=num_blocks,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dtype=dtype,
                activ=activ
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
    