import torch
import torch.nn as nn

from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.policies import TanhPolicy
from scale_rl.networks.layers import SimbaBlock
from scale_rl.networks.utils import orthogonal_init_


class Embedding(SimbaBlock):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dtype: torch.dtype,
        activ='ELU'
    ):
        super().__init__(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            activ=activ
        )
        self.head = nn.Linear(hidden_dim, output_dim, dtype=dtype)

        orthogonal_init_(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.head(x)
        return x


class MRDDPGEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        zs_dim: int,
        za_dim: int,
        zsa_dim: int,
        num_bins: int,
        num_blocks: int,
        dtype: torch.dtype,
        activ='ELU'
    ):
        super().__init__()
        self.zs_dim = zs_dim
        
        self.zs = Embedding(
            num_blocks=num_blocks,
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=zs_dim,
            dtype=dtype,
            activ='ELU'
        )
        self.za = self.mlp_za
        self.fc = nn.Linear(action_dim, za_dim, dtype=dtype)
        self.zsa = Embedding(
            num_blocks=num_blocks,
            input_dim=zs_dim+za_dim,
            hidden_dim=hidden_dim,
            output_dim=zsa_dim,
            dtype=dtype,
            activ='ELU'
        )
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.activ = getattr(nn, activ)()

        orthogonal_init_(self.model)

    def forward(
        self,
        zs: torch.Tensor,
        za: torch.Tensor
    ) -> torch.Tensor:
        zsa = self.zsa(torch.cat([zs, za], 1))
        return zsa
    
    def mlp_za(
        self,
        action: torch.Tensor
    ):
        return self.activ(self.fc(action))

    def model_all(
        self,
        zs: torch.Tensor,
        za: torch.Tensor
    ):  
        zsa = self.zsa(torch.cat([zs, za], 1))
        dzr = self.model(zsa)
        return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward


class MRDDPGActor(nn.Module):
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

        self.encoder = SimbaBlock(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            activ=activ
        )
        self.predictor=TanhPolicy(hidden_dim, action_dim, dtype=dtype)

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(zs)
        action = self.predictor(z)

        return action


class MRDDPGCritic(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        activ='ELU'
    ):
        super().__init__()

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
        zsa: torch.Tensor
    ) -> torch.Tensor:
        z = self.encoder(zsa)
        q = self.predictor(z)

        return q


class MRDDPGClippedDoubleCritic(nn.Module):
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
            MRDDPGCritic(
                num_blocks=num_blocks,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dtype=dtype,
                activ=activ
            ) for _ in range(num_qs)
        ])

    def forward(
        self,
        zsa: torch.Tensor
    ) -> torch.Tensor:
        qs = [critic(zsa) for critic in self.critics]
        qs = torch.stack(qs, dim=0)

        return qs
