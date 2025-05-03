from typing import Dict, Tuple, Union
from dataclasses import dataclass
import copy

import gymnasium as gym
import numpy as np
import torch

from scale_rl.buffers import Batch
from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.mrsac.mrsac_network import (
    MRSACEncoder,
    MRSACActor,
    MRSACCritic,
    MRSACClippedDoubleCritic,
    MRSACTemperature
)
from scale_rl.agents.mrsac.mrsac_update import Update


@dataclass(frozen=True)
class MRSACConfig:
    device: str
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    zs_dim: int
    za_dim: int
    zsa_dim: int
    encoder_horizon: int
    encoder_num_blocks: int
    encoder_hidden_dim: int
    encoder_activ: str
    encoder_learning_rate: float
    encoder_weight_decay: float

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_activ: str
    actor_learning_rate: float
    actor_weight_decay: float

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_activ: str
    critic_learning_rate: float
    critic_weight_decay: float
    critic_use_cdq: bool

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    encoder_update_freq: int
    target_tau: float
    gamma: float
    n_step: int

    dyn_weight: int
    reward_weight: float
    done_weight: float

    num_bins: int
    lower: float
    upper: float

    mixed_precision: bool


def _init_mrsac_networks(
    observation_dim: int,
    action_dim: int,
    cfg: MRSACConfig,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[
        MRSACEncoder,
        MRSACActor,
        Union[MRSACCritic, MRSACClippedDoubleCritic],
        MRSACEncoder,
        Union[MRSACCritic, MRSACClippedDoubleCritic],
        MRSACTemperature
    ]:

    encoder = MRSACEncoder(
        state_dim=observation_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        action_dim=action_dim,
        zs_dim=cfg.zs_dim,
        za_dim=cfg.za_dim,
        zsa_dim=cfg.zsa_dim,
        num_bins=cfg.num_bins,
        num_blocks=cfg.encoder_num_blocks,
        dtype=dtype,
        activ=cfg.encoder_activ
    ).to(device)

    actor = MRSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=cfg.zs_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
        dtype=dtype,
        activ=cfg.actor_activ
    ).to(device)
    
    if cfg.critic_use_cdq:
        critic = MRSACClippedDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            input_dim=cfg.zsa_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=dtype,
            activ=cfg.critic_activ
        ).to(device)

    else:
        critic = MRSACCritic(
            num_blocks=cfg.critic_num_blocks,
            input_dim=cfg.zsa_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=dtype,
            activ=cfg.critic_activ
        ).to(device)

    target_encoder = copy.deepcopy(encoder)
    target_critic = copy.deepcopy(critic)

    temperature = MRSACTemperature(cfg.temp_initial_value).to(device)

    return encoder, actor, critic, target_encoder, target_critic, temperature


class MRSACAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: MRSACConfig,
    ):
        super().__init__(
            observation_space,
            action_space,
            cfg
        )

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]
        cfg['temp_target_entropy'] = cfg['temp_target_entropy_coef'] * self._action_dim
        self._cfg = MRSACConfig(**cfg)
        self.device = torch.device(self._cfg.device)
        self.dtype = torch.float16 if self._cfg.mixed_precision else torch.float32

        self._init_network()

        self._encoder_optimizer = torch.optim.AdamW(
            self._encoder.parameters(),
            lr=self._cfg.encoder_learning_rate,
            weight_decay=self._cfg.encoder_weight_decay
        )
        self._actor_optimizer = torch.optim.AdamW(
            self._actor.parameters(),
            lr=self._cfg.actor_learning_rate,
            weight_decay=self._cfg.actor_weight_decay
        )
        self._critic_optimizer = torch.optim.AdamW(
            self._critic.parameters(),
            lr=self._cfg.critic_learning_rate,
            weight_decay=self._cfg.critic_weight_decay
        )
        self._temp_optimizer = torch.optim.AdamW(
            self._temperature.parameters(),
            lr=self._cfg.temp_learning_rate,
            weight_decay=self._cfg.temp_weight_decay
        )

        self.update_batch = Update(
            encoder=self._encoder,
            actor=self._actor,
            critic=self._critic,
            temperature=self._temperature,
            target_encoder=self._target_encoder,
            target_critic=self._target_critic,
            encoder_optimizer=self._encoder_optimizer,
            actor_optimizer=self._actor_optimizer,
            critic_optimizer=self._critic_optimizer,
            temp_optimizer=self._temp_optimizer,
            cfg=self._cfg
        )

    def _init_network(self):
        (
            self._encoder,
            self._actor,
            self._critic,
            self._target_encoder,
            self._target_critic,
            self._temperature,
        ) = _init_mrsac_networks(self._observation_dim, self._action_dim, self._cfg, self.device, self.dtype)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        with torch.no_grad():
            # current timestep observation is "next" observations from the previous timestep
            observations = torch.as_tensor(prev_timestep["next_observation"], device=self.device, dtype=self.dtype)
            zs = self._encoder.zs(observations)
            dist = self._actor(zs, temperature)
            actions = dist.sample()

        return actions

    def update(
        self,
        update_step: int,
        batch: Batch
    ) -> Dict:
        cur_obs = torch.as_tensor(batch["observation"], device=self.device, dtype=self.dtype)
        actions = torch.as_tensor(batch["action"], device=self.device, dtype=self.dtype)
        rewards = torch.as_tensor(batch["reward"], device=self.device, dtype=self.dtype)
        terminated = torch.as_tensor(batch["terminated"], device=self.device, dtype=self.dtype)
        next_obs = torch.as_tensor(batch["next_observation"], device=self.device, dtype=self.dtype)

        update_info = self.update_batch.update_ac_networks(
            update_step=update_step,
            cur_obs=cur_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            next_obs=next_obs
        )

        return update_info
    
    def update_encoder(
        self,
        batch: Batch
    ) -> Dict:
        cur_obs = torch.as_tensor(batch["observation"], device=self.device, dtype=self.dtype)
        actions = torch.as_tensor(batch["action"], device=self.device, dtype=self.dtype)
        rewards = torch.as_tensor(batch["reward"], device=self.device, dtype=self.dtype)
        terminated = torch.as_tensor(batch["terminated"], device=self.device, dtype=self.dtype)
        next_obs = torch.as_tensor(batch["next_observation"], device=self.device, dtype=self.dtype)

        encoder_info = self.update_batch.update_encoder(
            cur_obs=cur_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            next_obs=next_obs
        )
        
        return encoder_info

    def update_target_encoder(self):
        self._target_encoder.load_state_dict(self._encoder.state_dict())
