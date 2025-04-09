from typing import Dict, Tuple, Union
from dataclasses import dataclass
import copy

import gymnasium as gym
import numpy as np
import torch

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.sac.sac_network import (
    SACActor,
    SACCritic,
    SACClippedDoubleCritic,
    SACTemperature
)
from scale_rl.agents.sac.sac_update import update_sac_networks


@dataclass(frozen=True)
class SACConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    actor_block_type: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate: float
    actor_weight_decay: float

    critic_block_type: str
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate: float
    critic_weight_decay: float
    critic_use_cdq: bool

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    target_tau: float
    gamma: float
    n_step: int

    mixed_precision: bool
    device: str


def _init_sac_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SACConfig,
    device: torch.device
) -> Tuple[
        SACActor,
        Union[SACCritic, SACClippedDoubleCritic],
        Union[SACCritic, SACClippedDoubleCritic],
        SACTemperature
    ]:
    compute_dtype = torch.float16 if cfg.mixed_precision else torch.float32

    actor = SACActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            input_dim=observation_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype
        ).to(device)
    
    if cfg.critic_use_cdq:
        critic = SACClippedDoubleCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype
        ).to(device)

    else:
        critic = SACCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype
        ).to(device)

    target_critic = copy.deepcopy(critic)

    temperature = SACTemperature(cfg.temp_initial_value).to(device)

    return actor, critic, target_critic, temperature


class SACAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: SACConfig,
    ):
        super().__init__(
            observation_space,
            action_space,
            cfg
        )

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]
        cfg['temp_target_entropy'] = cfg['temp_target_entropy_coef'] * self._action_dim
        self._cfg = SACConfig(**cfg)
        self._device = torch.device(self._cfg.device)

        self._init_network()

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

    def _init_network(self):
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_sac_networks(self._observation_dim, self._action_dim, self._cfg, self._device)

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
            observations = torch.as_tensor(prev_timestep["next_observation"]).to(self._device)
            dist = self._actor(observations, temperature)
            actions = dist.sample()

        return actions

    def update(
        self,
        update_step: int,
        batch: Dict[str, np.ndarray]
    ) -> Dict:
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = update_sac_networks(
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            actor_optimizer=self._actor_optimizer,
            critic_optimizer=self._critic_optimizer,
            temp_optimizer=self._temp_optimizer,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy
        )

        for key, value in update_info.items():
            update_info[key] = float(value)

        return update_info
