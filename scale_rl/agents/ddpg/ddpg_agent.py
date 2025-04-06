from typing import Dict, Tuple, Union
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.ddpg.ddpg_network import (
    DDPGActor,
    DDPGClippedDoubleCritic,
    DDPGCritic,
)
from scale_rl.agents.ddpg.ddpg_update import update_ddpg_networks
from scale_rl.common.colored_noise import ColoredNoiseProcess
from scale_rl.common.scheduler import linear_decay_scheduler


@dataclass(frozen=True)
class DDPGConfig:
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

    target_tau: float
    gamma: float
    n_step: int

    exp_noise_color: float
    exp_noise_scheduler: str
    exp_noise_decay_period: int
    exp_noise_std_init: float
    exp_noise_std_final: float

    mixed_precision: bool
    device: str


def _init_ddpg_networks(
    observation_dim: int,
    action_dim: int,
    cfg: DDPGConfig,
    device: torch.device
) -> Tuple[
    DDPGActor,
    Union[DDPGCritic, DDPGClippedDoubleCritic],
    Union[DDPGCritic, DDPGClippedDoubleCritic],
    ]:
    compute_dtype = torch.float16 if cfg.mixed_precision else torch.float32

    actor = DDPGActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            input_dim=observation_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype
        ).to(device)
    
    if cfg.critic_use_cdq:
        critic = DDPGClippedDoubleCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype
        ).to(device)
        target_critic = DDPGClippedDoubleCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype
        ).to(device)

    else:
        critic = DDPGCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype
        ).to(device)
        target_critic = DDPGCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype
        ).to(device)

    return actor, critic, target_critic


class DDPGAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: DDPGConfig
    ):
        super().__init__(
            observation_space,
            action_space,
            cfg
        )

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]
        self._cfg = DDPGConfig(**cfg)
        self._device = torch.device(self._cfg.device)
    
        self._init_network()
        self._init_exp_scheduler()
        self._init_action_noise()

        self._actor_optimizer = torch.optim.AdamW(self._actor.parameters(),lr=self._cfg.actor_learning_rate)
        self._critic_optimizer = torch.optim.AdamW(self._critic.parameters(), lr=self._cfg.critic_learning_rate)

    def _init_network(self):
        (
            self._actor,
            self._critic,
            self._target_critic,
        ) = _init_ddpg_networks(self._observation_dim, self._action_dim, self._cfg, self._device)

    def _init_exp_scheduler(self):
        if self._cfg.exp_noise_scheduler == "linear":
            self._exp_scheduler = linear_decay_scheduler(
                decay_period=self._cfg.exp_noise_decay_period,
                initial_value=self._cfg.exp_noise_std_init,
                final_value=self._cfg.exp_noise_std_final,
            )

        else:
            raise NotImplementedError(f"Unsupported exp_noise_scheduler: {self._cfg.exp_noise_scheduler}")

    def _init_action_noise(self):
        self._action_noise = []

        # each train environment has a separate noise schedule.
        for _ in range(self._cfg.num_train_envs):
            self._action_noise.append(
                ColoredNoiseProcess(
                    beta=self._cfg.exp_noise_color,
                    size=(self._action_dim, self._cfg.max_episode_steps)
                )
            )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool
    ) -> torch.Tensor:
        if training:
            # reinitialize the noise if env was reinitialized
            prev_terminated = prev_timestep["terminated"]
            prev_truncated = prev_timestep["truncated"]
            for env_idx in range(self._cfg.num_train_envs):
                done = prev_terminated[env_idx] or prev_truncated[env_idx]
                if done:
                    self._action_noise[env_idx].reset()

            action_noise = np.array(
                [noise_sampler.sample() for noise_sampler in self._action_noise]
            )

            # scale the action noise with exp_noise_std
            self._noise_std = noise_std = self._exp_scheduler(interaction_step)
            action_noise = action_noise * noise_std

        else:
            action_noise = 0.0

        with torch.no_grad():
            # current timestep observation is "next" observations from the previous timestep
            observations = torch.as_tensor(prev_timestep["next_observation"]).to(self._device)
            actions = self._actor(observations)
            action_noise = torch.as_tensor(action_noise, device=self._device)
            actions = torch.clamp(actions+action_noise, -1.0, 1.0)

        return actions

    def update(
        self,
        update_step: int,
        batch: Dict[str, np.ndarray]
    ) -> Dict:
        for key, value in batch.items():
            batch[key] = torch.as_tensor(value)

        (
            self._actor,
            self._critic,
            self._target_critic,
            update_info
        ) = update_ddpg_networks(
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            actor_optimizer=self._actor_optimizer,
            critic_optimizer=self._critic_optimizer,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            target_tau=self._cfg.target_tau,
            critic_use_cdq=self._cfg.critic_use_cdq,
            noise_std=self._noise_std
        )

        for key, value in update_info.items():
            update_info[key] = float(value)
        
        return update_info
