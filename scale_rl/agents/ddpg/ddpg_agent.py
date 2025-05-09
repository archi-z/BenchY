from typing import Dict, Tuple, Union
from dataclasses import dataclass
import copy

import gymnasium as gym
import numpy as np
import torch

from scale_rl.buffers import Batch
from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.ddpg.ddpg_network import (
    DDPGActor,
    DDPGCritic,
    DDPGClippedDoubleCritic
)
from scale_rl.agents.ddpg.ddpg_update import Update
from scale_rl.agents.ddpg.ddpg_metric import get_actor_metrics, get_critic_metrics
from scale_rl.common.colored_noise import ColoredNoiseProcess
from scale_rl.common.scheduler import linear_decay_scheduler
from scale_rl.networks.metrics import count_parameters, format_params_str


@dataclass(frozen=True)
class DDPGConfig:
    device: str
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

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

    target_tau: float
    gamma: float
    n_step: int

    exp_noise_color: float
    exp_noise_scheduler: str
    exp_noise_decay_period: int
    exp_noise_std_init: float
    exp_noise_std_final: float

    mixed_precision: bool


def _init_ddpg_networks(
    observation_dim: int,
    action_dim: int,
    cfg: DDPGConfig,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[
        DDPGActor,
        Union[DDPGCritic, DDPGClippedDoubleCritic],
        Union[DDPGCritic, DDPGClippedDoubleCritic],
    ]:

    actor = DDPGActor(
            num_blocks=cfg.actor_num_blocks,
            input_dim=observation_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=dtype,
            activ=cfg.actor_activ
        ).to(device)
    
    if cfg.critic_use_cdq:
        critic = DDPGClippedDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=dtype,
            activ=cfg.critic_activ
        ).to(device)

    else:
        critic = DDPGCritic(
            num_blocks=cfg.critic_num_blocks,
            input_dim=observation_dim+action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=dtype,
            activ=cfg.critic_activ
        ).to(device)

    target_critic = copy.deepcopy(critic)

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
        self.device = torch.device(self._cfg.device)
        self.dtype = torch.float16 if self._cfg.mixed_precision else torch.float32
    
        self._init_network()
        self._init_exp_scheduler()
        self._init_action_noise()

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

        self.update_batch = Update(
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            actor_optimizer=self._actor_optimizer,
            critic_optimizer=self._critic_optimizer,
            cfg=self._cfg
        )

    def _init_network(self):
        (
            self._actor,
            self._critic,
            self._target_critic,
        ) = _init_ddpg_networks(self._observation_dim, self._action_dim, self._cfg, self.device, self.dtype)

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
            observations = torch.as_tensor(prev_timestep["next_observation"], device=self.device, dtype=self.dtype)
            actions = self._actor(observations)
            action_noise = torch.as_tensor(action_noise, device=self.device, dtype=self.dtype)
            actions = (actions+action_noise).clamp(-1.0, 1.0)

        return actions

    def update(
        self,
        update_step: int,
        batch: Batch
    ) -> Dict[str, float]:
        cur_obs = torch.as_tensor(batch["observation"], device=self.device, dtype=self.dtype)
        actions = torch.as_tensor(batch["action"], device=self.device, dtype=self.dtype)
        rewards = torch.as_tensor(batch["reward"], device=self.device, dtype=self.dtype)
        terminated = torch.as_tensor(batch["terminated"], device=self.device, dtype=self.dtype)
        next_obs = torch.as_tensor(batch["next_observation"], device=self.device, dtype=self.dtype)

        update_info = self.update_batch.update_ddpg_networks(
            update_step=update_step,
            cur_obs=cur_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            next_obs=next_obs,
            noise_std=self._noise_std
        )
        
        return update_info

    def count_parameters(self) -> Tuple[str, str, str]:
        actor_num_params = count_parameters(self._actor)
        critic_num_params = count_parameters(self._critic)
        total_num_params = actor_num_params + critic_num_params

        num_total = format_params_str(total_num_params)
        num_actor = format_params_str(actor_num_params)
        num_critic = format_params_str(critic_num_params)

        return num_total, num_actor, num_critic

    def get_metrics(
        self,
        update_step: int,
        batch: Batch
    ) -> Dict[str, float]:
        cur_obs = torch.as_tensor(batch["observation"], device=self.device, dtype=self.dtype)
        actions = torch.as_tensor(batch["action"], device=self.device, dtype=self.dtype)
        next_obs = torch.as_tensor(batch["next_observation"], device=self.device, dtype=self.dtype)

        actor_metrics_info = get_actor_metrics(
            actor=self._actor,
            cur_obs=cur_obs
        )
        critic_metrics_info = get_critic_metrics(
            actor=self._actor,
            critic=self._critic,
            cur_obs=cur_obs,
            actions=actions,
            next_obs=next_obs,
            critic_use_cdq=self._cfg.critic_use_cdq
        )
        
        return {**actor_metrics_info, **critic_metrics_info}
