from typing import Dict, Union

import torch
import torch.nn as nn

from scale_rl.agents.ddpg.ddpg_network import (
    DDPGActor,
    DDPGCritic,
    DDPGClippedDoubleCritic
)


class Update:
    def __init__(
        self,
        actor: DDPGActor,
        critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
        target_critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        cfg
    ):
        self._actor = actor
        self._critic = critic
        self._target_critic = target_critic
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._cfg = cfg

    def update_ddpg_networks(
        self,
        update_step: int,
        cur_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        next_obs: torch.Tensor,
        noise_std: float
    ) -> Dict[str, float]:
        actor_info = self.update_actor(
            cur_obs=cur_obs,
            noise_std=noise_std
        )

        critic_info = self.update_critic(
            cur_obs=cur_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            next_obs=next_obs,
            noise_std=noise_std
        )

        update_target_network(
            network=self._critic,
            target_network=self._target_critic,
            target_tau=self._cfg.target_tau
        )

        info = {
            **actor_info,
            **critic_info
        }

        return info

    def update_actor(
        self,
        cur_obs: torch.Tensor,
        noise_std: float
    ) -> Dict[str, float]:
        actions = self._actor(cur_obs)
        noise = noise_std * torch.randn_like(actions)
        actions = (actions+noise).clamp(-1.0, 1.0)

        if self._cfg.critic_use_cdq:
            q1, q2 = self._critic(cur_obs, actions)
            q = torch.min(q1, q2).reshape(-1)
        else:
            q = self._critic(cur_obs, actions).reshape(-1)
        
        actor_loss = -q.mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        
        actor_pnorm = sum(p.norm(2).item() for p in self._actor.parameters())
        actor_gnorm = sum(p.grad.norm(2).item() for p in self._actor.parameters() if p.grad is not None)

        self._actor_optimizer.step()

        info = {
            'train/actor_loss': actor_loss.item(),
            'train/actor_action': actions.abs().mean().item(),
            'train/actor_pnorm': actor_pnorm,
            'train/actor_gnorm': actor_gnorm
        }

        return info


    def update_critic(
        self,
        cur_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        next_obs: torch.Tensor,
        noise_std: float
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_actions = self._actor(next_obs)
            noise = noise_std * torch.randn_like(next_actions)
            next_actions = (next_actions+noise).clamp(-1.0, 1.0)

            if self._cfg.critic_use_cdq:
                next_q1, next_q2 = self._target_critic(next_obs, next_actions)
                next_q = torch.min(next_q1, next_q2).reshape(-1)
            else:
                next_q = self._target_critic(next_obs, next_actions).reshape(-1)

            target_q = rewards + (self._cfg.gamma**self._cfg.n_step) * (1-terminated) * next_q

        if self._cfg.critic_use_cdq:
            pred_q1, pred_q2 = self._critic(cur_obs, actions)
            pred_q1 = pred_q1.reshape(-1)
            pred_q2 = pred_q2.reshape(-1)
            critic_loss = ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean()
        else:
            pred_q = self._critic(cur_obs, actions).reshape(-1)
            pred_q1 = pred_q2 = pred_q
            critic_loss = ((pred_q - target_q) ** 2).mean()
        
        self._critic_optimizer.zero_grad()
        critic_loss.backward()

        critic_pnorm = sum(p.norm(2).item() for p in self._critic.parameters())
        critic_gnorm = sum(p.grad.norm(2).item() for p in self._critic.parameters() if p.grad is not None)

        self._critic_optimizer.step()

        info = {
            'train/critic_loss': critic_loss.item(),
            'train/q1_mean': pred_q1.mean().item(),
            'train/q2_mean': pred_q2.mean().item(),
            'train/rew_mean': rewards.mean().item(),
            'train/critic_pnorm': critic_pnorm,
            'train/critic_gnorm': critic_gnorm,
        }

        return info


def update_target_network(
    network: nn.Module,
    target_network: nn.Module,
    target_tau: float,
):
    with torch.no_grad():
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.copy_(target_tau*param + (1-target_tau)*target_param)
