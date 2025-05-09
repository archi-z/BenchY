from typing import Dict, Union

import torch
import torch.nn as nn

from scale_rl.agents.sac.sac_network import (
    SACActor,
    SACCritic,
    SACClippedDoubleCritic,
    SACTemperature
)


class Update:
    def __init__(
        self,
        actor: SACActor,
        critic: Union[SACCritic, SACClippedDoubleCritic],
        temperature: SACTemperature,
        target_critic: Union[SACCritic, SACClippedDoubleCritic],
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temp_optimizer: torch.optim.Optimizer,
        cfg
    ):
        self._actor = actor
        self._critic = critic
        self._temperature = temperature
        self._target_critic = target_critic
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._temp_optimizer = temp_optimizer
        self._cfg = cfg

    def update_sac_networks(
        self,
        update_step: int,
        cur_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Dict[str, float]:
        actor_info = self.update_actor(
            cur_obs=cur_obs
        )

        temperature_info = self.update_temperature(
            entropy=actor_info['train_actor/entropy']
        )

        critic_info = self.update_critic(
            cur_obs=cur_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            next_obs=next_obs
        )

        update_target_network(
            network=self._critic,
            target_network=self._target_critic,
            target_tau=self._cfg.target_tau,
        )

        info = {
            **actor_info,
            **critic_info,
            **temperature_info,
        }

        return  info


    def update_actor(
        self,
        cur_obs: torch.Tensor
    ) -> Dict[str, float]:
        dist = self._actor(cur_obs)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)

        if self._cfg.critic_use_cdq:
            q1, q2 = self._critic(cur_obs, actions)
            q = torch.min(q1, q2).reshape(-1)
        else:
            q = self._critic(cur_obs, actions).reshape(-1)

        actor_loss = (log_probs * self._temperature() - q).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()

        actor_pnorm = sum(p.norm(2).item() for p in self._actor.parameters())
        actor_gnorm = sum(p.grad.norm(2).item() for p in self._actor.parameters() if p.grad is not None)

        self._actor_optimizer.step()

        info = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/entropy': -log_probs.mean().item(),
            'train_actor/action': actions.abs().mean().item(),
            'train_actor/pnorm': actor_pnorm,
            'train_actor/gnorm': actor_gnorm
        }

        return info


    def update_critic(
        self,
        cur_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_dist = self._actor(next_obs)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions)

            if self._cfg.critic_use_cdq:
                next_q1, next_q2 = self._target_critic(next_obs, next_actions)
                next_q = torch.min(next_q1, next_q2).reshape(-1)
            else:
                next_q = self._target_critic(next_obs, next_actions).reshape(-1)
            
            target_q = rewards + (self._cfg.gamma**self._cfg.n_step) * (1-terminated) * next_q
            target_q -= (
                (self._cfg.gamma**self._cfg.n_step) * (1 - terminated) *
                self._temperature() * next_log_probs
            )

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
            'train_critic/loss': critic_loss.item(),
            'train_critic/q1_mean': pred_q1.mean().item(),
            'train_critic/q2_mean': pred_q2.mean().item(),
            'train/rew_mean': rewards.mean().item(),
            'train_critic/pnorm': critic_pnorm,
            'train_critic/gnorm': critic_gnorm,
        }

        return info

    def update_temperature(
        self,
        entropy: float
    ) -> Dict[str, float]:
        temperature_value = self._temperature()
        temperature_loss = temperature_value * (entropy - self._cfg.temp_target_entropy)

        self._temp_optimizer.zero_grad()
        temperature_loss.backward()

        temperature_gnorm = sum(p.grad.norm(2).item() for p in self._temperature.parameters() if p.grad is not None)

        self._temp_optimizer.step()

        info = {
            'train/temperature': temperature_value.item(),
            'train/temperature_loss': temperature_loss.item(),
            'train/temperature_gnorm': temperature_gnorm
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
