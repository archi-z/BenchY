from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scale_rl.agents.mrddpg.mrddpg_network import (
    MRDDPGEncoder,
    MRDDPGActor,
    MRDDPGCritic,
    MRDDPGClippedDoubleCritic
)
from scale_rl.networks.loss import TwoHot, masked_mse


class Update:
    def __init__(
        self,
        encoder: MRDDPGEncoder,
        actor: MRDDPGActor,
        critic: Union[MRDDPGCritic, MRDDPGClippedDoubleCritic],
        target_encoder: MRDDPGEncoder,
        target_critic: Union[MRDDPGCritic, MRDDPGClippedDoubleCritic],
        encoder_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        cfg
    ):
        self._encoder = encoder
        self._actor = actor
        self._critic = critic
        self._target_encoder = target_encoder
        self._target_critic = target_critic
        self._encoder_optimizer = encoder_optimizer
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._cfg = cfg

        self._two_hot = TwoHot(
            cfg.device,
            cfg.lower,
            cfg.upper,
            cfg.num_bins
        )

    def update_ac_networks(
        self,
        update_step: int,
        cur_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        next_obs: torch.Tensor,
        noise_std: float
    ) -> Dict[str, float]:
        with torch.no_grad():
            zs = self._encoder.zs(cur_obs)
            za = self._encoder.za(actions)
            zsa = self._encoder(zs, za)
            next_zs = self._target_encoder.zs(next_obs)

        actor_info = self.update_actor(
            zs=zs, 
            noise_std=noise_std
        )
        critic_info = self.update_critic(
            zsa=zsa,
            next_zs=next_zs,
            rewards=rewards,
            terminated=terminated,
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

    def update_encoder(
        self,
        cur_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_zs = self._target_encoder.zs(next_obs)

        pred_zs = self._encoder.zs(cur_obs[:, 0])
        prev_not_done = 1
        encoder_loss = 0

        for i in range(self._cfg.encoder_horizon):
            za = self._encoder.za(actions[:, i])
            pred_d, pred_zs, pred_r = self._encoder.model_all(pred_zs, za)

            dyn_loss = masked_mse(pred_zs, next_zs[:, i], prev_not_done)
            reward_loss = (self._two_hot.cross_entropy_loss(pred_r, rewards[:, i].reshape(-1, 1)) * prev_not_done).mean()
            done_loss = masked_mse(pred_d, terminated[:, i].reshape(-1, 1), prev_not_done)

            encoder_loss += self._cfg.dyn_weight * dyn_loss + self._cfg.reward_weight * reward_loss + self._cfg.done_weight * done_loss
            prev_not_done = (1 - terminated[:,i].reshape(-1,1)) * prev_not_done

        self._encoder_optimizer.zero_grad()
        encoder_loss.backward()

        encoder_pnorm = sum(p.norm(2).item() for p in self._encoder.parameters())
        encoder_gnorm = sum(p.grad.norm(2).item() for p in self._encoder.parameters() if p.grad is not None)

        self._encoder_optimizer.step()

        info = {
            'train_encoder/dyn_loss': dyn_loss.item(),
            'train_encoder/reward_loss': reward_loss.item(),
            'train_encoder/done_loss': done_loss.item(),
            'train_encoder/loss': encoder_loss.item(),
            'train_encoder/pnorm': encoder_pnorm,
            'train_encoder/gnorm': encoder_gnorm
        }

        return info

    def update_actor(
        self,
        zs: torch.Tensor,
        noise_std: float
    ) -> Dict[str, float]:
        actions = self._actor(zs)
        noise = noise_std * torch.randn_like(actions)
        actions = (actions+noise).clamp(-1.0, 1.0)

        za = self._encoder.za(actions)
        zsa = self._encoder(zs, za)

        if self._cfg.critic_use_cdq:
            q1, q2 = self._critic(zsa)
            q = torch.min(q1, q2).reshape(-1)
        else:
            q = self._critic(zsa).reshape(-1)
        
        actor_loss = -q.mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        
        actor_pnorm = sum(p.norm(2).item() for p in self._actor.parameters())
        actor_gnorm = sum(p.grad.norm(2).item() for p in self._actor.parameters() if p.grad is not None)

        self._actor_optimizer.step()

        info = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/action': actions.abs().mean().item(),
            'train_actor/pnorm': actor_pnorm,
            'train_actor/gnorm': actor_gnorm
        }

        return info


    def update_critic(
        self,
        zsa: torch.Tensor,
        next_zs: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        noise_std: float
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_actions = self._actor(next_zs)
            noise = noise_std * torch.randn_like(next_actions)
            next_actions = (next_actions+noise).clamp(-1.0, 1.0)

            next_za = self._target_encoder.za(next_actions)
            next_zsa = self._target_encoder(next_zs, next_za)

            if self._cfg.critic_use_cdq:
                next_q1, next_q2 = self._target_critic(next_zsa)
                next_q = torch.min(next_q1, next_q2).reshape(-1)
            else:
                next_q = self._target_critic(next_zsa).reshape(-1)

            target_q = rewards + (self._cfg.gamma**self._cfg.n_step) * (1-terminated) * next_q

        if self._cfg.critic_use_cdq:
            pred_q1, pred_q2 = self._critic(zsa)
            pred_q1 = pred_q1.reshape(-1)
            pred_q2 = pred_q2.reshape(-1)
            critic_loss = ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean()
        else:
            pred_q = self._critic(zsa).reshape(-1)
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
    
def update_target_network(
    network: nn.Module,
    target_network: nn.Module,
    target_tau: float,
):
    with torch.no_grad():
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.copy_(target_tau*param + (1-target_tau)*target_param)
