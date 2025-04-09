from typing import Dict, Tuple, Union

import torch

from scale_rl.buffers import Batch
from scale_rl.agents.ddpg.ddpg_network import (
    DDPGActor,
    DDPGCritic,
    DDPGClippedDoubleCritic
)

def update_actor(
    actor: DDPGActor,
    critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
    batch: Batch,
    critic_use_cdq: bool,
    noise_std: float,
    actor_optimizer: torch.optim.Optimizer
) -> Tuple[DDPGActor, Dict[str, float]]:
    observations = batch['observation']
    
    actions = actor(observations)
    noise = noise_std * torch.randn_like(actions)
    actions = (actions+noise).clamp(-1.0, 1.0)

    if critic_use_cdq:
        q1, q2 = critic(observations, actions)
        q = torch.min(q1, q2).reshape(-1)
    else:
        q = critic(observations, actions).reshape(-1)
    
    actor_loss = -q.mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    
    actor_pnorm = sum(p.norm(2).item() for p in actor.parameters())
    actor_gnorm = sum(p.grad.norm(2).item() for p in actor.parameters() if p.grad is not None)

    actor_optimizer.step()

    info = {
        'train/actor_loss': actor_loss.item(),
        'train/actor_action': actions.abs().mean().item(),
        'train/actor_pnorm': actor_pnorm,
        'train/actor_gnorm': actor_gnorm
    }

    return actor, info


def update_critic(
    actor: DDPGActor,
    critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
    target_critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    noise_std: float,
    critic_optimizer: torch.optim.Optimizer
) -> Tuple[Union[DDPGCritic, DDPGClippedDoubleCritic], Dict[str, float]]:
    next_obs = batch['next_observation']
    cur_obs = batch['observation']
    actions = batch['action']
    rewards = batch['reward']
    terminated = batch['terminated']

    with torch.no_grad():
        next_actions = actor(next_obs)
        noise = noise_std * torch.randn_like(next_actions)
        next_actions = (next_actions+noise).clamp(-1.0, 1.0)

        if critic_use_cdq:
            next_q1, next_q2 = target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).reshape(-1)
        else:
            next_q = target_critic(next_obs, next_actions).reshape(-1)

        target_q = rewards + (gamma**n_step) * (1-terminated) * next_q

    if critic_use_cdq:
        pred_q1, pred_q2 = critic(cur_obs, actions)
        pred_q1 = pred_q1.reshape(-1)
        pred_q2 = pred_q2.reshape(-1)
        critic_loss = ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean()
    else:
        pred_q = critic(cur_obs, actions).reshape(-1)
        pred_q1 = pred_q2 = pred_q
        critic_loss = ((pred_q - target_q) ** 2).mean()
    
    critic_optimizer.zero_grad()
    critic_loss.backward()

    critic_pnorm = sum(p.norm(2).item() for p in critic.parameters())
    critic_gnorm = sum(p.grad.norm(2).item() for p in critic.parameters() if p.grad is not None)

    critic_optimizer.step()

    info = {
        'train/critic_loss': critic_loss.item(),
        'train/q1_mean': pred_q1.mean().item(),
        'train/q2_mean': pred_q2.mean().item(),
        'train/rew_mean': rewards.mean().item(),
        'train/critic_pnorm': critic_pnorm,
        'train/critic_gnorm': critic_gnorm,
    }

    return critic, info


def update_target_network(
    network: Union[DDPGCritic, DDPGClippedDoubleCritic],
    target_network: Union[DDPGCritic, DDPGClippedDoubleCritic],
    target_tau: float,
) -> Tuple[Union[DDPGCritic, DDPGClippedDoubleCritic], Dict[str, float]]:
    with torch.no_grad():
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.copy_(target_tau*param + (1-target_tau)*target_param)
    info = {}

    return target_network, info


def update_ddpg_networks(
    actor: DDPGActor,
    critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
    target_critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    noise_std: float
):
    new_actor, actor_info = update_actor(
        actor=actor,
        critic=critic,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
        noise_std=noise_std,
        actor_optimizer=actor_optimizer
    )

    new_critic, critic_info = update_critic(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=critic_use_cdq,
        noise_std=noise_std,
        critic_optimizer=critic_optimizer
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info
    }

    return new_actor, new_critic, new_target_critic, info
