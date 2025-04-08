from typing import Dict, Tuple, Union

import torch

from scale_rl.buffers import Batch
from scale_rl.agents.sac.sac_network import (
    SACActor,
    SACCritic,
    SACClippedDoubleCritic,
    SACTemperature
)


def update_actor(
    actor: SACActor,
    critic: Union[SACCritic, SACClippedDoubleCritic],
    temperature: SACTemperature,
    batch: Batch,
    critic_use_cdq: bool,
    actor_optimizer: torch.optim.Optimizer
) -> Tuple[SACActor, Dict[str, float]]:
    observations = batch['observation']
    
    dist = actor(observations)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    if critic_use_cdq:
        q1, q2 = critic(observations, actions)
        q = torch.min(q1, q2).reshape(-1)
    else:
        q = critic(observations, actions).reshape(-1)

    actor_loss = (log_probs * temperature() - q).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()

    actor_pnorm = sum(p.norm(2).item() for p in actor.parameters())
    actor_gnorm = sum(p.grad.norm(2).item() for p in actor.parameters() if p.grad is not None)

    actor_optimizer.step()

    info = {
        'actor_loss': actor_loss.item(),
        'entropy': -log_probs.mean().item(),
        'actor_action': actions.abs().mean().item(),
        'actor_pnorm': actor_pnorm,
        'actor_gnorm': actor_gnorm
    }

    return actor, info


def update_critic(
    actor: SACActor,
    critic: Union[SACCritic, SACClippedDoubleCritic],
    target_critic: Union[SACCritic, SACClippedDoubleCritic],
    temperature: SACTemperature,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    critic_optimizer: torch.optim.Optimizer
) -> Tuple[Union[SACCritic, SACClippedDoubleCritic], Dict[str, float]]:
    next_obs = batch['next_observation']
    cur_obs = batch['observation']
    actions = batch['action']
    rewards = batch['reward']
    terminated = batch['terminated']

    with torch.no_grad():
        next_dist = actor(next_obs)
        next_actions = next_dist.sample()
        next_log_probs = next_dist.log_prob(next_actions)

        if critic_use_cdq:
            next_q1, next_q2 = target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).reshape(-1)
        else:
            next_q = target_critic(next_obs, next_actions).reshape(-1)
        
        target_q = rewards + (gamma**n_step) * (1-terminated) * next_q
        target_q -= (
            (gamma**n_step) * (1 - terminated) *
            temperature() * next_log_probs
        )

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
        'critic_loss': critic_loss.item(),
        'q1_mean': pred_q1.mean().item(),
        'q2_mean': pred_q2.mean().item(),
        'rew_mean': rewards.mean().item(),
        'critic_pnorm': critic_pnorm,
        'critic_gnorm': critic_gnorm,
    }

    return critic, info


def update_target_network(
    network: Union[SACCritic, SACClippedDoubleCritic],
    target_network: Union[SACCritic, SACClippedDoubleCritic],
    target_tau: float,
) -> Tuple[Union[SACCritic, SACClippedDoubleCritic], Dict[str, float]]:
    with torch.no_grad():
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.copy_(target_tau*param + (1-target_tau)*target_param)
    info = {}

    return target_network, info


def update_temperature(
    temperature: SACTemperature,
    entropy: float,
    target_entropy: float,
    temp_optimizer: torch.optim.Optimizer
) -> Tuple[SACTemperature, Dict[str, float]]:
    temperature_value = temperature()
    temperature_loss = temperature_value * (entropy - target_entropy)

    temp_optimizer.zero_grad()
    temperature_loss.backward()

    temperature_gnorm = sum(p.grad.norm(2).item() for p in temperature.parameters() if p.grad is not None)

    temp_optimizer.step()

    info = {
        'temperature': temperature_value.item(),
        'temperature_loss': temperature_loss.item(),
        'temperature_gnorm': temperature_gnorm
    }

    return temperature, info


def update_sac_networks(
    actor: SACActor,
    critic: Union[SACCritic, SACClippedDoubleCritic],
    target_critic: Union[SACCritic, SACClippedDoubleCritic],
    temperature: SACTemperature,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    temp_optimizer: torch.optim.Optimizer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    temp_target_entropy: float
):
    new_actor, actor_info = update_actor(
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
        actor_optimizer=actor_optimizer
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info['entropy'],
        target_entropy=temp_target_entropy,
        temp_optimizer=temp_optimizer
    )

    new_critic, critic_info = update_critic(
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=critic_use_cdq,
        critic_optimizer=critic_optimizer
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return  new_actor, new_critic, new_target_critic, new_temperature, info
