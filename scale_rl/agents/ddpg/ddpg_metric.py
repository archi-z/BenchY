from typing import Dict, Union

import torch

from scale_rl.agents.ddpg.ddpg_network import (
    DDPGActor,
    DDPGCritic,
    DDPGClippedDoubleCritic
)
from scale_rl.networks.metrics import (
    Intermediates,
    get_dormant_ratio,
    get_feature_norm,
    get_srank,
    get_critic_featdot,
    get_critic_featdot_double
)


def get_actor_metrics(
    actor: DDPGActor,
    cur_obs: torch.Tensor
) -> Dict[str, float]:
    intermediates = Intermediates(net=actor)
    intermediates.register_hook()
    actor(cur_obs)

    dr1 = get_dormant_ratio(intermediates.features, tau=0.1)
    dr2 = get_dormant_ratio(intermediates.features, tau=0.2)
    fnorm = get_feature_norm(intermediates.features)
    srank = get_srank(intermediates.features['encoder.layer_norm'],thershold=0.01)

    actor_metrics_info = {
        'train_actor/DR0.1': dr1['total'],
        'train_actor/DR0.2': dr2['total'],
        'train_actor/fnorm': fnorm['encoder.layer_norm'],
        'train_actor/srank': srank
    }

    intermediates.remove()

    return actor_metrics_info


def get_critic_metrics(
    actor: DDPGActor,
    critic: Union[DDPGCritic, DDPGClippedDoubleCritic],
    cur_obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    critic_use_cdq: bool
) -> Dict[str, float]:
    intermediates = Intermediates(net=critic)
    intermediates.register_hook()
    critic(cur_obs, actions)

    if critic_use_cdq:
        q1_features = {}
        q2_features = {}
        for layer_name, activi in list(intermediates.features.items()):
            if "critics.0" in layer_name:
                q1_features.update({layer_name: activi})
            elif "critics.1" in layer_name:
                q2_features.update({layer_name: activi})
            else:
                raise KeyError(f"Unable to identify {layer_name}")
            
        q1_dr1 = get_dormant_ratio(q1_features,tau=0.1)
        q1_dr2 = get_dormant_ratio(q1_features,tau=0.2)
        q1_fnorm = get_feature_norm(q1_features)
        q1_srank = get_srank(q1_features['critics.0.encoder.layer_norm'], thershold=0.01)

        q2_dr1 = get_dormant_ratio(q2_features,tau=0.1)
        q2_dr2 = get_dormant_ratio(q2_features,tau=0.2)
        q2_fnorm = get_feature_norm(q2_features)
        q2_srank = get_srank(q2_features['critics.1.encoder.layer_norm'], thershold=0.01)

        q1_featdot, q2_featdot = get_critic_featdot_double(
            actor=actor,
            critic=critic,
            next_obs=next_obs,
            current_critic_feat=(q1_features['critics.0.encoder.layer_norm'],
                                q2_features['critics.1.encoder.layer_norm']),
            intermediates=intermediates
        )

        critic_metrics_info = {
            'train_critic/q1_DR0.1': q1_dr1['total'],
            'train_critic/q1_DR0.2': q1_dr2['total'],
            'train_critic/q2_DR0.1': q2_dr1['total'],
            'train_critic/q2_DR0.2': q2_dr2['total'],
            'train_critic/q1_fnorm': q1_fnorm['critics.0.encoder.layer_norm'],
            'train_critic/q2_fnorm': q2_fnorm['critics.1.encoder.layer_norm'],
            'train_critic/q1_srank': q1_srank,
            'train_critic/q2_srank': q2_srank,
            'train_critic/q1_featdot': q1_featdot,
            'train_critic/q2_featdot': q2_featdot,
        }

    else:
        dr1 = get_dormant_ratio(intermediates.features,tau=0.1)
        dr2 = get_dormant_ratio(intermediates.features,tau=0.2)
        fnorm = get_feature_norm(intermediates.features)
        srank = get_srank(intermediates.features['encoder.layer_norm'],thershold=0.01)
        featdot = get_critic_featdot(
            actor=actor,
            critic=critic,
            next_obs=next_obs,
            current_critic_feat=intermediates.features['encoder.layer_norm'],
            intermediates=intermediates
        )

        critic_metrics_info = {
            'train_critic/DR0.1': dr1['total'],
            'train_critic/DR0.2': dr2['total'],
            'train_critic/fnorm': fnorm['encoder.layer_norm'],
            'train_critic/srank': srank,
            'train_critic/featdot': featdot
        }

    intermediates.remove()

    return critic_metrics_info
