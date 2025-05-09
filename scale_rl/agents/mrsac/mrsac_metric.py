from typing import Dict, Union

import torch

from scale_rl.agents.mrsac.mrsac_network import (
    MRSACEncoder,
    MRSACActor,
    MRSACCritic,
    MRSACClippedDoubleCritic
)
from scale_rl.networks.metrics import (
    Intermediates,
    get_dormant_ratio,
    get_feature_norm,
    get_srank,
    get_mr_critic_featdot,
    get_mr_critic_featdot_double
)


def get_encoder_metrics(
    encoder: MRSACEncoder,
    cur_obs: torch.Tensor,
    actions: torch.Tensor
) -> Dict[str, float]:
    intermediates = Intermediates(net=encoder)
    intermediates.register_hook()

    zs = encoder.zs(cur_obs)
    za = encoder.za(actions)
    zsa = encoder(zs, za)
    encoder.model_all(zs, za)

    zs_features = {k: v for k, v in intermediates.features.items() if 'zs' in k}
    za_features = {'fc': intermediates.features['fc'], 'activ': intermediates.features['activ']}
    zsa_features = {k: v for k, v in intermediates.features.items() if 'zsa' in k}
    model_features = {'model': intermediates.features['model']}

    zs_dr1 = get_dormant_ratio(zs_features, tau=0.1)
    zs_dr2 = get_dormant_ratio(zs_features, tau=0.2)
    za_dr1 = get_dormant_ratio(za_features, tau=0.1)
    za_dr2 = get_dormant_ratio(za_features, tau=0.2)
    zsa_dr1 = get_dormant_ratio(zsa_features, tau=0.1)
    zsa_dr2 = get_dormant_ratio(zsa_features, tau=0.2)
    model_dr1 = get_dormant_ratio(model_features, tau=0.1)
    model_dr2 = get_dormant_ratio(model_features, tau=0.2)
    zs_srank = get_srank(zs_features['zs.layer_norm'], thershold=0.01)
    za_srank = get_srank(za_features['activ'], thershold=0.01)
    zsa_srank = get_srank(zsa_features['zsa.layer_norm'], thershold=0.01)
    model_srank = get_srank(model_features['model'], thershold=0.01)
    fnorm = get_feature_norm(intermediates.features)

    actor_metrics_info = {
        'train_encoder/zs_DR0.1': zs_dr1['total'],
        'train_encoder/zs_DR0.2': zs_dr2['total'],
        'train_encoder/za_DR0.1': za_dr1['total'],
        'train_encoder/za_DR0.2': za_dr2['total'],
        'train_encoder/zsa_DR0.1': zsa_dr1['total'],
        'train_encoder/zsa_DR0.2': zsa_dr2['total'],
        'train_encoder/model_DR0.1': model_dr1['total'],
        'train_encoder/model_DR0.2': model_dr2['total'],
        'train_encoder/zs_fnorm': fnorm['zs.layer_norm'],
        'train_encoder/za_fnorm': fnorm['activ'],
        'train_encoder/zsa_fnorm': fnorm['zsa.layer_norm'],
        'train_encoder/model_fnorm': fnorm['model'],
        'train_encoder/zs_srank': zs_srank,
        'train_encoder/za_srank': za_srank,
        'train_encoder/zsa_srank': zsa_srank,
        'train_encoder/model_srank': model_srank
    }

    intermediates.remove()

    return actor_metrics_info, zs, zsa


def get_actor_metrics(
    actor: MRSACActor,
    zs: torch.Tensor
) -> Dict[str, float]:
    intermediates = Intermediates(net=actor)
    intermediates.register_hook()
    actor(zs)

    dr1 = get_dormant_ratio(intermediates.features, tau=0.1)
    dr2 = get_dormant_ratio(intermediates.features, tau=0.2)
    fnorm = get_feature_norm(intermediates.features)
    srank = get_srank(intermediates.features['encoder.layer_norm'], thershold=0.01)

    actor_metrics_info = {
        'train_actor/DR0.1': dr1['total'],
        'train_actor/DR0.2': dr2['total'],
        'train_actor/fnorm': fnorm['encoder.layer_norm'],
        'train_actor/srank': srank
    }

    intermediates.remove()

    return actor_metrics_info


def get_critic_metrics(
    encoder: MRSACEncoder,
    actor: MRSACActor,
    critic: Union[MRSACCritic, MRSACClippedDoubleCritic],
    zsa: torch.Tensor,
    next_zs: torch.Tensor,
    critic_use_cdq: bool
) -> Dict[str, float]:
    intermediates = Intermediates(net=critic)
    intermediates.register_hook()
    critic(zsa)

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

        q1_featdot, q2_featdot = get_mr_critic_featdot_double(
            encoder=encoder,
            actor=actor,
            critic=critic,
            next_zs=next_zs,
            current_critic_feat=(q1_features['critics.0.encoder.layer_norm'],
                                q2_features['critics.1.encoder.layer_norm']),
            intermediates=intermediates,
            deterministic_policy=False
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
        featdot = get_mr_critic_featdot(
            encoder=encoder,
            actor=actor,
            critic=critic,
            next_zs=next_zs,
            current_critic_feat=intermediates.features['encoder.layer_norm'],
            intermediates=intermediates,
            deterministic_policy=False
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
