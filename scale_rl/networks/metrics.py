from typing import Dict, Tuple

import torch
import torch.nn as nn


class Intermediates():
    def __init__(self, net: nn.Module):
        self.net = net
        self.features = {}
        self.hooks = []

    def register_hook(self):
        for name, module in self.net.named_modules():
            if name == '' or len(list(module.children())) > 0:
                continue
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def _hook_fn(self, layer_name):
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self.features[layer_name] = output.detach()
        
        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def get_dormant_ratio(
    features: Dict[str, torch.Tensor],
    tau: float
) -> Dict[str, float]:
    ratios = {}
    masks = []

    for layer_name, activs in features.items():
        # For double critics, lets just stack them into one batch
        if activs.dim() > 2:
            activs = activs.reshape(-1, activs.shape[-1])

        score = activs.abs().mean(dim=0)
        normalized_score = score / (score.mean() + 1e-9)

        if tau > 0.0:
            dormant_mask = (normalized_score <= tau)
        else:
            dormant_mask = torch.isclose(normalized_score, torch.zeros_like(normalized_score))

        percent = dormant_mask.sum().item() / dormant_mask.numel() * 100.0
        ratios[f"{layer_name}"] = percent
        masks.append(dormant_mask)

    # aggregated mask of entire network
    all_masks = torch.cat(masks)
    total_percent = all_masks.sum().item() / all_masks.numel() * 100.0
    ratios['total'] = total_percent

    return ratios


def get_feature_norm(
    features: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    norms = {}
    total_norm = 0.0
    for layer_name, activs in features.items():
        # For double critics, lets just stack them into one batch
        if activs.dim() > 2:
            activs = activs.reshape(-1, activs.shape[-1])

        # Compute the L2 norm for all examples in the batch at once
        batch_norms = torch.linalg.norm(activs, ord=2, axis=-1)

        # Compute the expected (mean) L2 norm across the batch
        expected_norm = batch_norms.mean().item()

        norms[f"{layer_name}"] = expected_norm
        total_norm += expected_norm

    norms['total'] = total_norm

    return norms


def get_srank(
    matrix: torch.Tensor,
    thershold=0.01
) -> int:
    if matrix.dim() > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])

    singular_vals = torch.linalg.svdvals(matrix)
    total = singular_vals.sum().item()
    target = total * (1.0 - thershold)

    acc = 0.0
    k = 0
    for sv in singular_vals:
        acc += sv.item()
        k += 1
        if acc >= target:
            break
    return k


def get_critic_featdot(
    actor: nn.Module,
    critic: nn.Module,
    next_obs: torch.Tensor,
    current_critic_feat: torch.Tensor,
    intermediates: Intermediates,
    deterministic_policy=True
) -> float:
    if deterministic_policy:
        next_actions = actor(next_obs)
        
    else:
        dist = actor(next_obs, temperature=1.0)
        next_actions = dist.sample()
        
    critic(next_obs, next_actions)

    next_critic_feat = intermediates.features['encoder.layer_norm']
    
    result = (next_critic_feat * current_critic_feat).sum(dim=1).mean(dim=0).item()

    return result


def get_critic_featdot_double(
    actor: nn.Module,
    critic: nn.Module,
    next_obs: torch.Tensor,
    current_critic_feat: Tuple[torch.Tensor, torch.Tensor],
    intermediates: Intermediates,
    deterministic_policy=True
):
    if deterministic_policy:
        next_actions = actor(next_obs)

    else:
        dist = actor(next_obs, temperature=1.0)
        next_actions = dist.sample()
        
    critic(next_obs, next_actions)

    q1_next_critic_feat = intermediates.features['critics.0.encoder.layer_norm']
    q2_next_critic_feat = intermediates.features['critics.1.encoder.layer_norm']

    result1 = (q1_next_critic_feat * current_critic_feat[0]).sum(dim=1).mean(dim=0).item()
    result2 = (q2_next_critic_feat * current_critic_feat[1]).sum(dim=1).mean(dim=0).item()

    return result1, result2


def get_mr_critic_featdot(
    encoder: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    next_zs: torch.Tensor,
    current_critic_feat: torch.Tensor,
    intermediates: Intermediates,
    deterministic_policy=True
) -> float:
    if deterministic_policy:
        next_actions = actor(next_zs)
        
    else:
        dist = actor(next_zs, temperature=1.0)
        next_actions = dist.sample()

    next_za = encoder.za(next_actions)
    next_zsa = encoder(next_zs, next_za)
    critic(next_zsa)

    next_critic_feat = intermediates.features['encoder.layer_norm']
    
    result = (next_critic_feat * current_critic_feat).sum(dim=1).mean(dim=0).item()

    return result


def get_mr_critic_featdot_double(
    encoder: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    next_zs: torch.Tensor,
    current_critic_feat: Tuple[torch.Tensor, torch.Tensor],
    intermediates: Intermediates,
    deterministic_policy=True
):
    if deterministic_policy:
        next_actions = actor(next_zs)

    else:
        dist = actor(next_zs, temperature=1.0)
        next_actions = dist.sample()

    next_za = encoder.za(next_actions)
    next_zsa = encoder(next_zs, next_za)
    critic(next_zsa)

    q1_next_critic_feat = intermediates.features['critics.0.encoder.layer_norm']
    q2_next_critic_feat = intermediates.features['critics.1.encoder.layer_norm']

    result1 = (q1_next_critic_feat * current_critic_feat[0]).sum(dim=1).mean(dim=0).item()
    result2 = (q2_next_critic_feat * current_critic_feat[1]).sum(dim=1).mean(dim=0).item()

    return result1, result2


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
    

def format_params_str(num_params):
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"


def cal_pge(model: nn.Module) -> Tuple[float, float, float]:
    total_l1 = sum(p.abs().sum() for p in model.parameters())
    eps = 1e-12

    pnorm = torch.tensor(0.0, device=total_l1.device)
    gnorm = torch.tensor(0.0, device=total_l1.device)
    elr = torch.tensor(0.0, device=total_l1.device)
    for p in model.parameters():
        wi = p.abs().sum() / (total_l1 + eps)
        gi = p.grad.norm(2)
        thetai = p.norm(2)
        elr += wi * (gi / (thetai + eps))

        pnorm += thetai
        gnorm += gi
    
    return pnorm, gnorm, elr.sqrt()
