import math

import torch.nn as nn

# TODO delete
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))


def orthogonal_init_(layer: nn.Module, gain=math.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def he_normal_init_(layer: nn.Module):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
