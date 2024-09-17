import flax.linen as nn
import jax.numpy as jnp


class SimpleModel(nn.module.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2)(x)
        return jnp.sum(x)
