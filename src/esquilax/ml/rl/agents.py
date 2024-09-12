from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from typing_extensions import Self


class Agents(TrainState):
    @classmethod
    def batch_init(
        cls,
        k: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        n_agents: int,
        observation_shape: Tuple[int, ...],
    ) -> Self:
        fake_args = jnp.zeros(observation_shape)

        def init(_k):
            params = model.init(_k, fake_args)
            return cls.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(k, n_agents)

        return jax.vmap(init)(keys)

    def batch_apply_grads(self, grads: chex.ArrayTree) -> Self:
        return jax.vmap(lambda a, g: a.apply_gradients(grads=g))(self, grads)
