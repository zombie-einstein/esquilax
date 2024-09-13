from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from typing_extensions import Self


class Agent(TrainState):
    def sample_actions(self, k: chex.PRNGKey, observations: chex.Array):
        raise NotImplementedError

    def update(self, k: chex.PRNGKey, trajectories) -> Self:
        raise NotImplementedError

    def apply_grads(self, *, grads, **kwargs) -> Self:
        raise NotImplementedError


class SharedPolicyAgent(Agent):
    @classmethod
    def init(
        cls,
        k: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
    ) -> Self:
        fake_args = jnp.zeros(observation_shape)
        params = model.init(k, fake_args)
        return cls.create(apply_fn=model.apply, params=params, tx=tx)

    def apply_grads(self, *, grads, **kwargs) -> Self:
        return self.apply_gradients(grads=grads, **kwargs)


class BatchPolicyAgent(Agent):
    @classmethod
    def init(
        cls,
        k: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
        n_agents: int,
    ) -> Self:
        fake_args = jnp.zeros(observation_shape)

        def init(_k):
            params = model.init(_k, fake_args)
            return cls.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(k, n_agents)
        return jax.vmap(init)(keys)

    def apply_grads(self, *, grads, **kwargs) -> Self:
        return jax.vmap(lambda a, g: a.apply_gradients(grads=g, **kwargs))(self, grads)
