import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from esquilax.ml import rl


def test_agents():
    n_agents = 10
    observation_shape = (4,)

    k = jax.random.key(451)

    class Model(nn.module.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=2)(x)
            return jnp.sum(x)

    agents = rl.Agents.batch_init(
        k, Model(), optax.adam(1e-4), n_agents, observation_shape
    )

    assert agents.step.shape == (n_agents,)
    assert jnp.array_equal(agents.step, jnp.zeros((n_agents,), dtype=jnp.int32))
    assert agents.params["params"]["Dense_0"]["bias"].shape == (n_agents, 2)
    assert agents.params["params"]["Dense_0"]["kernel"].shape == (
        n_agents,
    ) + observation_shape + (2,)

    a = jnp.ones((n_agents,) + observation_shape)

    def loss(p, x):
        return agents.apply_fn(p, x)

    grads = jax.vmap(jax.grad(loss), in_axes=(0, 0))(agents.params, a)

    updated_agents = agents.batch_apply_grads(grads)

    assert isinstance(updated_agents, rl.Agents)
    assert jnp.array_equal(updated_agents.step, jnp.ones((n_agents,), dtype=jnp.int32))
    assert updated_agents.params["params"]["Dense_0"]["bias"].shape == (n_agents, 2)
    assert updated_agents.params["params"]["Dense_0"]["kernel"].shape == (
        n_agents,
    ) + observation_shape + (2,)
