import evosax
import jax
import jax.numpy as jnp
import pytest

from esquilax import evo


@pytest.fixture
def apply_fun():
    def f(params, obs):
        return jnp.sum(params * obs)

    return f


def test_evo_broadcast(apply_fun):
    k = jax.random.PRNGKey(101)
    strategy = evosax.strategies.SimpleGA(popsize=10, num_dims=5)
    es_params = strategy.default_params
    pop_state = strategy.initialize(k, es_params)

    pop, pop_state = strategy.ask_strategy(k, pop_state, es_params)
    obs = jnp.ones((10, 5))

    action = evo.broadcast_params(apply_fun, pop[0], obs)

    assert action.shape == (10,)


def test_evo_map(apply_fun):
    k = jax.random.PRNGKey(101)
    strategy = evosax.strategies.SimpleGA(popsize=10, num_dims=5)
    es_params = strategy.default_params
    pop_state = strategy.initialize(k, es_params)

    pop, pop_state = strategy.ask_strategy(k, pop_state, es_params)
    obs = jnp.ones((10, 5))

    action = evo.map_params(apply_fun, pop, obs)

    assert action.shape == (10,)
