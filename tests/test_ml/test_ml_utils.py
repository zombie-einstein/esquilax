import evosax
import jax
import jax.numpy as jnp
import pytest

from esquilax import ml


@pytest.fixture
def apply_fun():
    def f(params, obs):
        return jnp.sum(params * obs)

    return f


@pytest.fixture
def random_apply_fun():
    def f(k, params, obs):
        x = jax.random.uniform(k, obs.shape)
        return jnp.sum(x * params * obs)

    return f


@pytest.mark.parametrize("broadcast", [True, False])
def test_get_actions(broadcast, apply_fun):
    k = jax.random.PRNGKey(101)
    n_agents = 10
    n_obs = 5

    strategy = evosax.strategies.SimpleGA(popsize=n_agents, num_dims=n_obs)
    es_params = strategy.default_params
    pop_state = strategy.initialize(k, es_params)

    pop, pop_state = strategy.ask_strategy(k, pop_state, es_params)
    obs = jnp.ones((n_agents, n_obs))

    if broadcast:
        action = ml.get_actions(apply_fun, True, pop[0], obs)
    else:
        action = ml.get_actions(apply_fun, False, pop, obs)

    assert action.shape == (n_agents,)


@pytest.mark.parametrize("broadcast", [True, False])
def test_sample_action_actions(broadcast, random_apply_fun):
    k = jax.random.PRNGKey(101)

    n_agents = 10
    n_obs = 5

    params = jax.random.uniform(k, (n_agents, n_obs))
    obs = jnp.ones((n_agents, n_obs))

    if broadcast:
        action = ml.sample_actions(random_apply_fun, True, k, params[0], obs)
    else:
        action = ml.sample_actions(random_apply_fun, False, k, params, obs)

    assert action.shape == (n_agents,)
