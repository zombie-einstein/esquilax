import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from gymnax.environments.environment import Environment

from esquilax.ml import rl


def test_batch_init_agents():
    n_agents = 10
    observation_shape = (4,)

    k = jax.random.key(451)

    class Model(nn.module.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=2)(x)
            return jnp.sum(x)

    agents = rl.BatchPolicyAgent.init(
        k, Model(), optax.adam(1e-4), observation_shape, n_agents
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

    updated_agents = agents.apply_grads(grads=grads)

    assert isinstance(updated_agents, rl.Agent)
    assert jnp.array_equal(updated_agents.step, jnp.ones((n_agents,), dtype=jnp.int32))
    assert updated_agents.params["params"]["Dense_0"]["bias"].shape == (n_agents, 2)
    assert updated_agents.params["params"]["Dense_0"]["kernel"].shape == (
        n_agents,
    ) + observation_shape + (2,)


def test_training():
    k = jax.random.PRNGKey(101)

    observation_shape = (4,)

    k = jax.random.key(451)

    class Model(nn.module.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=2)(x)
            return jnp.sum(x)

    class TestAgent(rl.SharedPolicyAgent):
        def sample_actions(self, _k, observations):
            print(self.params, observations.shape)
            return self.apply_fn(self.params, observations), None

        def update(self, _k, trajectories):
            return self

    agent = TestAgent.init(k, Model(), optax.adam(1e-4), observation_shape)

    class Env(Environment):
        def step_env(
            self,
            key,
            state,
            action,
            params,
        ):
            return jnp.ones((4,)), 10, 0, False, None

        def reset_env(self, key, params):
            return jnp.ones((4,)), 10

        def get_obs(self, state, params):
            raise jnp.ones((4,))

    env = Env()

    n_epochs = 3
    n_env = 2
    n_env_steps = 5

    updated_agent, rewards = rl.train(
        k,
        agent,
        env,
        env.default_params,
        n_epochs,
        n_env,
        n_env_steps,
        show_progress=False,
    )

    assert isinstance(updated_agent, TestAgent)
    assert rewards.shape == (n_epochs, n_env, n_env_steps)
