import jax
import jax.numpy as jnp
import optax

from esquilax.ml import rl

from .conftest import SimpleModel


def test_batch_agent():
    n_agents = 10
    observation_shape = (4,)

    k = jax.random.key(451)

    agents = rl.BatchPolicyAgent.init(
        k, SimpleModel(), optax.adam(1e-4), observation_shape, n_agents
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


class Agent(rl.SharedPolicyAgent):
    def sample_actions(self, _k, observations):
        return self.apply_fn(self.params, observations), None

    def update(self, _k, trajectories):
        return self, (10, 10)


def test_training_shared_policy():
    k = jax.random.key(451)
    observation_shape = (4,)

    agent = Agent.init(k, SimpleModel(), optax.adam(1e-4), observation_shape)

    class TestEnv(rl.Environment):
        def step(
            self,
            key,
            params,
            state,
            action,
        ):
            return jnp.ones((4,)), 10, 0, False

        def reset(self, key, params):
            return jnp.ones((4,)), 10

        def get_obs(self, state, params):
            raise jnp.ones((4,))

    env = TestEnv()

    n_epochs = 3
    n_env = 2
    n_env_steps = 5

    updated_agent, rewards, train_data = rl.train(
        k,
        agent,
        env,
        env.default_params,
        n_epochs,
        n_env,
        n_env_steps,
        show_progress=False,
    )

    assert isinstance(updated_agent, Agent)
    assert rewards.shape == (n_epochs, n_env, n_env_steps)
    assert isinstance(train_data, tuple)
    assert train_data[0].shape == (n_epochs,)
    assert train_data[1].shape == (n_epochs,)

    trajectories = rl.test(
        k,
        agent,
        env,
        env.default_params,
        n_env,
        n_env_steps,
        show_progress=False,
    )

    assert isinstance(trajectories, rl.Trajectory)
    assert trajectories.rewards.shape == (n_env, n_env_steps)
    assert trajectories.obs.shape == (n_env, n_env_steps) + observation_shape
    assert trajectories.actions.shape == (n_env, n_env_steps)
    assert trajectories.action_values is None


def test_training_multi_policy():
    k = jax.random.key(451)
    observation_shape = (4,)

    agents = dict(
        a=Agent.init(k, SimpleModel(), optax.adam(1e-4), observation_shape),
        b=Agent.init(k, SimpleModel(), optax.adam(1e-4), observation_shape),
    )

    class TestEnv(rl.Environment):
        def reset(self, key, params):
            return (
                dict(a=jnp.ones((4,)), b=jnp.ones((4,))),
                10,
            )

        def step(
            self,
            key,
            params,
            state,
            action,
        ):
            return (
                dict(a=jnp.ones((4,)), b=jnp.ones((4,))),
                10,
                dict(a=0, b=0),
                dict(a=False, b=False),
            )

    env = TestEnv()

    n_epochs = 3
    n_env = 2
    n_env_steps = 5

    updated_agent, rewards, train_data = rl.train(
        k,
        agents,
        env,
        env.default_params,
        n_epochs,
        n_env,
        n_env_steps,
        show_progress=False,
    )

    assert isinstance(updated_agent, dict)
    assert isinstance(updated_agent["a"], Agent)
    assert isinstance(updated_agent["b"], Agent)
    assert isinstance(rewards, dict)
    assert rewards["a"].shape == (n_epochs, n_env, n_env_steps)
    assert rewards["b"].shape == (n_epochs, n_env, n_env_steps)
    assert isinstance(train_data, dict)
    assert train_data["a"][0].shape == (n_epochs,)
    assert train_data["a"][1].shape == (n_epochs,)
    assert train_data["b"][0].shape == (n_epochs,)
    assert train_data["b"][1].shape == (n_epochs,)
