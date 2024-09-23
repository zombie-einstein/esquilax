from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from esquilax.ml import rl
from esquilax.ml.rl import Trajectory

from .conftest import SimpleModel


def test_agent_state():
    n_agents = 10
    observation_shape = (4,)

    k = jax.random.key(451)

    state = rl.AgentState.init_from_model(
        k, SimpleModel(), optax.adam(1e-4), observation_shape
    )

    a = jnp.ones((n_agents,) + observation_shape)
    actions = state.apply(a)

    assert actions.shape == (n_agents,)


def test_batch_agent_state():
    n_agents = 10
    observation_shape = (4,)

    k = jax.random.key(451)

    state = rl.BatchAgentState.init_from_model(
        k, SimpleModel(), optax.adam(1e-4), observation_shape, n_agents
    )

    assert state.step.shape == (n_agents,)
    assert jnp.array_equal(state.step, jnp.zeros((n_agents,), dtype=jnp.int32))
    assert state.params["params"]["Dense_0"]["bias"].shape == (n_agents, 2)
    assert state.params["params"]["Dense_0"]["kernel"].shape == (
        n_agents,
    ) + observation_shape + (2,)

    a = jnp.ones((n_agents,) + observation_shape)

    def loss(p, x):
        return state.apply_fn(p, x)

    grads = jax.vmap(jax.grad(loss), in_axes=(0, 0))(state.params, a)

    updated_state = state.apply_gradients(grads=grads)

    assert isinstance(updated_state, rl.BatchAgentState)
    assert jnp.array_equal(updated_state.step, jnp.ones((n_agents,), dtype=jnp.int32))
    assert updated_state.params["params"]["Dense_0"]["bias"].shape == (n_agents, 2)
    assert updated_state.params["params"]["Dense_0"]["kernel"].shape == (
        n_agents,
    ) + observation_shape + (2,)

    actions = state.apply(a)

    assert actions.shape == (n_agents,)


class Agent(rl.Agent):
    def sample_actions(
        self,
        key: chex.PRNGKey,
        agent_state: rl.AgentState,
        observations: chex.Array,
        greedy: bool = False,
    ) -> Tuple[rl.AgentState, chex.ArrayTree]:
        return agent_state.apply_fn(agent_state.params, observations), (10, 11)

    def update(
        self,
        key: chex.PRNGKey,
        agent_state: rl.AgentState,
        trajectories,
    ) -> Tuple[rl.AgentState, chex.ArrayTree]:
        return agent_state, (10, 10)


def test_training_shared_policy():
    k = jax.random.key(451)
    observation_shape = (4,)

    agent_state = rl.AgentState.init_from_model(
        k, SimpleModel(), optax.adam(1e-4), observation_shape
    )

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
        Agent(),
        agent_state,
        env,
        env.default_params,
        n_epochs,
        n_env,
        n_env_steps,
        show_progress=False,
    )

    assert isinstance(updated_agent, rl.AgentState)
    assert rewards.shape == (n_epochs, n_env, n_env_steps)
    assert isinstance(train_data, tuple)
    assert train_data[0].shape == (n_epochs,)
    assert train_data[1].shape == (n_epochs,)

    states, trajectories = rl.test(
        k,
        Agent(),
        agent_state,
        env,
        env.default_params,
        n_env,
        n_env_steps,
        show_progress=False,
        return_trajectories=True,
    )

    assert isinstance(trajectories, rl.Trajectory)
    assert trajectories.rewards.shape == (n_env, n_env_steps)
    assert trajectories.obs.shape == (n_env, n_env_steps) + observation_shape
    assert trajectories.actions.shape == (n_env, n_env_steps)
    assert isinstance(trajectories.action_values, tuple)
    assert trajectories.action_values[0].shape == (n_env, n_env_steps)
    assert trajectories.action_values[1].shape == (n_env, n_env_steps)
    assert jnp.array_equal(
        trajectories.action_values[0],
        10 * jnp.ones((n_env, n_env_steps), dtype=jnp.int32),
    )
    assert jnp.array_equal(
        trajectories.action_values[1],
        11 * jnp.ones((n_env, n_env_steps), dtype=jnp.int32),
    )
    assert states.shape == (n_env, n_env_steps)


def test_training_multi_policy():
    k = jax.random.key(451)
    observation_shape_a = (4,)
    observation_shape_b = (3,)

    agents = dict(a=Agent(), b=Agent())
    agent_states = dict(
        a=rl.AgentState.init_from_model(
            k, SimpleModel(), optax.adam(1e-4), observation_shape_a
        ),
        b=rl.AgentState.init_from_model(
            k, SimpleModel(), optax.adam(1e-4), observation_shape_b
        ),
    )

    class TestEnv(rl.Environment):
        def reset(self, key, params):
            return (
                dict(a=jnp.ones(observation_shape_a), b=jnp.ones(observation_shape_b)),
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
                dict(a=jnp.ones(observation_shape_a), b=jnp.ones(observation_shape_b)),
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
        agent_states,
        env,
        env.default_params,
        n_epochs,
        n_env,
        n_env_steps,
        show_progress=False,
    )

    assert isinstance(updated_agent, dict)
    assert isinstance(updated_agent["a"], rl.AgentState)
    assert isinstance(updated_agent["b"], rl.AgentState)
    assert isinstance(rewards, dict)
    assert rewards["a"].shape == (n_epochs, n_env, n_env_steps)
    assert rewards["b"].shape == (n_epochs, n_env, n_env_steps)
    assert isinstance(train_data, dict)
    assert train_data["a"][0].shape == (n_epochs,)
    assert train_data["a"][1].shape == (n_epochs,)
    assert train_data["b"][0].shape == (n_epochs,)
    assert train_data["b"][1].shape == (n_epochs,)

    states, trajectories = rl.test(
        k,
        agents,
        agent_states,
        env,
        env.default_params,
        n_env,
        n_env_steps,
        show_progress=False,
        return_trajectories=True,
    )

    assert isinstance(trajectories, dict)
    assert sorted(trajectories.keys()) == ["a", "b"]
    for v in trajectories.values():
        assert isinstance(v, Trajectory)
        assert v.actions.shape == (n_env, n_env_steps)
        assert v.rewards.shape == (n_env, n_env_steps)
        assert isinstance(v.action_values, tuple)
        assert v.action_values[0].shape == (n_env, n_env_steps)
        assert jnp.array_equal(
            v.action_values[0], 10 * jnp.ones((n_env, n_env_steps), dtype=jnp.int32)
        )
        assert v.action_values[1].shape == (n_env, n_env_steps)
        assert jnp.array_equal(
            v.action_values[1], 11 * jnp.ones((n_env, n_env_steps), dtype=jnp.int32)
        )

    assert trajectories["a"].obs.shape == (n_env, n_env_steps) + observation_shape_a
    assert trajectories["b"].obs.shape == (n_env, n_env_steps) + observation_shape_b

    assert states.shape == (n_env, n_env_steps)
