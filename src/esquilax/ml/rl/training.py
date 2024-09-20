"""
RL training and testing functionality
"""

from typing import Callable, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax_tqdm

from esquilax.typing import TEnvParams, TypedPyTree

from . import tree_utils
from .agent import Agent
from .agent_state import AgentState, BatchAgentState
from .environment import Environment
from .types import Trajectory


def _step(
    agents: TypedPyTree[Agent], env_params: TEnvParams, env: Environment
) -> Callable:
    def inner_step(carry, _) -> Tuple[Tuple, Trajectory]:
        _k, _env_state, _obs, _agent_states = carry
        _k, _k_act, _k_step = jax.random.split(_k, 3)
        _actions, _action_values = tree_utils.sample_actions(
            agents, _k_act, _agent_states, _obs
        )
        _new_obs, _env_state, _rewards, _done = env.step(
            _k_step, env_params, _env_state, _actions
        )
        trajectories = jax.tree.map(
            lambda *x: Trajectory(
                obs=x[0],
                actions=x[1],
                action_values=x[2],
                rewards=x[3],
                done=x[4],
            ),
            _new_obs,
            _actions,
            _action_values,
            _rewards,
            _done,
        )
        return (
            (_k, _env_state, _new_obs, _agent_states),
            trajectories,
        )

    return inner_step


def train(
    key: chex.PRNGKey,
    agents: TypedPyTree[Agent],
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    env: Environment,
    env_params: TEnvParams,
    n_epochs: int,
    n_env: int,
    n_env_steps: int,
    show_progress: bool = True,
) -> Tuple[TypedPyTree[Agent], chex.ArrayTree, chex.ArrayTree]:
    """
    Train an RL-agent or agents with a given environment

    Parameters
    ----------
    key
        JAX random key.
    agents
        RL agent, or collection of agents functionality.
    agent_states
        Corresponding RL agent(s) states
    env
        Training environment/simulation. This should
        implement the :py:class:`esquilax.ml.rl.Environment`
        interface.
    env_params
        Environment parameters.
    n_epochs
        Number of training epochs.
    n_env
        Number of environments to train across per epoch.
    n_env_steps
        Number of steps to run in each environment per epoch.
    show_progress
        If ``True`` a training progress bar will be displayed.
        Default ``True``.

    Returns
    -------
    tuple
        Tuple containing:

        - Update agent or collection of agents
        - Training rewards
        - Additional training data
    """
    step = _step(agents, env_params, env)

    def sample_trajectories(_k, _agent_states):
        _obs, _state = env.reset(_k, env_params)
        _, trajectories = jax.lax.scan(
            step, (_k, _state, _obs, _agent_states), None, length=n_env_steps
        )
        return trajectories

    def epoch(carry, _):
        _k, _agent_states = carry
        _k, _k_sample, _k_train = jax.random.split(_k, 3)
        _k_sample = jax.random.split(_k_sample, n_env)
        _trajectories = jax.vmap(sample_trajectories, in_axes=(0, None))(
            _k_sample, _agent_states
        )
        _agent_states, _train_data = tree_utils.update_agents(
            agents, _k_train, _agent_states, _trajectories
        )

        return (_k, _agent_states), (
            jax.tree.map(
                lambda _, t: t.rewards,
                agents,
                _trajectories,
                is_leaf=lambda x: isinstance(x, Agent),
            ),
            _train_data,
        )

    if show_progress:
        epoch = jax_tqdm.scan_tqdm(n_epochs, desc="Epoch")(epoch)

    (_, agent_states), (rewards, train_data) = jax.lax.scan(
        epoch,
        (key, agent_states),
        jnp.arange(n_epochs),
    )

    return agent_states, rewards, train_data


def test(
    key: chex.PRNGKey,
    agents: TypedPyTree[Agent],
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    env: Environment,
    env_params: TEnvParams,
    n_env: int,
    n_env_steps: int,
    show_progress: bool = True,
) -> Trajectory:
    """
    Test agent(s) performance

    Asses agents against a test environment,
    returning trajectory data gathered over training.

    Parameters
    ----------
    key
        JAX random key.
    agents
        RL agent, or collection of agents functionality.
    agent_states
        Corresponding RL agent(s) states
    env
        Training environment/simulation. This should
        implement a Gymnax Environment base class.
    env_params
        Environment parameters.
    n_env
        Number of environments to test across.
    n_env_steps
        Number of steps to run in each environment.
    show_progress
        If ``True`` a testing progress bar will be displayed.
        Default ``True``

    Returns
    -------
    esquilax.ml.rl.Trajectory
        Update trajectories gathered over testing.
    """
    step = _step(agents, env_params, env)

    if show_progress:
        step = jax_tqdm.scan_tqdm(n_env_steps, desc="Step")(step)

    def sample_trajectories(_k, _agent_states):
        _obs, _state = env.reset(_k, env_params)
        _, _trajectories = jax.lax.scan(
            step, (_k, _state, _obs, _agent_states), None, length=n_env_steps
        )
        return _trajectories

    k_sample = jax.random.split(key, n_env)

    trajectories = jax.vmap(sample_trajectories, in_axes=(0, None))(
        k_sample, agent_states
    )

    return trajectories
