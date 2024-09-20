"""
RL training and testing functionality
"""

from typing import Callable, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax_tqdm

from esquilax.typing import TEnvParams, TEnvState, TypedPyTree

from . import tree_utils
from .agent import Agent
from .agent_state import AgentState, BatchAgentState
from .environment import Environment
from .types import Trajectory


def _step(
    agents: TypedPyTree[Agent],
    env_params: TEnvParams,
    env: Environment,
    greedy: bool = False,
) -> Callable:
    def inner_step(carry, _) -> Tuple[Tuple, Tuple[Trajectory, TEnvState]]:
        _k, _env_state, _obs, _agent_states = carry
        _k, _k_act, _k_step = jax.random.split(_k, 3)
        _actions, _action_values = tree_utils.sample_actions(
            agents,
            _k_act,
            _agent_states,
            _obs,
            greedy=greedy,
        )
        _new_obs, _new_env_state, _rewards, _done = env.step(
            _k_step, env_params, _env_state, _actions
        )
        _trajectories = jax.tree.map(
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
            (_k, _new_env_state, _new_obs, _agent_states),
            (_trajectories, _env_state),
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
        _, (trajectories, _) = jax.lax.scan(
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
    return_trajectories: bool = False,
    greedy_actions: bool = False,
) -> Tuple[TEnvState, Union[Trajectory, chex.ArrayTree]]:
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
    return_trajectories
        If ``True`` recorded trajectory data will be returned
        along with recorded state. If ``False`` the states and
        recorded rewards will be returned. Default ``False``.

        .. warning::

           Recording trajectories could result in a large amount
           of data (given it will record observations, actions etc.
           for each individual agent).

    greedy_actions
        Flag to indicate actions should be greedily sampled.

    Returns
    -------
    tuple[esquilax.typing.TEnvState, esquilax.ml.rl.Trajectory | chex.ArrayTree]
        Update trajectories gathered over testing.
    """
    step = _step(agents, env_params, env, greedy=greedy_actions)

    if show_progress:
        step = jax_tqdm.scan_tqdm(n_env_steps, desc="Step")(step)

    def sample_trajectories(_k, _agent_states):
        _obs, _state = env.reset(_k, env_params)
        _, (_trajectories, _states) = jax.lax.scan(
            step, (_k, _state, _obs, _agent_states), None, length=n_env_steps
        )
        print(_trajectories)
        if return_trajectories:
            return _states, _trajectories
        else:
            return _states, jax.tree.map(
                lambda _, x: x.rewards,
                agents,
                _trajectories,
                is_leaf=lambda x: isinstance(x, Agent),
            )

    k_sample = jax.random.split(key, n_env)

    states, recorded = jax.vmap(sample_trajectories, in_axes=(0, None))(
        k_sample, agent_states
    )

    return states, recorded
