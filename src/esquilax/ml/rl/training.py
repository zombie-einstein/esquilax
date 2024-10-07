"""
RL training and testing functionality
"""
from functools import partial
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


def step(
    agents: TypedPyTree[Agent],
    env_params: TEnvParams,
    env: Environment,
    greedy: bool = False,
) -> Callable:
    """
    Single update of the training environment

    Returns a step function intended for use inside
    a JAX scan. The step samples agent actions
    from the current state and updates the environment.

    Parameters
    ----------
    agents
        PyTree of RL agents
    env_params
        Environment parameters
    env
        Training environment
    greedy
        Flag to be passed to RL agent, indicating sampling
        actions from a greedy policy

    Returns
    -------
    typing.Callable
        Step function for use in :py:meth:`jax.lax.scan`.
        Carries random keys, environment state and observations.
        Then returns transitions/trajectories and environment
        state to be recorded.
    """

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
            lambda _, *x: Trajectory(
                obs=x[0],
                actions=x[1],
                action_values=x[2],
                rewards=x[3],
                done=x[4],
            ),
            agents,
            _obs,
            _actions,
            _action_values,
            _rewards,
            _done,
            is_leaf=lambda x: isinstance(x, Agent),
        )
        return (
            (_k, _new_env_state, _new_obs, _agent_states),
            (_trajectories, _env_state),
        )

    return inner_step


def generate_samples(
    agents: TypedPyTree[Agent],
    env_params: TEnvParams,
    env: Environment,
    n_env_steps: int,
    key: chex.PRNGKey,
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    greedy: bool = False,
    show_progress: bool = False,
    pbar_id: int = 0,
) -> Tuple[Trajectory, TEnvState]:
    """
    Run the environment forward generating trajectory and state records

    Run the simulation environment and collecting trajectory and
    environment state records.

    Parameters
    ----------
    agents
        PyTree of RL agents.
    env_params
        Environment parameters.
    env
        Training environment.
    n_env_steps
        Number of steps to run environment.
    key
        JAX random key
    agent_states
        PyTree of RL agent states
    greedy
        Flag to be passed to RL agent, indicating sampling
        actions from a greedy policy.
    show_progress
        If ``True`` a progress bar will show execution progress.
    pbar_id
        Optional progress bar index. Can be used to print
        to multiple progress bars.

    Returns
    -------
    tuple[esquilax.ml.rl.Trajectory, esquilax.typing.TEnvState]
        Environment observation-action trajectories and
        recorded environment states. Trajectories have a shape
        ``[n-steps, n-agents]``.
    """
    step_fun = step(agents, env_params, env, greedy=greedy)

    k_reset, k_run = jax.random.split(key, 2)
    obs, env_state = env.reset(k_reset, env_params)
    init = (k_run, env_state, obs, agent_states)

    if show_progress:
        step_fun = jax_tqdm.scan_tqdm(n_env_steps, desc="Step")(step_fun)
        init = jax_tqdm.PBar(id=pbar_id, carry=init)

    _, (trajectories, env_states) = jax.lax.scan(
        step_fun, init, jnp.arange(n_env_steps)
    )
    return trajectories, env_states


def batch_generate_samples(
    agents: TypedPyTree[Agent],
    env_params: TEnvParams,
    env: Environment,
    n_env_steps: int,
    n_env: int,
    key: chex.PRNGKey,
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    greedy: bool = False,
    show_progress: bool = False,
) -> Tuple[Trajectory, TEnvState]:
    """
    Sample trajectories across multiple environments

    Generate samples across multiple environments, across
    random seeds.

    Parameters
    ----------
    agents
        PyTree of RL agents.
    env_params
        Environment parameters.
    env
        Training environment.
    n_env_steps
        Number of steps to run environment.
    n_env
        Number of environments to execute.
    key
        JAX random key
    agent_states
        PyTree of RL agent states
    greedy
        Flag to be passed to RL agent, indicating sampling
        actions from a greedy policy.
    show_progress
        If ``True`` a progress bar will show execution progress.

    Returns
    -------
    tuple[esquilax.ml.rl.Trajectory, esquilax.typing.TEnvState]
        Environment observation-action trajectories and
        recorded environment states. Trajectories have a shape
        ``[n-env, n-steps, n-agents]``.
    """
    sampling_func = partial(
        generate_samples,
        agents,
        env_params,
        env,
        n_env_steps,
        greedy=greedy,
        show_progress=show_progress,
    )
    keys = jax.random.split(key, n_env)
    trajectories, env_states = jax.vmap(
        lambda k, a, i: sampling_func(k, a, pbar_id=i), in_axes=(0, None, 0)
    )(keys, agent_states, jnp.arange(n_env))
    return trajectories, env_states


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

    batch_sampling_func = partial(
        batch_generate_samples,
        agents,
        env_params,
        env,
        n_env_steps,
        n_env,
        greedy=False,
        show_progress=False,
    )

    def epoch(carry, _):
        _k, _agent_states = carry
        _k, _k_sample, _k_train = jax.random.split(_k, 3)
        _trajectories, _ = batch_sampling_func(_k_sample, _agent_states)
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
    sampling_func = partial(
        generate_samples,
        agents,
        env_params,
        env,
        n_env_steps,
        greedy=greedy_actions,
        show_progress=show_progress,
    )

    def sample_trajectories(_k, i, _agent_states):
        _trajectories, _states = sampling_func(_k, _agent_states, pbar_id=i)
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

    states, recorded = jax.vmap(sample_trajectories, in_axes=(0, 0, None))(
        k_sample, jnp.arange(n_env), agent_states
    )

    return states, recorded


def train_and_test(
    key: chex.PRNGKey,
    agents: TypedPyTree[Agent],
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    env: Environment,
    env_params: TEnvParams,
    n_train_steps: int,
    test_every: int,
    n_train_env: int,
    n_test_env: int,
    n_env_steps: int,
    show_progress: bool = True,
    return_trajectories: bool = False,
    greedy_test_actions: bool = False,
) -> Tuple[
    TypedPyTree[Union[AgentState, BatchAgentState]],
    chex.Array,
    chex.ArrayTree,
    chex.ArrayTree,
    chex.ArrayTree,
]:
    """
    RL train and test loop

    Train an RL agent(s) and test at fixed intervals.
    Agent rewards and losses are tracked over the
    course of training. Rewards (or optionally state
    transitions) and the state of the environment are
    recorded during testing.

    It's assumed that collecting samples across multiple
    agents and environments so the agent training is
    performed each training step.

    Parameters
    ----------
    key
        JAX random key.
    agents
        PyTree of RL agent definitions.
    agent_states
        PyTree of RL agent states.
    env
        Training environment definition.
    env_params
        Environment parameters.
    n_train_steps
        Total number of training steps.
    test_every
        Number of steps between testing.
    n_train_env
        Number of environments to run during training.
    n_test_env
        Number of environments to run during testing.
    n_env_steps
        Number of environment updates per episode.
    show_progress
        If ``True`` training/testing progress will
        be displayed.
    return_trajectories
        If ``True`` during testing trajectory data will be recorded
        along with environment state. If ``False`` the state and
        rewards will be recorded. Default ``False``.

        .. warning::

           Recording trajectories could result in a large amount
           of data (given it will record observations, actions etc.
           for each individual agent at each step).

    greedy_test_actions
        If ``True`` greedy action sampling flag will be passed
        to the RL agent(s) during testing.

    Returns
    -------
    tuple
        Tuple containing

        - PyTree of trained agent states
        - Individual agent rewarded recorded over training
        - Loss data recorded during training
        - Env state history recorded during testing
        - Rewards or trajectories recorded during testing
    """
    assert (
        n_train_steps % test_every == 0
    ), "n_train_steps should be a multiple of test_every"

    n_steps = n_train_steps // test_every

    def train_test_step(carry, _):
        _key, _agent_states = carry

        _key, k_train, k_test = jax.random.split(_key, 3)
        _agent_states, _train_rewards, _train_losses = train(
            k_train,
            agents,
            _agent_states,
            env,
            env_params,
            test_every,
            n_train_env,
            n_env_steps,
            show_progress=show_progress,
        )
        _env_state_records, _test_rewards = test(
            k_test,
            agents,
            _agent_states,
            env,
            env_params,
            n_test_env,
            n_env_steps,
            show_progress=show_progress,
            return_trajectories=return_trajectories,
            greedy_actions=greedy_test_actions,
        )
        return (_key, _agent_states), (
            _train_rewards,
            _train_losses,
            _env_state_records,
            _test_rewards,
        )

    (_, trained_agent_states), (
        train_rewards,
        train_losses,
        env_state_records,
        test_rewards,
    ) = jax.lax.scan(train_test_step, (key, agent_states), None, length=n_steps)

    def reshape_train_data(x):
        s0 = x.shape[0] * x.shape[1]
        return jnp.reshape(x, (s0,) + x.shape[2:])

    train_rewards = reshape_train_data(train_rewards)
    train_losses = jax.tree.map(reshape_train_data, train_losses)

    return (
        trained_agent_states,
        train_rewards,
        train_losses,
        env_state_records,
        test_rewards,
    )
