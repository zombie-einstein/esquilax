"""
Agent PyTree mapping utilities
"""
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp

from esquilax.typing import TypedPyTree
from esquilax.utils import tree

from .agent import Agent
from .agent_state import AgentState, BatchAgentState
from .types import Trajectory


def sample_actions(
    agents: TypedPyTree[Agent],
    key: chex.PRNGKey,
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    observations: chex.ArrayTree,
    greedy: bool = False,
) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
    """
    Map over a tree of agents, sampling actions from observations

    Can be used to sample actions across multiple
    RL policies.

    Parameters
    ----------
    key
        JAX random key.
    agents
        Pytree of RL Agents.
    agent_states
        Pytree of corresponding agent states
    observations
        PyTree of observations, with tree structure corresponding
        to the agents.
    greedy
        Flag to indicate greedy action sampling.

    Returns
    -------
    tuple[chex.ArrayTree, chex.ArrayTree]
        Tuple containing actions, and any additional values
        associated with the actions. Both have the same
        tree structure as the argument agents.
    """
    keys = tree.key_tree_split(key, agents, typ=Agent)
    results = jax.tree.map(
        lambda agent, k, state, obs: agent.sample_actions(k, state, obs, greedy=greedy),
        agents,
        keys,
        agent_states,
        observations,
        is_leaf=lambda x: isinstance(x, Agent),
    )
    actions, action_values = tree.transpose_tree_of_tuples(agents, results, 2, Agent)
    return actions, action_values


def update_agents(
    agents: TypedPyTree[Agent],
    key: chex.PRNGKey,
    agent_states: TypedPyTree[Union[AgentState, BatchAgentState]],
    trajectories: TypedPyTree[Trajectory],
) -> Tuple[TypedPyTree[Agent], chex.ArrayTree]:
    """
    Update agent states from gathered trajectories

    Parameters
    ----------
    key
        JAX random key.
    agents
        PyTree of RL-agents.
    agent_states
        Pytree of corresponding agent states
    trajectories
        PyTree of environment trajectories.

    Returns
    -------
    tuple[esquilax.typing.TypedPyTree[esquilax.ml.rl.AgentState], chex.ArrayTree]
        Tuple containing PyTrees of updated
        agents, and any data returned from training
        (e.g. training loss). Both trees have the same
        structure as the argument agents.
    """
    keys = tree.key_tree_split(key, agents, typ=Agent)

    updates = jax.tree.map(
        lambda agent, k, state, traj: agent.update(k, state, traj),
        agents,
        keys,
        agent_states,
        trajectories,
        is_leaf=lambda x: isinstance(x, Agent),
    )
    agents, train_data = tree.transpose_tree_of_tuples(agents, updates, 2, Agent)
    return agents, train_data


def reshape_trajectories(trajectories: Trajectory) -> Trajectory:
    """
    Reshape a batch of trajectories into individual agent histories

    Reshape batch of trajectories gathered across
    multiple environments, i.e. with shape
    ``[n-envs, n_steps, n-agents, ...]`` into trajectory
    histories per agent and environment, i.e. with shape
    ``[n-envs * n-agents, n_steps, ...]``.

    Parameters
    ----------
    trajectories
        Batch of trajectories with array shapes
        ``[n-envs, n_steps, n-agents, ...]``.

    Returns
    -------
    esquilax.ml.rl.Trajectory
        Reshaped trajectories with shape
        ``[n-envs * n-agents, n_steps, ...]``.
    """

    def reshape(x):
        x = jnp.swapaxes(x, 1, 2)
        x = jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])
        return x

    trajectories = jax.tree.map(reshape, trajectories)

    return trajectories
