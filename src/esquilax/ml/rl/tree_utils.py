"""
Tree-mapping utilities
"""
from typing import Any, Tuple

import chex
import jax

from esquilax.ml import common

from .agents import Agent
from .types import Trajectory


def sample_actions(
    key: chex.PRNGKey, agents: common.TypedPyTree[Agent], observations: chex.ArrayTree
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
    observations
        PyTree of observations, with tree structure corresponding
        to the agents.

    Returns
    -------
    tuple
        Tuple containing actions, and any additional values
        associated with the actions. Both have the same
        tree structure as the argument agents.
    """
    keys = common.key_tree_split(key, agents, typ=Agent)
    results = jax.tree.map(
        lambda agent, k, obs: agent.sample_actions(k, obs),
        agents,
        keys,
        observations,
        is_leaf=lambda x: isinstance(x, Agent),
    )
    actions, action_values = common.transpose_tree_of_tuples(agents, results, 2, Agent)
    return actions, action_values


def update_agents(
    key: chex.PRNGKey,
    agents: common.TypedPyTree[Agent],
    trajectories: common.TypedPyTree[Trajectory],
) -> Tuple[common.TypedPyTree[Agent], Any]:
    """
    Update agent states from gathered trajectories

    Parameters
    ----------
    key
        JAX random key.
    agents
        PyTree of RL-agents.
    trajectories
        PyTree of environment trajectories.

    Returns
    -------
    tuple
        Tuple containing PyTrees of updated
        agents, and any data returned from training
        (e.g. training loss). Both trees have the same
        structure as the argument agents.
    """
    keys = common.key_tree_split(key, agents, typ=Agent)

    updates = jax.tree.map(
        lambda agent, k, t: agent.update(k, t),
        agents,
        keys,
        trajectories,
        is_leaf=lambda x: isinstance(x, Agent),
    )
    agents, train_data = common.transpose_tree_of_tuples(agents, updates, 2, Agent)
    return agents, train_data
