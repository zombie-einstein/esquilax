"""
Utilities for handling PyTrees of :py:class:`esquilax.ml.evo.strategy`
"""
import chex
import evosax
import jax

from esquilax.ml import common

from .strategy import Strategy


def tree_ask(
    key: chex.PRNGKey,
    strategies: common.TypedPyTree[Strategy],
    evo_states: common.TypedPyTree[evosax.EvoState],
    evo_params: common.TypedPyTree[evosax.EvoParams],
):
    """
    Call ``ask`` on PyTree of strategies, generating population samples

    Generate a new population sample of parameters, and
    also reshape those parameters for use in  a
    function/neural network.

    Parameters
    ----------
    key: jax.random.PRNGKey
        JAX random key
    strategies: TypedPyTree[Strategy]
        PyTree of strategies, could be a single strategy
        or a container/struct of strategies.
    evo_states: TypedPyTree[evosax.EvoState]
        PyTree of strategy states
    evo_params: TypedPyTree[evosax.EvoParams]
        PyTree of strategy parameters

    Returns
    -------
    tuple
        Tuple containing:

        - Updated strategy states
        - Population samples
        - Population samples reshaped for use in
          a function/neural network

        Each will have the same tree structure as
        the input arguments.
    """
    keys = common.key_tree_split(key, strategies, Strategy)

    def inner(strat, k, state, params):
        pop, state = strat.ask(k, state, params)
        pop_shaped = strat.reshape_params(pop)
        return state, pop, pop_shaped

    results = jax.tree.map(
        inner,
        strategies,
        keys,
        evo_states,
        evo_params,
        is_leaf=lambda x: isinstance(x, Strategy),
    )
    evo_states, population, population_shaped = common.transpose_tree_of_tuples(
        strategies, results, 3, Strategy
    )
    return evo_states, population, population_shaped


def tree_tell(
    strategies: common.TypedPyTree[Strategy],
    populations: chex.ArrayTree,
    rewards: chex.ArrayTree,
    evo_states: common.TypedPyTree[evosax.EvoState],
    evo_params: common.TypedPyTree[evosax.EvoParams],
):
    """
    Call ``tell`` on a PyTree of strategies updating state

    Reshape/rescale rewards, and use this to update the
    state of the strategy.

    Parameters
    ----------
    strategies: TypedPyTree[Strategy]
        PyTree of strategies, could be a single strategy
        or a container/struct of strategies.
    populations: chex.ArrayTree
        PyTree of populations.
    rewards: chex.ArrayTree
        PyTree of rewards.
    evo_states: TypedPyTree[evosax.EvoState]
        PyTree of strategy states.
    evo_params: TypedPyTree[evosax.EvoParams]
        PyTree of strategy parameters.

    Returns
    -------
    tuple
        Tuple containing:

        - PyTree of fitness values (i.e. the rescaled,
          reshaped rewards).
        - PyTree of updated strategy states.

        Each has the same tree structure as the arguments.
    """

    def inner(strat, pop, rew, state, params):
        f = strat.shape_rewards(pop, rew)
        state = strat.tell(pop, f, state, params)
        return f, state

    updates = jax.tree.map(
        inner,
        strategies,
        populations,
        rewards,
        evo_states,
        evo_params,
        is_leaf=lambda x: isinstance(x, Strategy),
    )
    fitness, evo_states = common.transpose_tree_of_tuples(
        strategies, updates, 2, Strategy
    )
    return fitness, evo_states
