"""
Simulation update utilities for RL and neuro-evolution
"""
from functools import partial
from typing import Callable

import chex
import jax


@partial(jax.jit, static_argnames=("f",))
def broadcast_params(
    f: Callable, params: chex.ArrayTree, obs: chex.ArrayTree
) -> chex.ArrayTree:
    """
    Broadcast function parameters over observations

    Applies parameters where agents all share
    the same parameters, across their individual
    observations, returning an output for
    each agent.

    Parameters
    ----------
    f: Callable
        Function parameterised by ``params`` (e.g.
        a neural-network apply function).
    params: chex.ArrayTree
        Function parameters. Since shared
        by agents these should be a single sample
        of parameters.
    obs: chex.ArrayTree
        Array/tree of individual observation for agents
        to be mapped over.

    Returns
    -------
    chex.ArrayTree
        Output of ``f`` for each agent
    """
    return jax.vmap(f, in_axes=(None, 0))(params, obs)


@partial(jax.jit, static_argnames=("f",))
def map_params(
    f: Callable, params: chex.ArrayTree, obs: chex.ArrayTree
) -> chex.ArrayTree:
    """
    Map a sample of parameters over observations

    Applies a sample of parameters across agents
    individual observations, returning an output for
    each agent.

    Parameters
    ----------
    f: Callable
        Function parameterised by ``params`` (e.g.
        a neural-network apply function).
    params: chex.ArrayTree
        Population of Evosax function parameters.
    obs: chex.ArrayTree
        Array/tree of individual observation for agents
        to be mapped over.

    Returns
    -------
    chex.ArrayTree
        Output of ``f`` for each agent
    """
    return jax.vmap(f, in_axes=(0, 0))(params, obs)
