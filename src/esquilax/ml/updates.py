"""
Simulation update utilities for RL and neuro-evolution
"""
from functools import partial
from typing import Callable

import chex
import jax


@partial(jax.jit, static_argnames=("f", "broadcast"))
def get_actions(
    f: Callable, broadcast: bool, params: chex.ArrayTree, observations: chex.ArrayTree
) -> chex.ArrayTree:
    """
    Apply a function and parameters to observations

    Generate actions from a function and corresponding
    parameters applied to observation. Typically, this
    may be a neural-network forward pass, with network
    parameters, applied to observation of some
    simulation state.

    This is intended to sample actions across multiple
    agents, with either shared parameters, or parameters
    per agent (each with corresponding individual
    observations),

    Parameters
    ----------
    f
        Function parameterised by ``params`` (e.g.
        a neural-network apply function). Should
        have a signature :code:`(params, observation) -> x`.
    broadcast
        If ``True`` the parameters will be shared across
        all the observations (i.e. a shared policy),
        otherwise they will be mapped over.
    params
        Function parameters. Since shared
        by agents these should be a single sample
        of parameters.
    observations
        Array/tree of individual observation for agents
        to be mapped over.

    Returns
    -------
    chex.ArrayTree
        Output of ``f`` for each agent
    """
    in_axes = (None, 0) if broadcast else (0, 0)
    return jax.vmap(f, in_axes=in_axes)(params, observations)


@partial(jax.jit, static_argnames=("f", "broadcast"))
def sample_actions(
    f: Callable,
    broadcast: bool,
    key: chex.PRNGKey,
    params: chex.ArrayTree,
    observations: chex.ArrayTree,
) -> chex.ArrayTree:
    """
    Apply a function and parameters to observations with a random key

    Generate actions from a function and corresponding
    parameters applied to observation. Also passes
    a random key for use when some randomness is
    required, e.g. for policy exploration during RL training.

    Parameters
    ----------
    f
        Function parameterised by ``params`` (e.g.
        a neural-network apply function). Should
        have a signature :code:`(key, params, observation) -> x`.
    broadcast
        If ``True`` the parameters will be shared across
        all the observations (i.e. a shared policy),
        otherwise they will be mapped over.
    key
        JAX random key.
    params
        Function parameters. Since shared
        by agents these should be a single sample
        of parameters.
    observations
        Array/tree of individual observation for agents
        to be mapped over.

    Returns
    -------
    chex.ArrayTree
        Output of ``f`` for each agent.
    """
    in_axes = (0, None, 0) if broadcast else (0, 0, 0)
    keys = jax.random.split(key, observations.shape[0])
    return jax.vmap(f, in_axes=in_axes)(keys, params, observations)
