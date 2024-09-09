"""
Batch simulation runner
"""

from typing import Optional, Union

import chex
import jax

from .env import Sim, TSimParams


def batch_sim_runner(
    sim: Sim,
    n_samples: int,
    n_steps: int,
    rng: Union[chex.PRNGKey, int],
    show_progress: bool = True,
    params: Optional[TSimParams] = None,
    param_samples: Optional[TSimParams] = None,
) -> chex.ArrayTree:
    """
    Batch Monte-Carlo and parameter sweep simulation execution

    Run simulations across multiple random keys and optionally
    across a sample of simulation parameters.

    Parameters
    ----------
    sim: Sim
        Simulation functionality class.
    n_steps: int
        Number of simulation steps.
    n_samples: int
        Number of Monte-Carlo samples.
    rng: int or jax.random.PRNGKey
        Either a JAX random key or integer random seed.
    show_progress: bool, optional
        If ``True`` simulation progress bar will be displayed.
        Default ``True``.
    params: TSimParams, optional
        Optional simulation parameters. If not provided default
        simulation parameters will be used.
    param_samples: TSimParams, optional
        Optional simulation parameter samples to generate
        data across. Should have the same tree structure as individual
        simulation parameters.

    Returns
    -------
    Any
        Simulation records, with shape ``[n_samples, n_steps]`` if run
        with a single set of parameters, or shape ``[n_params, n_samples, n_steps]``
        if a set of parameters is provided.
    """
    key = rng if isinstance(rng, chex.PRNGKey) else jax.random.PRNGKey(rng)

    assert (
        params is None or param_samples is None
    ), "Either params or param_samples should be provided, not both"

    if params is None and param_samples is None:
        params = sim.default_params()

    def inner(k, _params):
        k1, k2 = jax.random.split(k)
        _initial_state = sim.initial_state(k1)
        _, records, _ = sim.run(
            n_steps, k2, _params, _initial_state, show_progress=show_progress
        )
        return records

    def sample_params(k, _params):
        keys = jax.random.split(k, n_samples)
        return jax.vmap(inner, in_axes=(0, None))(keys, _params)

    if params is not None:
        batch_records = sample_params(key, params)
    else:
        batch_records = jax.vmap(sample_params, in_axes=(None, 0))(key, param_samples)

    return batch_records
