"""
Batch simulation runner
"""

from typing import Optional, Union

import chex
import jax

from .sim import Sim, TSimParams
from .utils.functions import get_size


def batch_sim_runner(
    sim: Sim,
    n_samples: int,
    n_steps: int,
    rng: Union[chex.PRNGKey, int],
    show_progress: bool = True,
    params: Optional[TSimParams] = None,
    param_samples: Optional[TSimParams] = None,
    **step_kwargs,
) -> chex.ArrayTree:
    """
    Batch Monte-Carlo and parameter sweep simulation execution

    Run simulations across multiple random keys and optionally
    across a sample of simulation parameters. If run across
    parameters, then in will run ``n_samples`` per parameter
    sample.

    Parameters
    ----------
    sim
        Simulation functionality class.
    n_steps
        Number of simulation steps.
    n_samples
        Number of Monte-Carlo samples.
    rng
        Either a JAX random key or integer random seed.
    show_progress
        If ``True`` simulation progress bar will be displayed.
        Default ``True``.
    params
        Optional simulation parameters. If not provided default
        simulation parameters will be used.
    param_samples
        Optional simulation parameter samples to generate
        data across. Should have the same tree structure as individual
        simulation parameters.
    **step_kwargs
        Any additional keyword arguments passed to the
        step function. Arguments are static over the
        course of the simulation.

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

    def inner(k, i, _params):
        _, records = sim.init_and_run(
            n_steps,
            k,
            show_progress=show_progress,
            pbar_id=i,
            params=_params,
            **step_kwargs,
        )
        return records

    def sample_params(k, _params, pbar_offset):
        pbar_offset = pbar_offset * n_samples
        keys = jax.random.split(k, n_samples)
        return jax.vmap(inner, in_axes=(0, 0, None))(
            keys, pbar_offset + jax.numpy.arange(n_samples), _params
        )

    if params is not None:
        batch_records = sample_params(key, params, 0)
    else:
        n_params = get_size(param_samples)
        batch_records = jax.vmap(sample_params, in_axes=(None, 0, 0))(
            key, param_samples, jax.numpy.arange(n_params)
        )

    return batch_records
