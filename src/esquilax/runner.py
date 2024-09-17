"""
Simulation and experiment runners
"""
from functools import partial
from typing import Any, Callable, Tuple, Union

import chex
import jax
import jax_tqdm


def sim_runner(
    step_fun: Callable,
    params: Any,
    initial_state: chex.ArrayTree,
    n_steps: int,
    rng: Union[chex.PRNGKey, int],
    show_progress: bool = True,
    **static_kwargs,
) -> Tuple[Any, Any, chex.PRNGKey]:
    """
    Run a simulation and track state

    Repeated applies a provided simulation update function and
    records simulation state over the course of execution.

    Parameters
    ----------
    step_fun: Callable
        Update function that should have the signature

        .. code-block:: python

           def step(i, k, params, state, **static_kwargs):
               ...
               return new_state, records

        where the arguments are

        - ``i``: The current step number
        - ``k``: A JAX random key
        - ``params``: Sim parameters
        - ``state``: The current simulation state
        - ``**static_kwargs``: Static keyword arguments

        and returns

        - ``new_state``: Updated simulation state
        - ``records``: State data to be recorded

    params
        Simulation parameters. Parameters are constant over the
        course of the simulation.
    initial_state
        Initial simulation state.
    n_steps
        Number of steps to run.
    rng
        Either an integer random seed, or a JAX PRNGKey.
    show_progress
        If ``True`` a progress bar will be shown.
    **static_kwargs
        Any keyword static values passed to the step function.
        These should be used for any values or functionality required
        to be known at compile time by JAX.

    Returns
    -------
    [Any, Any, chex.PRNGKey]
        Tuple containing

        - The final state of the simulation
        - Recorded values
        - Update random key
    """

    key = rng if isinstance(rng, chex.PRNGKey) else jax.random.PRNGKey(rng)
    step_fun = partial(step_fun, **static_kwargs)

    def step(carry, i):
        k, state = carry
        k, step_key = jax.random.split(k)
        new_state, records = step_fun(i, step_key, params, state)
        return (k, new_state), records

    if show_progress:
        step = jax_tqdm.scan_tqdm(n_steps, desc="Step")(step)

    (key, final_state), record_history = jax.lax.scan(
        step, (key, initial_state), jax.numpy.arange(n_steps)
    )

    return final_state, record_history, key
