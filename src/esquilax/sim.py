"""
Abstract class wrapping common simulation functionality
"""
from functools import partial
from typing import Any, Generic, Optional, Tuple

import chex
import jax

from esquilax.runner import sim_runner

from .typing import TSimParams, TSimState


class Sim(Generic[TSimParams, TSimState]):
    """
    Base class wrapping simulation functionality for batch execution and ML use-cases

    Simulation environment base-class with methods used by training functions.
    For use inside JIT compiled training loops, static simulation
    parameters can be assigned as attributes of the derived class.
    """

    @partial(jax.jit, static_argnums=(0,))
    def default_params(self) -> TSimParams:
        """
        Return default simulation parameters

        Returns
        -------
        esquilax.typing.TSimParams
            Simulation parameters
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def initial_state(self, key: chex.PRNGKey, params: TSimParams) -> TSimState:
        """
        Initialise the initial state of the simulation

        Parameters
        ----------
        key
            JAX random key.
        params
            Simulation parameters

        Returns
        -------
        esquilax.typing.TSimState
            The initial state of the environment.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        i: int,
        key: chex.PRNGKey,
        params: TSimParams,
        state: TSimState,
        **kwargs: Any
    ) -> Tuple[TSimState, chex.ArrayTree]:
        """
        A single step/update of the environment

        The step function should return a tuple containing the updated
        simulation state, and any data to be recorded each step
        (see :py:meth:`esquilax.runner.sim_runner` for more details).
        For example:

        .. code-block:: python

           class Sim(SimEnv):
               def step(self, i, k, params, state):
                   ...
                   return new_state, records

        Any static arguments required by the simulation can
        be accessed from the ``self`` argument of the method.

        Parameters
        ----------
        i
            Current step number
        key
            JAX random key
        params
            Simulation time-independent parameters
        state
            Simulation state
        **kwargs
            Any additional keyword arguments.

        Returns
        -------
        tuple[esquilax.typing.TSimState, chex.ArrayTree]
            Tuple containing the updated simulation state, and
            any data to be recorded over the course of the simulation.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("show_progress",))
    def run(
        self,
        n_steps: int,
        key: chex.PRNGKey,
        params: TSimParams,
        initial_state: TSimState,
        show_progress: bool = True,
        **step_kwargs: Any
    ) -> Tuple[TSimState, chex.ArrayTree, chex.PRNGKey]:
        """
        Convenience function to run the simulation for a fixed number of steps

        Parameters
        ----------
        n_steps
            Number of simulation steps
        key
            JAX random key
        params
            Simulation time-independent parameters
        initial_state
            Initial state of the simulation
        show_progress
            If ``True`` a progress bar will be displayed.
            Default ``True``
        **step_kwargs
            Any additional keyword arguments passed to the
            step function. Arguments are static over the
            course of the simulation.

        Returns
        -------
        tuple[esquilax.typing.TSimState, chex.ArrayTree, jax,random.PRNGKey]
            Tuple containing

            - The final state of the simulation
            - Tree of recorded data
            - Updated JAX random key
        """
        final_state, records, k = sim_runner(
            partial(self.step, **step_kwargs),
            params,
            initial_state,
            n_steps,
            key,
            show_progress=show_progress,
        )

        return final_state, records, k

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("show_progress",))
    def init_and_run(
        self,
        n_steps: int,
        key: chex.PRNGKey,
        show_progress: bool = True,
        params: Optional[TSimParams] = None,
        **step_kwargs
    ) -> Tuple[TSimState, chex.ArrayTree]:
        """
        Convenience function to initialise and run the simulation

        Parameters
        ----------
        n_steps
            Number of simulation steps to run
        key
            JAX random key
        show_progress
            If ``True`` a progress bar will be displayed.
            Default ``True``
        params
            Optional simulation parameters, if not provided
            default sim parameters will be used.
        **step_kwargs
            Any additional keyword arguments passed to the
            step function. Arguments are static over the
            course of the simulation.

        Returns
        -------
        tuple[esquilax.typing.TSimState, chex.ArrayTree, jax,random.PRNGKey]
            Tuple containing

            - The final state of the simulation
            - Tree of recorded data
            - Updated JAX random key
        """
        k1, k2 = jax.random.split(key, 2)

        params = self.default_params() if params is None else params
        initial_state = self.initial_state(k1, params)
        final_state, records, _ = self.run(
            n_steps,
            k2,
            params,
            initial_state,
            show_progress=show_progress,
            **step_kwargs
        )

        return final_state, records
