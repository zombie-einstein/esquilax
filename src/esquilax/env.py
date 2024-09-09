"""
Abstract class wrapping common simulation functionality
"""
from functools import partial
from typing import Generic, Optional, Tuple, TypeVar

import chex
import jax

from esquilax.runner import sim_runner

TSimState = TypeVar("TSimState")
TSimParams = TypeVar("TSimParams")


class Sim(Generic[TSimParams, TSimState]):
    """
    Base class wrapping simulation functionality for use in training loops

    Simulation environment base-class with methods used by training functions.
    For use inside JIT compiled training loops, static simulation
    parameters can be assigned as attributes of the derived class.
    """

    @partial(jax.jit, static_argnums=(0,))
    def default_params(self) -> TSimParams:
        """
        Return default simulation parameters

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        TSimParams
            Simulation parameters
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def initial_state(self, k: chex.PRNGKey) -> TSimState:
        """
        Initialise the initial state of the simulation

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        TSimState
            The initial state of the environment.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        i: int,
        k: chex.PRNGKey,
        params: TSimParams,
        state: TSimState,
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
        i: int
            Current step number
        k: jax.random.PRNGKey
            JAX random key
        params: TSimParams
            Simulation time-independent parameters
        state: TSimState
            Simulation state

        Returns
        -------
        tuple[TSimState, chex.ArrayTree]
            Tuple containing the updated simulation state, and
            any data to be recorded over the course of the simulation.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("show_progress",))
    def run(
        self,
        n_steps: int,
        k: chex,
        params: TSimParams,
        initial_state: TSimState,
        show_progress: bool = True,
    ) -> Tuple[TSimState, chex.ArrayTree, chex.PRNGKey]:
        """
        Convenience function to run the simulation for a fixed number of steps

        Parameters
        ----------
        n_steps: int
            Number of simulation steps
        k: jax.random.PRNGKey
            JAX random key
        params: TSimParams
            Simulation time-independent parameters
        initial_state: TSimState
            Initial state of the simulation
        show_progress: bool, optional
            If ``True`` a progress bar will be displayed.
            Default ``True``

        Returns
        -------
        tuple[TSimState, chex.ArrayTree, jax,random.PRNGKey]
            Tuple containing

            - The final state of the simulation
            - Tree of recorded data
            - Updated JAX random key
        """
        final_state, records, k = sim_runner(
            self.step, params, initial_state, n_steps, k, show_progress=show_progress
        )

        return final_state, records, k

    @partial(jax.jit, static_argnums=(0, 1), static_argnames=("show_progress",))
    def init_and_run(
        self,
        n_steps: int,
        k: chex,
        show_progress: bool = True,
        params: Optional[TSimParams] = None,
    ) -> Tuple[TSimState, chex.ArrayTree]:
        """
        Convenience function to initialise and run the simulation

        Parameters
        ----------
        n_steps: int
            Number of simulation steps to run
        k: jax.random.PRNGKey
            JAX random key
        show_progress: bool, optional
            If ``True`` a progress bar will be displayed.
            Default ``True``
        params: TSimParams, optional
            Optional simulation parameters, if not provided
            default sim parameters will be used.

        Returns
        -------
        tuple[TSimState, chex.ArrayTree, jax,random.PRNGKey]
            Tuple containing

            - The final state of the simulation
            - Tree of recorded data
            - Updated JAX random key
        """
        k1, k2 = jax.random.split(k, 2)

        params = self.default_params() if params is None else params
        initial_state = self.initial_state(k1)
        final_state, records, k = sim_runner(
            self.step, params, initial_state, n_steps, k2, show_progress=show_progress
        )

        return final_state, records
