from functools import partial
from typing import Generic, Tuple, TypeVar

import chex
import jax

TEnvState = TypeVar("TEnvState")
TEnvParams = TypeVar("TEnvParams")


class Environment(Generic[TEnvState, TEnvParams]):
    """
    RL environment interface for Esquilax simulations

    Basic abstract RL environment intended for use with
    built-in training and testing functionality
    (:py:mod:`esquilax.ml.rl.training`).
    """

    def default_params(self) -> TEnvParams:
        """
        Get default environment parameters

        Returns
        -------
        TEnvParams
            Environment parameters.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: TEnvParams
    ) -> Tuple[chex.Array, TEnvState]:
        """
        Reset the state of the environment

        Reset the environment and return the new state and
        agent observations.

        Parameters
        ----------
        key: jax.random.PRNGKey
            JAX random key.
        params: TEnvParams
            Environment parameters.

        Returns
        -------
        tuple[jax.numpy.ndarray, TEnvState]
            Tuple containing

            - Agent observations, i.e. each agents
              observation of the sim state. If multiple
              agents, then the observation should have
              the same tree structure.
            - Environment state.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        params: TEnvParams,
        state: TEnvState,
        actions: chex.ArrayTree,
    ) -> Tuple[chex.ArrayTree, TEnvState, chex.ArrayTree, chex.ArrayTree]:
        """
        Update the state of the environment given agent actions

        Parameters
        ----------
        key: jax.random.PRNGKey
            JAX random key.
        params: TEnvParams
            Environment parameters
        state: TEnvState
            Current environment state
        actions
            Agent actions, for multiple agents this could be
            a PyTree of arrays.

        Returns
        -------
        tuple
            Tuple containing:

            - Agent observations
            - New environment state
            - Agent rewards
            - Terminal flags

            In the case of multiple agents, the tree structure
            of each field should match the tree structure of the
            actions.
        """
        raise NotImplementedError
