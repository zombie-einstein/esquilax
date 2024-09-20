"""
RL agent interfaces and state
"""
from functools import partial
from typing import Tuple, Union

import chex
import jax

from .agent_state import AgentState, BatchAgentState


class Agent:
    """
    Abstract rl-agent class

    Outlines functionality required for agents to used
    as part of esquilax rl training functionality
    (see :py:mod:`esquilax.ml.rl.training`).
    """

    @partial(jax.jit, static_argnames=("self",))
    def sample_actions(
        self,
        key: chex.PRNGKey,
        agent_state: Union[AgentState, BatchAgentState],
        observations: chex.Array,
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """
        Sample actions given observations

        Parameters
        ----------
        key
            JAX random key.
        agent_state
            Current agent training state.
        observations
            Environment observations.

        Returns
        -------
        tuple[chex.ArrayTree, chex.ArrayTree]
            Tuple containing actions and
            any corresponding action values
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        key: chex.PRNGKey,
        agent_state: Union[AgentState, BatchAgentState],
        trajectories,
    ) -> Tuple[Union[AgentState, BatchAgentState], chex.ArrayTree]:
        """
        Update the state of the agent from observed trajectories

        Parameters
        ----------
        key
            JAX random key.
        agent_state
            Current agent training state.
        trajectories
            Struct of environment update trajectories.

        Returns
        -------
        tuple[AgentState | BatchAgentState, chex.ArrayTree]
            Tuple containing an updated ``Agent`` and any associated data
            e.g. training losses from the update.
        """
        raise NotImplementedError
