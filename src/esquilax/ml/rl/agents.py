"""
RL agent definitions
"""
from functools import partial
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from typing_extensions import Self


class AgentState(TrainState):
    @classmethod
    def init_from_model(
        cls,
        key: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
    ) -> Self:
        """
        Initialise the agent from a network

        Parameters
        ----------
        key
            JAX random key.
        model
            Flax neural network model definition.
        tx
            Optax optimiser.
        observation_shape
            Shape of observations

        Returns
        -------
        esquilax.ml.rl.SharedPolicyAgent
            Initialised agent.
        """
        fake_args = jnp.zeros(observation_shape)
        params = model.init(key, fake_args)
        return cls.create(apply_fn=model.apply, params=params, tx=tx)


class BatchAgentState(TrainState):
    @classmethod
    def init_from_model(
        cls,
        key: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
        n_agents: int,
    ) -> Self:
        """
        Initialise the agent from a network

        Parameters
        ----------
        key
            JAX random key.
        model
            Flax neural network model definition.
        tx
            Optax optimiser.
        observation_shape
            Shape of observations
        n_agents
            Number of agents to initialise state for.

        Returns
        -------
        esquilax.ml.rl.SharedPolicyAgent
            Initialised agent.
        """
        fake_args = jnp.zeros(observation_shape)

        def init(_k):
            params = model.init(_k, fake_args)
            return cls.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(key, n_agents)
        return jax.vmap(init)(keys)

    def apply_gradients(self, *, grads, **kwargs) -> Self:
        return jax.vmap(lambda a, g: TrainState.apply_gradients(a, grads=g, **kwargs))(
            self, grads
        )


class Agent:
    """
    Abstract rl-agent class

    Outlines functionality required for agents to used
    as part of esquilax rl training functionality.

    An ``Agent`` can represent multiple agents, dependent
    on it's expected observation and returned action shapes.
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
        tuple
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
            Structure of observation-action environment update trajectories.

        Returns
        -------
        tuple[AgentState | BatchAgentState, chex.ArrayTree]
            Tuple containing an updated ``Agent`` and any associated data
            e.g. training losses from the update.
        """
        raise NotImplementedError
