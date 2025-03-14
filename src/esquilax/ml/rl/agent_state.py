from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from typing_extensions import Self


class AgentState(TrainState):
    """
    Basic RL agent parameter and optimiser state

    Extends :py:class:`flax.training.train_state.TrainState`,
    tracks the current network parameters and optimiser state.
    """

    @classmethod
    def init_from_model(
        cls,
        key: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
    ) -> Self:
        """
        Initialise state from a model/policy

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
        esquilax.ml.rl.AgentState
            Initialised agent state.
        """
        fake_args = jnp.zeros(observation_shape)
        params = model.init(key, fake_args)
        return cls.create(apply_fn=model.apply, params=params, tx=tx)

    def apply(self, observations: chex.Array) -> chex.Array:
        """
        Apply the network to a batch of observations

        Parameters
        ----------
        observations
            Array of observations, where the first
            axis of the array should correspond to the
            number of agents.

        Returns
        -------
        chex.Array
            Array of network output for each agent.
        """
        return jax.vmap(self.apply_fn, in_axes=(None, 0))(self.params, observations)


class BatchAgentState(TrainState):
    """
    Batch agent state

    Agent state for use where each agent has individual parameters
    and optimiser state (but the same network structure and optimiser).
    """

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
        Initialise state from a network

        Will generate individual (random) initial
        parameters for each agent.

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
        esquilax.ml.rl.BatchAgentState
            Initialised agent-states.
        """
        fake_args = jnp.zeros(observation_shape)

        def init(_k):
            params = model.init(_k, fake_args)
            return cls.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(key, n_agents)
        return jax.vmap(init)(keys)

    def apply_gradients(self, *, grads, **kwargs) -> Self:
        """
        Apply gradient updates to agent parameters

        Map apply a batch of gradient (i.e. individual updates per agent)
        across the individual agent parameters.

        Parameters
        ----------
        grads
            Batch of gradients, the first axis of the updates
            should correspond to the individual agents.
        **kwargs
            Any additional keyword arguments to forward to
            the underlying ``apply_gradients`` method.

        Returns
        -------
        esquilax.ml.rl.BatchAgentState
            Updated state
        """
        return jax.vmap(lambda a, g: TrainState.apply_gradients(a, grads=g, **kwargs))(
            self, grads
        )

    def apply(self, observations: chex.Array):
        """
        Apply the network, using individual agent parameters

        Applies the network mapped across agent parameters
        and corresponding observations

        Parameters
        ----------
        observations
            Array of observations, where the first
            axis of the array should correspond to the
            number of agents.

        Returns
        -------
        chex.Array
            Array of network output for each agent.
        """
        return jax.vmap(self.apply_fn, in_axes=(0, 0))(self.params, observations)
