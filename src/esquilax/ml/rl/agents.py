"""
RL agent definitions
"""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from typing_extensions import Self


class Agent(TrainState):
    """
    Abstract rl-agent class

    Outlines functionality required for agents to used
    as part of esquilax rl training functionality.

    This class extends
    :py:class:`flax.training.train_state.TrainState`,
    containing a network function and parameters, along with an
    optimiser and corresponding state.

    An ``Agent`` can represent multiple agents, dependent
    on it's expected observation and returned action shapes.
    """

    def sample_actions(
        self, key: chex.PRNGKey, observations: chex.Array
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """
        Sample actions given observations

        Parameters
        ----------
        key
            JAX random key.
        observations
            Environment observations

        Returns
        -------
        tuple
            Tuple containing actions and
            any corresponding action values
        """
        raise NotImplementedError

    def update(self, key: chex.PRNGKey, trajectories) -> Tuple[Self, chex.ArrayTree]:
        """
        Update the state of the agent from observed trajectories

        Parameters
        ----------
        key
            JAX random key.
        trajectories
            Structure of observation-action environment update trajectories.

        Returns
        -------
        tuple[esquilax.ml.rl.Agent, chex.ArrayTree]
            Tuple containing an updated ``Agent`` and any associated data
            e.g. training losses from the update.
        """
        raise NotImplementedError

    def apply_grads(self, *, grads: chex.ArrayTree, **kwargs) -> Self:
        """
        Apply gradients to the agent parameters and optimiser

        Parameters
        ----------
        grads
            Gradients corresponding to the agent parameters.
        **kwargs
            Any keyword arguments to pass to underlying ``apply_gradients`` function.

        Returns
        -------
        esquilax.ml.rl.Agent
            Updated agent
        """
        raise NotImplementedError


class SharedPolicyAgent(Agent):
    """
    Agent with a single trained policy
    """

    @classmethod
    def init(
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

    def apply_grads(self, *, grads, **kwargs) -> Self:
        """
        Apply gradients to the agent parameters and optimiser

        Parameters
        ----------
        grads
            Gradients corresponding to the agent parameters.
        **kwargs
            Any keyword arguments to pass to underlying ``apply_gradients`` function.

        Returns
        -------
        esquilax.ml.rl.SharedPolicyAgent
            Updated agent
        """
        return self.apply_gradients(grads=grads, **kwargs)


class BatchPolicyAgent(Agent):
    """
    Agent with parameters specific to each agent

    Agent intended for use where a group of agents is each
    trained on its own individual policy (but they share the
    same network structure, and optimiser type).
    """

    @classmethod
    def init(
        cls,
        key: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
        n_agents: int,
    ) -> Self:
        """
        Initialise the agent from a network

        Initialises individual parameters and optimisers
        for required agents.

        Parameters
        ----------
        key
            JAX random key.
        model
            Flax neural network model definition.
        tx
            Optax optimiser.
        observation_shape
            Shape of observations.
        n_agents
            Number of agents to initialise state for.

        Returns
        -------
        esquilax.ml.rl.BatchPolicyAgent
            Initialised agent.
        """
        fake_args = jnp.zeros(observation_shape)

        def init(_k):
            params = model.init(_k, fake_args)
            return cls.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(key, n_agents)
        return jax.vmap(init)(keys)

    def apply_grads(self, *, grads, **kwargs) -> Self:
        """
        Apply gradients to the agent parameters and optimiser

        Apply gradients across the set of agents, with each
        agent updated individually.

        Parameters
        ----------
        grads
            Gradients corresponding to the agent parameters. Each agent
            should receive its own set of gradients, i.e. gradients
            should have a shape ``[n-agents, n-parameters, ...]``.
        **kwargs
            Any keyword arguments to pass to underlying ``apply_gradients`` function.

        Returns
        -------
        esquilax.ml.rl.BatchPolicyAgent
            Updated agent
        """
        return jax.vmap(lambda a, g: a.apply_gradients(grads=g, **kwargs))(self, grads)
