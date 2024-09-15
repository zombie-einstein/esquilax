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

    This class extends the flax Trainstate, containing
    a network function and parameters, along with an
    optimiser and corresponding state.

    An ``Agent`` can represent multiple agents, dependent
    on it's expected observation and returned action shapes.
    """

    def sample_actions(
        self, k: chex.PRNGKey, observations: chex.Array
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """
        Sample actions given observations

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key.
        observations: chex.Array
            Environment observations

        Returns
        -------
        tuple
            Tuple containing actions and
            any corresponding action values
        """
        raise NotImplementedError

    def update(self, k: chex.PRNGKey, trajectories) -> Tuple[Self, chex.ArrayTree]:
        """
        Update the state of the agent from observed trajectories

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key.
        trajectories: Trajectory
            Structure of observation-action environment update trajectories.

        Returns
        -------
        tuple
            Tuple containing an updated ``Agent`` and any associated data
            e.g. training losses from the update.
        """
        raise NotImplementedError

    def apply_grads(self, *, grads: chex.ArrayTree, **kwargs) -> Self:
        """
        Apply gradients to the agent parameters and optimiser

        Parameters
        ----------
        grads: chex.ArrayTree
            Gradients corresponding to the agent parameters.
        kwargs
            Any keyword arguments to pass to underlying ``apply_gradients`` function.

        Returns
        -------
        Agent
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
        k: chex.PRNGKey,
        model: nn.Module,
        tx: optax.GradientTransformation,
        observation_shape: Tuple[int, ...],
    ) -> Self:
        """
        Initialise the agent from a network

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key.
        model: flax.linen.Module
            Flax neural network model definition.
        tx: optax.GradientTransformation
            Optax optimiser.
        observation_shape: tuple[int]
            Shape of observations

        Returns
        -------
        SharedPolicyAgent
            Initialised agent.
        """
        fake_args = jnp.zeros(observation_shape)
        params = model.init(k, fake_args)
        return cls.create(apply_fn=model.apply, params=params, tx=tx)

    def apply_grads(self, *, grads, **kwargs) -> Self:
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
        k: chex.PRNGKey,
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
        k: jax.random.PRNGKey
            JAX random key.
        model: flax.linen.Module
            Flax neural network model definition.
        tx: optax.GradientTransformation
            Optax optimiser.
        observation_shape: tuple[int]
            Shape of observations.
        n_agents: int
            Number of agents to initialise state for.

        Returns
        -------
        BatchPolicyAgent
            Initialised agent.
        """
        fake_args = jnp.zeros(observation_shape)

        def init(_k):
            params = model.init(_k, fake_args)
            return cls.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(k, n_agents)
        return jax.vmap(init)(keys)

    def apply_grads(self, *, grads, **kwargs) -> Self:
        """
        Apply gradients to the agent parameters and optimiser

        Apply gradients across the set of agents, with each
        agent updated individually.

        Parameters
        ----------
        grads: chex.ArrayTree
            Gradients corresponding to the agent parameters. Each agent
            should receive its own set of gradients, i.e. gradients
            should have a shape ``[n-agents, n-parameters, ...]``.
        kwargs
            Any keyword arguments to pass to underlying ``apply_gradients`` function.

        Returns
        -------
        Agent
            Updated agent
        """
        return jax.vmap(lambda a, g: a.apply_gradients(grads=g, **kwargs))(self, grads)
