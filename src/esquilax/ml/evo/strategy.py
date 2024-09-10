"""
Wrappers for `Evosax`_ strategy functionality

.. _Evosax: https://github.com/RobertTLange/evosax?tab=readme-ov-file
"""
from functools import partial
from typing import Any, Dict, Tuple, Union

import chex
import evosax
import jax
from flax.typing import FrozenVariableDict


class Strategy:
    """
    Abstract class wrapping Evosax strategy functionality

    This class can be used to encapsulate multiple populations,
    for example to have multiple agent types parameterised
    by independent populations.

    .. note::

       In the case of multiple policies, the tree structure
       should be consistent across population states,
       population samples etc.
    """

    def initialize(
        self, k: chex.PRNGKey, evo_params: evosax.EvoParams
    ) -> evosax.EvoState:
        """
        Initialise strategy state/states

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key
        evo_params: evosax.EvoParams
            Strategy parameters

        Returns
        -------
        evosax.EvoState
            Strategy state or collection of states
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def default_params(self) -> evosax.EvoParams:
        """
        Get default strategy parameters

        Returns
        -------
        evosax.EvoParams
            Strategy parameters
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reshape_params(self, population) -> chex.ArrayTree:
        """
        Reshape parameters

        Parameters
        ----------
        population
            Array (or collection of arrays) of parameters
            sampled from the strategy/strategies.

        Returns
        -------
        Reshaped parameters or collection of parameters
        """
        return population

    @partial(jax.jit, static_argnums=(0,))
    def shape_rewards(self, population, fitness: chex.ArrayTree) -> chex.ArrayTree:
        """
        Reshape environment rewards

        Rescale/reshape rewards generated during a simulation.

        Parameters
        ----------
        population
            Strategy population or collection of populations
        fitness: jax.numpy.array
            Fitness/rewards (or collection of)

        Returns
        -------
        chex.ArrayTree
            Rescaled fitness
        """
        return fitness

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self, k: chex.PRNGKey, evo_state: evosax.EvoState, evo_params: evosax.EvoParams
    ) -> Tuple[chex.ArrayTree, evosax.EvoState]:
        """
        Sample parameters from the current strategy state

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key
        evo_state: evosax.EvoState
            Strategy state (or collection of states)
        evo_params: evosax.EvoParams
            Strategy parameters (or collection of parameters)

        Returns
        -------
        tuple
            Population array (or collection of arrays) and
            updated state of the strategy/strategies
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        population: chex.ArrayTree,
        fitness: chex.ArrayTree,
        evo_state: evosax.EvoState,
        evo_params: evosax.EvoParams,
    ) -> evosax.EvoState:
        """
        Update strategy state/states

        Parameters
        ----------
        population: chex.ArrayTree
            Population array/arrays
        fitness: chex.ArrayTree
            Fitness array/arrays
        evo_state: evosax.EvoState
            Strategy state or collection of states
        evo_params: evosax.EvoParams
            Strategy params or collection of params

        Returns
        -------
        evosax.EvoState
            Update state/states
        """
        raise NotImplementedError


class BasicStrategy(Strategy):
    """
    Basic strategy representing a single policy

    Wrapper around a single strategy, with parameters
    initialised from a Flax neural-network.
    """

    def __init__(
        self,
        network_params: Union[FrozenVariableDict, Dict[str, Any]],
        strategy,
        pop_size: int,
        centered_rank_fitness: bool = True,
        z_score_fitness: bool = False,
        maximize_fitness: bool = True,
        **strategy_kwargs,
    ):
        """
        Parameters
        ----------
        network_params: Union[FrozenVariableDict, Dict[str, Any]]
            Flax network parameters.
        strategy
            Evosax strategy class.
        pop_size: int
            Strategy population size.
        centered_rank_fitness: bool, optional
            Use ``centered_rank_fitness`` for fitness-shaping, default ``True``.
        z_score_fitness: bool, optional
            Use ``z_score_fitness`` for fitness-shaping, default ``False``.
        maximize_fitness: bool, optional
            Use ``maximize_fitness`` for fitness-shaping, default ``True``.
        **strategy_kwargs
            Keyword arguments to pass to the strategy constructor.

        Attributes
        ----------
        param_reshaper: evosax.ParameterReshaper
            Parameter reshaper function
        strategy
            Initialised strategy
        fitness_shaper: evosax.FitnessShaper
            Fitness rescaler
        """
        self.param_reshaper = evosax.ParameterReshaper(network_params)
        self.strategy = strategy(
            popsize=pop_size,
            num_dims=self.param_reshaper.total_params,
            **strategy_kwargs,
        )
        self.fitness_shaper = evosax.FitnessShaper(
            centered_rank=centered_rank_fitness,
            z_score=z_score_fitness,
            maximize=maximize_fitness,
        )

    def initialize(
        self, k: chex.PRNGKey, evo_params: evosax.EvoParams
    ) -> evosax.EvoState:
        """
        Initialise strategy state

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key
        evo_params: evosax.EvoParams
            Strategy parameters

        Returns
        -------
        evosax.EvoState
            Strategy state
        """
        return self.strategy.initialize(k, evo_params)

    @partial(jax.jit, static_argnums=(0,))
    def default_params(self) -> evosax.EvoParams:
        """
        Get default strategy parameters

        Returns
        -------
        evosax.EvoParams
            Strategy parameters
        """
        return self.strategy.default_params

    def reshape_params(self, population) -> chex.ArrayTree:
        """
        Reshape parameters

        Parameters
        ----------
        population: jax.numpy.array
            Array of parameters sampled from the strategy.

        Returns
        -------
        jax.numpy.array
            Rescaled parameters
        """
        return self.param_reshaper.reshape(population)

    def ask(
        self, k: chex.PRNGKey, evo_state: evosax.EvoState, evo_params: evosax.EvoParams
    ) -> Tuple[chex.ArrayTree, evosax.EvoState]:
        """
        Sample parameters from the current strategy state

        Parameters
        ----------
        k: jax.random.PRNGKey
            JAX random key
        evo_state: evosax.EvoState
            Strategy state
        evo_params: evosax.EvoParams
            Strategy parameters

        Returns
        -------
        tuple[jax.numpy.array, evosax.EvoState]
            Population array and updated state of the strategy
        """
        return self.strategy.ask_strategy(k, evo_state, evo_params)

    def shape_rewards(self, population, fitness: chex.ArrayTree) -> chex.ArrayTree:
        """
        Reshape rewards

        Rescale/reshape rewards generated during a simulation.

        Parameters
        ----------
        population
            Strategy population or collection of populations
        fitness: jax.numpy.array
            Fitness/rewards (or collection of)

        Returns
        -------
        chex.ArrayTree
            Rescaled fitness
        """
        return self.fitness_shaper.apply(population, fitness)

    def tell(
        self,
        population: chex.ArrayTree,
        fitness: chex.ArrayTree,
        evo_state: evosax.EvoState,
        evo_params: evosax.EvoParams,
    ) -> evosax.EvoState:
        """
        Update strategy state

        Parameters
        ----------
        population: chex.ArrayTree
            Population array
        fitness: chex.ArrayTree
            Fitness array
        evo_state: evosax.EvoState
            Strategy state
        evo_params: evosax.EvoParams
            Strategy params

        Returns
        -------
        evosax.EvoState
            Update state
        """
        return self.strategy.tell(population, fitness, evo_state, evo_params)
