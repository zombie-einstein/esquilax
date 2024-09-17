"""
Wrappers for `Evosax`_ strategy functionality

Class definitions wrapping strategy functionality
for use when training policies within an Esquilax
simulation. Classes implementing
:py:class:`esquilax.ml.evo.Strategy` can be passed
to :py:class:`esquilax.ml.evo.train` which will then
automatically handle optimisation.

Multiple strategies can be passed to
:py:class:`esquilax.ml.evo.train` allowing training
of multiple strategies within the same class.

.. _Evosax: https://github.com/RobertTLange/evosax
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
    """

    def initialize(
        self, key: chex.PRNGKey, evo_params: evosax.EvoParams
    ) -> evosax.EvoState:
        """
        Initialise strategy state

        Parameters
        ----------
        key
            JAX random key
        evo_params: evosax.EvoParams
            Strategy parameters

        Returns
        -------
        evosax.EvoState
            Strategy state
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
            Array of parameters sampled from the strategy.

        Returns
        -------
        Reshaped parameters
        """
        return population

    @partial(jax.jit, static_argnums=(0,))
    def shape_rewards(self, population, fitness: chex.ArrayTree) -> chex.ArrayTree:
        """
        Reshape environment rewards

        Rescale/reshape rewards generated during a simulation.

        .. warning::

           Evosax expects to fitness minimised, so either the
           training environment should return values to
           be minimised, or rewards returned from the
           environment should be rescaled by this method.

        Parameters
        ----------
        population
            Strategy population
        fitness
            Fitness/rewards

        Returns
        -------
        chex.ArrayTree
            Rescaled fitness
        """
        return fitness

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        key: chex.PRNGKey,
        evo_state: evosax.EvoState,
        evo_params: evosax.EvoParams,
    ) -> Tuple[chex.ArrayTree, evosax.EvoState]:
        """
        Sample parameters from the current strategy state

        Parameters
        ----------
        key
            JAX random key
        evo_state
            Strategy state
        evo_params
            Strategy parameters

        Returns
        -------
        tuple
            Population array and updated state of the strategy
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
        Update strategy state

        Parameters
        ----------
        population
            Population array
        fitness
            Fitness array
        evo_state
            Strategy state
        evo_params
            Strategy params

        Returns
        -------
        evosax.EvoState
            Update state
        """
        raise NotImplementedError


class BasicStrategy(Strategy):
    """
    Basic strategy derived from a Flax neural network

    Wrapper around a strategy, with parameters
    initialised from a Flax neural-network.
    """

    param_reshaper: evosax.ParameterReshaper
    strategy: evosax.Strategy
    fitness_shaper: evosax.FitnessShaper

    def __init__(
        self,
        network_params: Union[FrozenVariableDict, Dict[str, Any]],
        strategy: evosax.Strategy,
        pop_size: int,
        centered_rank_fitness: bool = True,
        z_score_fitness: bool = False,
        maximize_fitness: bool = True,
        **strategy_kwargs,
    ):
        """
        Parameters
        ----------
        network_params
            Flax network parameters.
        strategy
            Evosax strategy class.
        pop_size
            Strategy population size.
        centered_rank_fitness
            Use ``centered_rank_fitness`` for fitness-shaping, default ``True``.
        z_score_fitness
            Use ``z_score_fitness`` for fitness-shaping, default ``False``.
        maximize_fitness
            Use ``maximize_fitness`` for fitness-shaping, default ``True``.

            .. warning::

               Evosax expects that fitness should be minimised, so this should
               be ``True`` if the environment returns rewards to be maximised.

        **strategy_kwargs
            Keyword arguments to pass to the strategy constructor.
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
        self, key: chex.PRNGKey, evo_params: evosax.EvoParams
    ) -> evosax.EvoState:
        """
        Initialise strategy state

        Parameters
        ----------
        key
            JAX random key
        evo_params
            Strategy parameters

        Returns
        -------
        evosax.EvoState
            Strategy state
        """
        return self.strategy.initialize(key, evo_params)

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
        population
            Array of parameters sampled from the strategy.

        Returns
        -------
        jax.numpy.array
            Rescaled parameters
        """
        return self.param_reshaper.reshape(population)

    def ask(
        self,
        key: chex.PRNGKey,
        evo_state: evosax.EvoState,
        evo_params: evosax.EvoParams,
    ) -> Tuple[chex.ArrayTree, evosax.EvoState]:
        """
        Sample parameters from the current strategy state

        Parameters
        ----------
        key
            JAX random key
        evo_state
            Strategy state
        evo_params
            Strategy parameters

        Returns
        -------
        tuple[jax.numpy.array, evosax.EvoState]
            Population array and updated state of the strategy
        """
        return self.strategy.ask_strategy(key, evo_state, evo_params)

    def shape_rewards(self, population, fitness: chex.ArrayTree) -> chex.ArrayTree:
        """
        Reshape rewards

        Rescale/reshape rewards generated during a simulation.

        Parameters
        ----------
        population
            Strategy population or collection of populations
        fitness
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
        population
            Population array
        fitness
            Fitness array
        evo_state
            Strategy state
        evo_params
            Strategy params

        Returns
        -------
        evosax.EvoState
            Update state
        """
        return self.strategy.tell(population, fitness, evo_state, evo_params)
