"""
`Evosax`_ policy training and testing functionality

Basic training and testing loops of neuro-evolution
strategies. Also supports training multiple strategies
within the same loop.

.. _Evosax: https://github.com/RobertTLange/evosax
"""
from functools import partial
from typing import Collection, Optional, Tuple

import chex
import evosax
import jax
import jax.numpy as jnp
import jax_tqdm

from esquilax.batch_runner import batch_sim_runner
from esquilax.sim import Sim
from esquilax.typing import TSimParams, TypedPyTree

from . import tree_utils
from .strategy import Strategy
from .types import TrainingData


@partial(
    jax.jit,
    static_argnames=(
        "strategies",
        "env",
        "n_generations",
        "n_env",
        "n_steps",
        "map_population",
        "show_progress",
    ),
)
def train(
    strategies: TypedPyTree[Strategy],
    env: Sim,
    n_generations: int,
    n_env: int,
    n_steps: int,
    map_population: bool,
    key: chex.PRNGKey,
    evo_params: TypedPyTree[evosax.EvoParams],
    evo_states: TypedPyTree[evosax.EvoState],
    show_progress: bool = True,
    env_params: Optional[TSimParams] = None,
) -> Tuple[Collection[evosax.EvoState], chex.ArrayTree]:
    """
    Train a policy/policies using neuro-evolution

    Training loop that optimises a policy or multiple
    policies evaluated against an Esquilax simulation
    training environment.

    Agent policies/behaviours are sampled and updated
    using Evosax, and can either be tested as
    a shard policy (i.e. all agents share a policy
    from the population), or each agent is controlled
    by an individual member of the population.

    The training loop broadly follows these steps:

    - Sample a new population from the strategy
    - Initialise a simulation(s) and run with
      the population/parameters
    - Collect rewards over the simulation
    - Aggregate and rescale rewards
    - Update the strategy state from the
      rewards/fitness

    .. note::

       Rewards are summed over the simulation steps, i.e.
       the total rewards for each agent (or policy)
       are used to measure fitness during training.

    Parameters
    ----------
    strategies
        PyTree of Strategy classes. This could be either a
        single Strategy, or a container/struct of strategies.
    env
        Esquilax simulation training environment

        .. warning::

           The step function of the environment should
           include the population/parameters as
           a keyword argument `agent_params`, and
           return a `TrainingData` class, for example:

           .. code-block:: python

              def step(self, i, k, params, state, *, agent_params):
                  ...
                  # Rewards are returned as part of the TrainingData
                  return (
                      state,
                      TrainingData(rewards=rewards, records=records)
                  )

           The current population and rewards will then be
           passed and handled automatically by the training
           loop.

           If multiple strategies are being used, then the returned
           training data should have the same structure, e.g.:

           .. code-block:: python

              # Multiple strategies as a dictionary
              strategies = {"a": ..., "b": ...}

              def step(self, i, k, params, state, *, agent_params):
                  ...
                  # Should return dict matching the strategy structure
                  return (
                      state,
                      dict(
                        a = TrainingData(...),
                        b = TrainingData(...),
                      )
                  )

    n_generations
        Number of training generations
    n_env
        Number of Monte-Carlo samples to test each population
        or parameter samples across.
    n_steps
        Number of steps to run each simulation
    map_population
        If ``True`` each member of the population is
        evaluated in a separate environment (i.e. the
        individual parameters are shared by agents). If
        ``False`` the whole population is passed
        to the environment as part of the simulation
        state.
    key
        JAX random key
    evo_params
        Evosax strategy parameters
    evo_states
        Evosax strategy state
    show_progress
        If ``True`` a progress bar will be displayed
        showing training progress. Default value is ``True``.
    env_params
        Optional environment parameters, if not provided the
        default environment parameters will be used.

    Returns
    -------
    tuple[evosax.EvoState, chex.ArrayTree]
        Trained strategy state and record of rewards
        gathered over the course of training. If multiple
        strategies were provided, then the output will
        have the same structure.
    """

    env_params = env.default_params() if env_params is None else env_params

    def generation(carry, _):
        _k, _evo_states = carry

        _k, k1, k2 = jax.random.split(_k, 3)

        _evo_states, _population, _population_shaped = tree_utils.tree_ask(
            k1, strategies, _evo_states, evo_params
        )

        def inner(_pop) -> TrainingData:
            return batch_sim_runner(
                env,
                n_env,
                n_steps,
                k2,
                show_progress=False,
                params=env_params,
                agent_params=_pop,
            )

        if map_population:
            _training_data = jax.vmap(inner)(_population_shaped)
            _total_rewards = jax.tree.map(
                lambda t: jnp.sum(t.rewards, axis=(0, 2, 3)),
                _training_data,
                is_leaf=lambda x: isinstance(x, TrainingData),
            )
        else:
            _training_data = inner(_population_shaped)
            _total_rewards = jax.tree.map(
                lambda t: jnp.sum(t.rewards, axis=(0, 1)),
                _training_data,
                is_leaf=lambda x: isinstance(x, TrainingData),
            )

        _fitness, _evo_states = tree_utils.tree_tell(
            strategies, _population, _total_rewards, _evo_states, evo_params
        )

        return (_k, _evo_states), _total_rewards

    if show_progress:
        generation = jax_tqdm.scan_tqdm(n_generations, desc="Generation")(generation)

    (_, evo_states), rewards = jax.lax.scan(
        generation,
        (key, evo_states),
        jnp.arange(n_generations),
    )

    return evo_states, rewards


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_env",
        "n_steps",
        "map_population",
        "show_progress",
    ),
)
def test(
    population_shaped: chex.ArrayTree,
    env: Sim,
    n_env: int,
    n_steps: int,
    map_population: bool,
    key: chex.PRNGKey,
    show_progress: bool = True,
    env_params: Optional[TSimParams] = None,
) -> TrainingData:
    """
    Test population performance and gather telemetry

    Test a population against a simulation environment
    gathering rewards and recorded simulation state
    over the course of the simulation

    Parameters
    ----------
    population_shaped
        Reshaped population parameters for use
        by simulation agents.
    env
        Esquilax simulation training environment
    n_env
        Number of Monte-Carlo samples to test each population
        or parameter samples across.
    n_steps
        Number of simulation steps.
    map_population
        If ``True`` each member of the population is
        evaluated in a separate environment (i.e. the
        individual parameters are shared by agents). If
        ``False`` the whole population is passed
        to the environment as part of the simulation
        state.
    key
        JAX random key
    show_progress
        If ``True`` a progress bar will be displayed
        showing simulation progress. Default value is ``True``.
    env_params
        Optional environment parameters, if not provided the
        default environment parameters will be used.

    Returns
    -------
    esquilax.ml.evo.TrainingData
        Data collected over the course of the simulation.
    """
    k1, k2 = jax.random.split(key)

    env_params = env.default_params() if env_params is None else env_params

    def inner(_pop):
        return batch_sim_runner(
            env,
            n_env,
            n_steps,
            k2,
            show_progress=show_progress,
            params=env_params,
            agent_params=_pop,
        )

    if map_population:
        records = jax.vmap(inner)(population_shaped)
    else:
        records = inner(population_shaped)

    return records
