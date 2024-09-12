"""
`Evosax`_ policy training and testing functionality

.. _Evosax: https://github.com/RobertTLange/evosax?tab=readme-ov-file
"""
from functools import partial
from typing import Optional, Tuple

import chex
import evosax
import jax
import jax.numpy as jnp
import jax_tqdm

from esquilax.batch_runner import batch_sim_runner
from esquilax.env import Sim, TSimParams

from .strategy import Strategy


@chex.dataclass
class TrainingData:
    """
    Dataclass tracking training rewards and recorded data

    Simulations used for training should return
    a ``TrainingData`` class as its recorded data.

    Examples
    --------

    .. code-block:: python

       def step(self, i, k, params, state):
           ...
           # Rewards are returned as part of the TrainingData
           return (
               state,
               TrainingData(rewards=rewards, records=records)
           )

    Parameters
    ----------
    rewards: chex.ArrayTree
        PyTree of rewards generated by simulation agents.
    records: chex.ArrayTree
        Any additional data recorded over the course
        of the simulation.
    """

    rewards: chex.Array
    records: chex.ArrayTree


@partial(
    jax.jit,
    static_argnames=(
        "strategy",
        "env",
        "n_generations",
        "n_samples",
        "n_steps",
        "map_population",
        "show_progress",
    ),
)
def train(
    strategy: Strategy,
    env: Sim,
    n_generations: int,
    n_samples: int,
    n_steps: int,
    map_population: bool,
    k: chex.PRNGKey,
    evo_params: evosax.EvoParams,
    evo_state: evosax.EvoState,
    show_progress: bool = True,
    env_params: Optional[TSimParams] = None,
) -> Tuple[evosax.EvoState, chex.ArrayTree]:
    """
    Train a policy/policies using neuro-evolution

    Training loop that optimises a policy evaluated
    against an Esquilax simulation training
    environment.

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
    strategy: esquilax.ml.evo.Strategy
        Evosax evolutionary strategy
    env: esquilax.env.SimEnv
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
    n_generations: int
        Number of training generations
    n_samples: int
        Number of Monte-Carlo samples to test each population
        or parameter samples across.
    n_steps: int
        Number of steps to run each simulation
    map_population: bool
        If ``True`` each member of the population is
        evaluated in a separate environment (i.e. the
        individual parameters are shared by agents). If
        ``False`` the whole population is passed
        to the environment as part of the simulation
        state.
    k: jax.random.PRNGKey
        JAX random key
    evo_params: evosax.EvoParams
        Evosax strategy parameters
    evo_state: evosax.EvoState
        Evosax strategy state
    show_progress: bool, optional
        If ``True`` a progress bar will be displayed
        showing training progress. Default value is ``True``.
    env_params: TSimParams, optional
        Optional environment parameters, if not provided the
        default environment parameters will be used.

    Returns
    -------
    tuple[evosax.EvoState, chex.ArrayTree]
        Trained strategy state and record of rewards
        gathered over the course of training.
    """

    env_params = env.default_params() if env_params is None else env_params

    def generation(carry, _):
        _k, _evo_state = carry

        _k, k1, k2 = jax.random.split(_k, 3)

        _population, _evo_state = strategy.ask(k1, _evo_state, evo_params)
        _population_shaped = strategy.reshape_params(_population)

        def inner(_pop) -> TrainingData:
            return batch_sim_runner(
                env,
                n_samples,
                n_steps,
                k2,
                show_progress=False,
                params=env_params,
                agent_params=_pop,
            )

        if map_population:
            _training_data = jax.vmap(inner)(_population_shaped)
            _total_rewards = jnp.sum(_training_data.rewards, axis=(0, 2, 3))
        else:
            _training_data = inner(_population_shaped)
            _total_rewards = jnp.sum(_training_data.rewards, axis=(0, 1))

        _fitness = strategy.shape_rewards(_population, _total_rewards)
        _evo_state = strategy.tell(_population, _fitness, _evo_state, evo_params)

        return (_k, _evo_state), _total_rewards

    if show_progress:
        generation = jax_tqdm.scan_tqdm(n_generations, desc="Generation")(generation)

    (k, evo_state), rewards = jax.lax.scan(
        generation,
        (k, evo_state),
        jnp.arange(n_generations),
    )

    return evo_state, rewards


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_steps",
        "map_population",
        "show_progress",
    ),
)
def test(
    population_shaped: chex.ArrayTree,
    env: Sim,
    n_steps: int,
    map_population: bool,
    k: chex.PRNGKey,
    show_progress: bool = True,
    env_params: Optional[TSimParams] = None,
) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    """
    Test population performance and gather telemetry

    Test a population against a simulation environment
    gathering rewards and recorded simulation state
    over the course of the simulation

    Parameters
    ----------
    population_shaped: chex.ArrayTree
        Reshaped population parameters for use
        by simulation agents.
    env: esquilax.env.SimEnv
        Esquilax simulation training environment
    n_steps: int
        Number of simulation steps.
    map_population: bool
        If ``True`` each member of the population is
        evaluated in a separate environment (i.e. the
        individual parameters are shared by agents). If
        ``False`` the whole population is passed
        to the environment as part of the simulation
        state.
    k: jax.random.PRNGKey
        JAX random key
    show_progress: bool, optional
        If ``True`` a progress bar will be displayed
        showing simulation progress. Default value is ``True``.
    env_params: TSimParams, optional
        Optional environment parameters, if not provided the
        default environment parameters will be used.

    Returns
    -------
    tuple[chex.ArrayTree, chex.ArrayTree]
        Tuple containing records of rewards and recorded
        state for each step of the simulation.
    """
    k1, k2 = jax.random.split(k)

    env_params = env.default_params() if env_params is None else env_params

    def inner(_pop):
        _initial_state = env.initial_state(k1, env_params)
        _, _testing_data, _ = env.run(
            n_steps,
            k2,
            env_params,
            _initial_state,
            show_progress=show_progress,
            agent_params=_pop,
        )
        return _testing_data

    if map_population:
        records = jax.vmap(inner)(population_shaped)
    else:
        records = inner(population_shaped)

    return records