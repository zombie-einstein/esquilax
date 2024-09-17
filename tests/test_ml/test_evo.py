import evosax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from esquilax import Sim
from esquilax.ml import evo

from .conftest import SimpleModel


def test_strategy_dict_training():
    n_agents = 5
    k = jax.random.PRNGKey(451)

    strategy = evosax.strategies.SimpleGA

    network = SimpleModel()
    net_params = network.init(k, jnp.zeros(4))

    strategies = FrozenDict(
        a=evo.BasicStrategy(net_params, strategy, n_agents),
        b=evo.BasicStrategy(net_params, strategy, n_agents),
    )
    evo_params = FrozenDict(
        a=strategies["a"].default_params(),
        b=strategies["b"].default_params(),
    )
    evo_state = FrozenDict(
        a=strategies["a"].initialize(k, evo_params["a"]),
        b=strategies["b"].initialize(k, evo_params["b"]),
    )

    class Env(Sim):
        def default_params(self):
            return 10

        def initial_state(self, _k, params):
            return 10

        def step(self, i, _k, params, state, *, agent_params):
            # Just check we should receive the params as a dict
            assert isinstance(agent_params, FrozenDict)
            assert sorted(agent_params.keys()) == ["a", "b"]
            training_data = FrozenDict(
                a=evo.TrainingData(rewards=jnp.zeros((n_agents,)), records=10),
                b=evo.TrainingData(rewards=jnp.zeros((n_agents,)), records=10),
            )

            return 10, training_data

    n_generations = 2
    n_samples = 3
    n_steps = 4

    env = Env()

    new_state, rewards = evo.train(
        strategies,
        env,
        n_generations,
        n_samples,
        n_steps,
        False,
        k,
        evo_params,
        evo_state,
        show_progress=False,
    )

    assert isinstance(new_state, FrozenDict)
    assert sorted(new_state.keys()) == ["a", "b"]
    assert isinstance(rewards, FrozenDict)
    assert rewards["a"].shape == (n_generations, n_agents)
    assert rewards["b"].shape == (n_generations, n_agents)

    _, _, pop_shaped = evo.tree_utils.tree_ask(k, strategies, new_state, evo_params)

    test_data = evo.test(
        pop_shaped, env, n_samples, n_steps, False, k, show_progress=False
    )

    assert isinstance(test_data, FrozenDict)
    for v in test_data.values():
        assert isinstance(v, evo.TrainingData)
        assert v.rewards.shape == (n_samples, n_steps, n_agents)
        assert v.records.shape == (n_samples, n_steps)
