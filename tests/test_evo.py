import evosax
import flax.linen as nn
import jax
import jax.numpy as jnp

from esquilax import Sim
from esquilax.ml import evo


def test_multi_strategy_training():
    n_agents = 5
    k = jax.random.PRNGKey(451)

    strategy = evosax.strategies.SimpleGA

    class Model(nn.module.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=2)(x)
            return jnp.sum(x)

    network = Model()
    net_params = network.init(k, jnp.zeros(4))

    strategies = (
        evo.BasicStrategy(net_params, strategy, n_agents),
        evo.BasicStrategy(net_params, strategy, n_agents),
    )
    evo_params = (
        strategies[0].default_params(),
        strategies[1].default_params(),
    )
    evo_state = (
        strategies[0].initialize(k, evo_params[0]),
        strategies[1].initialize(k, evo_params[1]),
    )

    class Env(Sim):
        def default_params(self):
            return 10

        def initial_state(self, k, params):
            return 10

        def step(
            self,
            i,
            k,
            params,
            state,
            *,
            agent_params,
        ):
            return (
                10,
                (
                    evo.TrainingData(
                        rewards=jnp.zeros((n_agents,)),
                        records=(
                            jnp.zeros(
                                n_agents,
                            )
                        ),
                    ),
                    evo.TrainingData(
                        rewards=jnp.zeros((n_agents,)),
                        records=(
                            jnp.zeros(
                                n_agents,
                            )
                        ),
                    ),
                ),
            )

    new_state, rewards = evo.train(
        strategies,
        Env(),
        2,
        3,
        4,
        False,
        k,
        evo_params,
        evo_state,
        show_progress=False,
    )
