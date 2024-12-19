from typing import Callable, Optional

import chex
import evosax
import jax
import jax.numpy as jnp

import esquilax

from . import updates


class BoidEnv(esquilax.Sim[updates.Params, updates.Boid]):
    def __init__(
        self,
        apply_fun: Callable,
        n_agents: int,
        shared_policy: bool,
    ):
        self.apply_fun = apply_fun
        self.n_agents = n_agents
        self.shared_policy = shared_policy

    def default_params(self) -> updates.Params:
        return updates.Params()

    def initial_state(self, k: chex.PRNGKey, params: updates.Params) -> updates.Boid:
        k1, k2, k3 = jax.random.split(k, 3)

        boids = updates.Boid(
            pos=jax.random.uniform(k1, (self.n_agents, 2)),
            speed=jax.random.uniform(
                k2,
                (self.n_agents,),
                minval=params.min_speed,
                maxval=params.max_speed,
            ),
            heading=jax.random.uniform(
                k3, (self.n_agents,), minval=0.0, maxval=2.0 * jnp.pi
            ),
        )

        return boids

    def step(
        self,
        _i: int,
        _k: chex.PRNGKey,
        params: updates.Params,
        boids: updates.Boid,
        *,
        agent_params,
    ):
        n_nb, x_nb, s_nb, h_nb = updates.observe(params, boids, boids, pos=boids.pos)
        obs = updates.flatten_observations(params, (boids, n_nb, x_nb, s_nb, h_nb))
        actions = esquilax.ml.get_actions(
            self.apply_fun, self.shared_policy, agent_params, obs
        )
        headings, speeds = updates.update_velocity(params, (actions, boids))
        pos = updates.move(params, (boids.pos, headings, speeds))
        rewards = updates.rewards(params, pos, pos, pos=pos)
        boids = updates.Boid(pos=pos, heading=headings, speed=speeds)

        return boids, esquilax.ml.evo.TrainingData(
            rewards=rewards, records=(pos, headings)
        )


def evo_boids(
    env_params: updates.Params,
    n_agents: int,
    n_generations: int,
    n_samples: int,
    n_steps: int,
    shared_policy: bool,
    show_progress: bool = True,
    strategy=evosax.strategies.OpenES,
    layer_width: int = 16,
    pop_size: Optional[int] = None,
):
    k = jax.random.PRNGKey(101)

    network = updates.MLP(layer_width=layer_width, actions=2)
    net_params = network.init(k, jnp.zeros(4))

    if shared_policy:
        assert pop_size is not None, "Pop size required if shared_policy"
    else:
        pop_size = n_agents

    strategy = esquilax.ml.evo.BasicStrategy(net_params, strategy, pop_size)
    evo_params = strategy.default_params()
    evo_state = strategy.initialize(k, evo_params)

    env = BoidEnv(
        network.apply,
        n_agents,
        shared_policy,
    )

    evo_state, agent_rewards = esquilax.ml.evo.train(
        strategy,
        env,
        n_generations,
        n_samples,
        n_steps,
        shared_policy,
        k,
        evo_params,
        evo_state,
        show_progress=show_progress,
        env_params=env_params,
    )

    params, evo_state = strategy.ask(k, evo_state, evo_params)
    params_shaped = strategy.reshape_params(params)

    test_data = esquilax.ml.evo.test(
        params_shaped, env, n_samples, n_steps, shared_policy, k, env_params=env_params
    )

    return evo_state, agent_rewards, test_data.records, test_data.rewards
