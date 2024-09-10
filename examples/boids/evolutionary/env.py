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
        min_speed: float,
        max_speed: float,
    ):
        self.apply_fun = apply_fun
        self.n_agents = n_agents
        self.shared_policy = shared_policy
        self.min_speed = min_speed
        self.max_speed = max_speed

    def default_params(self) -> updates.Params:
        return updates.Params()

    def initial_state(self, k: chex.PRNGKey) -> updates.Boid:
        k1, k2, k3 = jax.random.split(k, 3)

        boids = updates.Boid(
            pos=jax.random.uniform(k1, (self.n_agents, 2)),
            speed=jax.random.uniform(
                k2,
                (self.n_agents,),
                minval=self.min_speed,
                maxval=self.max_speed,
            ),
            heading=jax.random.uniform(
                k3, (self.n_agents,), minval=0.0, maxval=2.0 * jnp.pi
            ),
        )

        return boids

    def step(
        self,
        _i: int,
        k: chex.PRNGKey,
        params: updates.Params,
        boids: updates.Boid,
    ):
        net_params, params = params
        n_nb, x_nb, s_nb, h_nb = updates.observe(k, params, boids, boids, pos=boids.pos)
        obs = updates.flatten_observations(k, params, (boids, n_nb, x_nb, s_nb, h_nb))

        if self.shared_policy:
            actions = esquilax.ml.broadcast_params(self.apply_fun, net_params, obs)
        else:
            actions = esquilax.ml.map_params(self.apply_fun, net_params, obs)

        headings, speeds = updates.update_velocity(k, params, (actions, boids))
        pos = updates.move(k, params, (boids.pos, headings, speeds))
        rewards = updates.rewards(k, params, pos, pos, pos=pos)
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
        env_params.min_speed,
        env_params.max_speed,
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
        params_shaped, env, n_steps, shared_policy, k, env_params=env_params
    )

    return evo_state, agent_rewards, test_data.records, test_data.rewards
