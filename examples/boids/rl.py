from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from esquilax import ml

from . import updates


class BoidEnv(ml.rl.Environment):
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    @property
    def default_params(self) -> updates.Params:
        return updates.Params()

    def reset(
        self, key: chex.PRNGKey, params: updates.Params
    ) -> Tuple[chex.Array, updates.Boid]:
        k1, k2, k3 = jax.random.split(key, 3)

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
        obs = self.get_obs(boids, params=params, key=key)
        return obs, boids

    def step(
        self,
        key: chex.PRNGKey,
        params: updates.Params,
        state: updates.Boid,
        actions: chex.Array,
    ) -> Tuple[chex.Array, updates.Boid, chex.Array, chex.Array, Dict[Any, Any]]:
        headings, speeds = updates.update_velocity(key, params, (actions, state))
        pos = updates.move(key, params, (state.pos, headings, speeds))
        rewards = updates.rewards(key, params, pos, pos, pos=pos)
        boids = updates.Boid(pos=pos, heading=headings, speed=speeds)
        obs = self.get_obs(boids, params=params, key=key)
        return obs, state, rewards, False

    def get_obs(
        self,
        state,
        params=None,
        key=None,
    ) -> chex.Array:
        n_nb, x_nb, s_nb, h_nb = updates.observe(
            key, params, state, state, pos=state.pos
        )
        obs = updates.flatten_observations(key, params, (state, n_nb, x_nb, s_nb, h_nb))
        return obs


class RLAgent(ml.rl.SharedPolicyAgent):
    def sample_actions(self, _k, observations):
        actions = ml.get_actions(self.apply_fn, True, self.params, observations)
        return actions, None

    def update(self, _k, trajectories):
        return self, -1


def rl_boids(
    env_params: updates.Params,
    n_agents: int,
    n_epochs: int,
    n_env: int,
    n_steps: int,
    layer_width: int = 16,
    show_progress: bool = True,
):
    k = jax.random.PRNGKey(451)
    k_init, k_train = jax.random.split(k)

    env = BoidEnv(n_agents)

    network = updates.MLP(layer_width=layer_width, actions=2)
    opt = optax.adam(1e-4)
    agents = RLAgent.init(k_init, network, opt, (4,))

    trained_agents, rewards, _ = ml.rl.train(
        k_train,
        agents,
        env,
        env_params,
        n_epochs,
        n_env,
        n_steps,
        show_progress=show_progress,
    )

    return trained_agents, rewards
