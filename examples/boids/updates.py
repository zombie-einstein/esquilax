from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

import esquilax


@chex.dataclass
class Boid:
    pos: chex.Array
    heading: float
    speed: float


@chex.dataclass
class Params:
    max_speed: float = 0.05
    min_speed: float = 0.025
    max_rotate: float = 0.1
    max_accelerate: float = 0.005
    close_range: float = 0.01
    collision_penalty: float = 0.1


@esquilax.transforms.spatial(
    10,
    (jnp.add, jnp.add, jnp.add, jnp.add),
    (0, jnp.zeros(2), 0.0, 0.0),
    include_self=False,
)
def observe(_k: chex.PRNGKey, _params: Params, a: Boid, b: Boid):
    dh = esquilax.utils.shortest_vector(a.heading, b.heading, length=2 * jnp.pi)
    return 1, b.pos, b.speed, dh


@esquilax.transforms.amap
def flatten_observations(_k: chex.PRNGKey, params: Params, observations):
    boid, n_nb, x_nb, s_nb, h_nb = observations

    def obs_to_nbs():
        _x_nb = x_nb / n_nb
        _s_nb = s_nb / n_nb
        _h_nb = h_nb / n_nb

        dx = esquilax.utils.shortest_vector(boid.pos, _x_nb)

        d = jnp.sqrt(jnp.sum(dx * dx)) / 0.1

        phi = jnp.arctan2(dx[1], dx[0]) + jnp.pi
        d_phi = esquilax.utils.shortest_vector(boid.heading, phi, 2 * jnp.pi) / jnp.pi

        dh = _h_nb / jnp.pi
        ds = (_s_nb - boid.speed) / (params.max_speed - params.min_speed)

        return jnp.array([d, d_phi, dh, ds])

    return jax.lax.cond(
        n_nb > 0,
        obs_to_nbs,
        lambda: jnp.array([-1.0, 0.0, 0.0, -1.0]),
    )


@esquilax.transforms.amap
def update_velocity(_k: chex.PRNGKey, params: Params, x: Tuple[chex.Array, Boid]):
    actions, boid = x
    rotation = actions[0] * params.max_rotate * jnp.pi
    acceleration = actions[1] * params.max_accelerate

    new_heading = (boid.heading + rotation) % (2 * jnp.pi)
    new_speeds = jnp.clip(
        boid.speed + acceleration,
        min=params.min_speed,
        max=params.max_speed,
    )

    return new_heading, new_speeds


@esquilax.transforms.amap
def move(_key: chex.PRNGKey, _params: Params, x):
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


@esquilax.transforms.spatial(
    5,
    jnp.add,
    0.0,
    include_self=False,
)
def rewards(_k: chex.PRNGKey, params: Params, a: chex.Array, b: chex.Array):
    d = esquilax.utils.shortest_distance(a, b, norm=True)

    reward = jax.lax.cond(
        d < params.close_range,
        lambda _: -params.collision_penalty,
        lambda _d: jnp.exp(-50 * _d),
        d,
    )

    return reward


class MLP(nn.Module):
    layer_width: int
    actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.layer_width)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.layer_width)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.actions)(x)
        x = nn.tanh(x)

        return x
