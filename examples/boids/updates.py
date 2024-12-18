from functools import partial
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


@partial(
    esquilax.transforms.spatial,
    i_range=0.1,
    reduction=(jnp.add, jnp.add, jnp.add, jnp.add),
    default=(0, jnp.zeros(2), 0.0, 0.0),
    include_self=False,
)
def observe(_k: chex.PRNGKey, _params: Params, a: Boid, b: Boid):
    """
    Count neighbours and accumulate their relative velocities and positions
    """
    dh = esquilax.utils.shortest_vector(a.heading, b.heading, length=2 * jnp.pi)
    dx = esquilax.utils.shortest_vector(a.pos, b.pos)
    return 1, dx, b.speed, dh


@esquilax.transforms.amap
def flatten_observations(params: Params, observations):
    """
    Convert aggregate neighbour observation into a flattened array
    """
    boid, n_nb, dx_nb, s_nb, h_nb = observations

    def obs_to_nbs():
        _dx_nb = dx_nb / n_nb
        _s_nb = s_nb / n_nb
        _h_nb = h_nb / n_nb

        d = jnp.sqrt(jnp.sum(_dx_nb * _dx_nb)) / 0.1
        phi = jnp.arctan2(_dx_nb[1], _dx_nb[0]) + jnp.pi
        d_phi = esquilax.utils.shortest_vector(boid.heading, phi, 2 * jnp.pi) / jnp.pi

        dh = _h_nb / jnp.pi
        ds = (_s_nb - boid.speed) / (params.max_speed - params.min_speed)

        return jnp.array([d, d_phi, dh, ds])

    return jax.lax.cond(
        n_nb > 0,
        obs_to_nbs,
        lambda: jnp.array([-1.0, 0.0, 0.0, 0.0]),
    )


@esquilax.transforms.amap
def update_velocity(params: Params, x: Tuple[chex.Array, Boid]):
    """
    Update agent velocities from actions
    """
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
def move(_params: Params, x):
    """Update agent positions based on current velocity"""
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


@partial(
    esquilax.transforms.spatial,
    i_range=0.1,
    reduction=jnp.add,
    default=0.0,
    include_self=False,
)
def rewards(_k: chex.PRNGKey, params: Params, a: chex.Array, b: chex.Array):
    """Calculate rewards based on distance from neighbours"""
    d = esquilax.utils.shortest_distance(a, b, norm=True)

    reward = jax.lax.cond(
        d < params.close_range,
        lambda _: -params.collision_penalty,
        lambda _d: jnp.exp(-50 * _d),
        d,
    )

    return reward


class MLP(nn.Module):
    """Simple multi layered network"""

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
