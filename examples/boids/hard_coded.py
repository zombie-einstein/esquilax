from functools import partial

import chex
import jax
import jax.numpy as jnp

import esquilax


@chex.dataclass
class Boids:
    pos: chex.Array
    vel: chex.Array


@chex.dataclass
class Params:
    cohesion: float
    avoidance: float
    alignment: float
    max_speed: float
    min_speed: float
    close_range: float


@partial(
    esquilax.transforms.spatial,
    n_bins=5,
    reduction=(jnp.add, jnp.add, jnp.add, jnp.add),
    default=(0, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)),
    include_self=False,
)
def observe(_key: chex.PRNGKey, params: Params, a: Boids, b: Boids):
    v = esquilax.utils.shortest_vector(b.pos, a.pos)
    d = jnp.sum(v**2)

    close_vec = jax.lax.cond(
        d < params.close_range**2,
        lambda: v,
        lambda: jnp.zeros(2),
    )

    return 1, b.pos, b.vel, close_vec


@esquilax.transforms.amap
def steering(_key: chex.PRNGKey, params: Params, observations):
    x, v, n_nb, x_nb, v_nb, v_cl = observations

    def steer():
        x_nb_avg = x_nb / n_nb
        v_nb_avg = v_nb / n_nb
        _dv_x = params.cohesion * esquilax.utils.shortest_vector(x, x_nb_avg)
        _dv_v = params.alignment * esquilax.utils.shortest_vector(v, v_nb_avg)
        return _dv_x + _dv_v

    dv_nb = jax.lax.cond(n_nb > 0, steer, lambda: jnp.zeros(2))
    v = v + dv_nb + v_cl

    return v


@esquilax.transforms.amap
def limit_speed(_key: chex.PRNGKey, params: Params, v: chex.Array):
    s = jnp.sqrt(jnp.sum(v * v))

    v = jax.lax.cond(
        s < params.min_speed,
        lambda _v: params.min_speed * _v / s,
        lambda _v: _v,
        v,
    )

    v = jax.lax.cond(
        s > params.max_speed, lambda _v: params.max_speed * _v / s, lambda _v: _v, v
    )

    return v


@esquilax.transforms.amap
def move(_key: chex.PRNGKey, _params: Params, x):
    pos, vel = x
    return (pos + vel) % 1.0


def step(_i: int, k: chex.PRNGKey, params: Params, boids: Boids):
    n_nb, x_nb, v_nb, v_cl = observe(k, params, boids, boids, pos=boids.pos)

    vel = steering(k, params, (boids.pos, boids.vel, n_nb, x_nb, v_nb, v_cl))
    vel = limit_speed(k, params, vel)
    pos = move(k, params, (boids.pos, vel))

    return Boids(pos=pos, vel=vel), pos


def boids_sim(n: int, n_steps: int, show_progress: bool = True):
    k = jax.random.PRNGKey(101)
    k1, k2 = jax.random.split(k)

    pos = jax.random.uniform(k1, (n, 2))
    vel = 0.01 * jax.random.uniform(k2, (n, 2))
    boids = Boids(pos=pos, vel=vel)

    params = Params(
        cohesion=0.001,
        avoidance=0.05,
        alignment=0.05,
        max_speed=0.05,
        min_speed=0.01,
        close_range=0.02,
    )

    _, history, _ = esquilax.sim_runner(
        step, params, boids, n_steps, k, show_progress=show_progress
    )

    return history


if __name__ == "__main__":
    boids_sim(200, 100)
