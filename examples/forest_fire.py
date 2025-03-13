"""
Simple forest fire spread model

Simple model of forest fire spread adapted from
the NetLogo version https://ccl.northwestern.edu/netlogo/models/Fire

Trees are randomly distributed on a grid with a given density
if an unburnt tree has a neighbouring tree on fire then it will
catch fire. Trees that are on fire become burnt/embers in the
next turn.
"""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

import esquilax


@chex.dataclass
class State:
    status: chex.Array
    co_ords: chex.Array


def step(
    _i: int,
    _k: chex.PRNGKey,
    _params: None,
    state: State,
    *,
    dimensions: Tuple[int, int],
) -> Tuple[State, chex.Array]:
    catch_fire = esquilax.transforms.grid(
        lambda _p, a, b: jnp.logical_and(a == 0, b == 1),
        reduction=esquilax.reductions.logical_or(),
        dims=dimensions,
        include_self=False,
    )(_params, state.status, state.status, co_ords=state.co_ords)

    new_status = jnp.where(state.status == 1, 2, state.status)
    new_status = jnp.where(catch_fire, 1, new_status)

    new_state = State(status=new_status, co_ords=state.co_ords)

    return new_state, state.status


def init_state(key: chex.PRNGKey, density: float, dimensions: Tuple[int, int]) -> State:
    n = dimensions[0] * dimensions[1]
    cell_co_ords = (
        jnp.stack(jnp.meshgrid(jnp.arange(dimensions[0]), jnp.arange(dimensions[1])))
        .reshape(-1, n)
        .T
    )
    k1, k2 = jax.random.split(key, 2)
    probs = jax.random.uniform(k1, (n,))
    idxs = jnp.argwhere(probs < density)[:, 0]
    co_ords = cell_co_ords[idxs]
    statuses = jnp.zeros(co_ords.shape[:1], dtype=int)
    fire_start = jax.random.choice(k2, statuses.shape[0])
    statuses = statuses.at[fire_start].set(1)
    return State(status=statuses, co_ords=co_ords)


def flatten_state(
    dimensions: Tuple[int, int], co_ords: chex.Array, statuses: chex.Array
) -> chex.Array:
    def flatten(s):
        x = jnp.zeros(dimensions, dtype=int)
        x = x.at[co_ords[:, 0], co_ords[:, 1]].set(s + 1)
        return x

    return jax.vmap(flatten)(statuses)


def run_model(
    seed: int = 101,
    n_steps: int = 200,
    density: float = 0.5,
    dimensions: Tuple[int, int] = (100, 100),
    show_progress: bool = True,
) -> chex.Array:
    k = jax.random.PRNGKey(seed)
    s0 = init_state(k, density, dimensions)
    _, status_hist, _ = esquilax.sim_runner(
        step, None, s0, n_steps, k, show_progress=show_progress, dimensions=dimensions
    )
    state_hist = flatten_state(dimensions, s0.co_ords, status_hist)
    return state_hist


if __name__ == "__main__":
    states = run_model()

    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        f, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(states[0])
        ax.set_xticks([])
        ax.set_yticks([])

        def update_fn(s):
            img.set_data(s)
            return (img,)

        anim = animation.FuncAnimation(f, update_fn, states[1:])
        plt.show()

    except ImportError:
        print("Matplotlib required to show animation")
