from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random

import esquilax


def step(
    _i: int,
    _k: chex.PRNGKey,
    _params: None,
    state: chex.Array,
    *,
    dimensions: Tuple[int, int],
) -> Tuple[chex.Array, chex.Array]:
    # Co-ordinates of individual cells on a grid
    cell_co_ords = (
        jnp.stack(jnp.meshgrid(jnp.arange(dimensions[0]), jnp.arange(dimensions[1])))
        .reshape(-1, dimensions[0] * dimensions[1])
        .T
    )

    # Accumulate values of neighbouring cells
    accumulated = esquilax.transforms.grid(
        lambda _p, _a, b: b,
        reduction=esquilax.reductions.add(dtype=int),
        dims=dimensions,
        include_self=False,
    )(_params, state, state, co_ords=cell_co_ords)

    # Update state according to game-of-life rules
    new_state = jnp.where(
        jnp.equal(state, 0),
        jnp.where(jnp.equal(accumulated, 3), 1, 0),
        jnp.where(accumulated < 2, 0, jnp.where(accumulated > 3, 0, 1)),
    )

    return new_state, state


def init_state(key: chex.PRNGKey, dimensions: Tuple[int, int]) -> chex.Array:
    return jax.random.choice(key, 2, dimensions).reshape(-1)


def run_model(
    seed: int = 101,
    n_steps: int = 200,
    dimensions=(100, 100),
    show_progress: bool = True,
) -> chex.Array:
    k = jax.random.PRNGKey(seed)
    x = init_state(k, dimensions)
    _, state_hist, _ = esquilax.sim_runner(
        step, None, x, n_steps, k, show_progress=show_progress, dimensions=dimensions
    )
    state_hist = state_hist.reshape(n_steps, dimensions[0], dimensions[1])
    return state_hist


if __name__ == "__main__":
    run_model()
