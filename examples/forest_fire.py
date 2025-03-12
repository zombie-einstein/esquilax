from typing import Tuple

import chex
import jax
import jax.numpy as jnp

import esquilax


@chex.dataclass
class State:
    status: int
    co_ords: chex.Array


def step(
    _i: int,
    _k: chex.PRNGKey,
    _params: None,
    state: State,
    *,
    dimensions: Tuple[int, int],
) -> Tuple[State, State]:
    catch_fire = esquilax.transforms.grid(
        lambda _p, a, b: jnp.logical_and(a == 0, b == 1),
        reduction=esquilax.reductions.logical_or(),
        dims=dimensions,
        include_self=False,
    )(_params, state.status, state.status, co_ords=state.co_ords)

    new_status = jnp.where(state.status == 1, 2, state.status)
    new_status = jnp.where(catch_fire, 1, new_status)

    new_state = State(status=new_status, co_ords=state.co_ords)

    return new_state, state


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
    statuses = jnp.zeros(co_ords.shape[:1])
    fire_start = jax.random.choice(k2, statuses.shape[0])
    statuses = statuses.at[fire_start].set(1)
    return State(status=statuses, co_ords=co_ords)


def run_model(
    seed: int = 101,
    n_steps: int = 200,
    density: float = 0.5,
    dimensions: Tuple[int, int] = (100, 100),
    show_progress: bool = True,
) -> State:
    k = jax.random.PRNGKey(seed)
    s0 = init_state(k, density, dimensions)
    _, state_hist, _ = esquilax.sim_runner(
        step, None, s0, n_steps, k, show_progress=show_progress, dimensions=dimensions
    )
    return state_hist


if __name__ == "__main__":
    run_model()
