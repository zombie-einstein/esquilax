from functools import partial

import chex
import jax
import jax.numpy as jnp

import esquilax


@chex.dataclass
class Params:
    strength: jnp.float32
    threshold: jnp.float32


@chex.dataclass
class SimState:
    opinions: chex.Array
    edges: chex.Array
    weights: chex.Array


@partial(
    esquilax.transforms.graph_reduce, reduction=(jnp.add, jnp.add), default=(0, 0.0)
)
def collect_opinions(_, params: Params, my_opinion, your_opinion, weight):
    d = jnp.abs(my_opinion - your_opinion)
    w = params.strength * weight

    return jax.lax.cond(
        d < params.threshold,
        lambda: (1, (1.0 - w) * my_opinion + w * your_opinion),
        lambda: (0, 0.0),
    )


def step(_, k, params: Params, state: SimState):
    n, new_opinions = collect_opinions(
        k, params, state.opinions, state.opinions, state.weights, edge_idxs=state.edges
    )

    new_opinions = jnp.where(n > 0, new_opinions / n, state.opinions)

    new_state = SimState(
        opinions=new_opinions, edges=state.edges, weights=state.weights
    )

    return new_state, new_opinions


def opinion_dynamics(
    n_agents: int, n_edges: int, n_steps: int, show_progress: bool = True
):
    k = jax.random.PRNGKey(101)
    k, k1, k2, k3 = jax.random.split(k, 4)

    edges = jax.random.choice(k1, n_agents, shape=(2, n_edges))
    weights = jax.random.uniform(k3, shape=(n_edges,))
    edges, weights = esquilax.utils.sort_edges(edges, weights)

    opinions = jax.random.uniform(k2, shape=(n_agents,))

    initial_state = SimState(opinions=opinions, edges=edges, weights=weights)

    params = Params(strength=0.1, threshold=0.2)

    _, history, _ = esquilax.sim_runner(
        step, params, initial_state, n_steps, k, show_progress=show_progress
    )

    return history


if __name__ == "__main__":
    opinion_dynamics(100, 400, 200)
