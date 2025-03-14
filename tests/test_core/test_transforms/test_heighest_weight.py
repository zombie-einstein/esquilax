import chex
import jax.numpy as jnp
import jax.random

import esquilax


def test_highest_weight_edge():
    edges = jnp.array([[0, 0, 2, 2], [2, 3, 0, 1]])
    weights = jnp.array([0.1, 0.4, 0.3, 0.1])
    x = jnp.arange(4)

    def foo(_p, _a, b, e):
        return b + e

    result = esquilax.transforms.highest_weight(foo, default=-1)(
        None, x, x, x, edge_idxs=edges, weights=weights
    )

    assert jnp.array_equal(result, jnp.array([4, -1, 2, -1]))


def test_highest_weight_edge_no_start():
    edges = jnp.array([[0, 0, 2, 2], [2, 3, 0, 1]])
    weights = jnp.array([0.1, 0.4, 0.3, 0.1])
    x = jnp.arange(4)

    def foo(_p, _a, b, _e):
        return b

    result = esquilax.transforms.highest_weight(foo, default=-1, n=4)(
        None, None, x, None, edge_idxs=edges, weights=weights
    )

    assert jnp.array_equal(result, jnp.array([3, -1, 0, -1]))


def test_highest_weight_edge_w_static():
    edges = jnp.array([[0, 0, 2, 2], [2, 3, 0, 1]])
    weights = jnp.array([0.1, 0.4, 0.3, 0.1])
    x = jnp.arange(4)

    def foo(_p, _a, b, _e, *, f):
        return f(b)

    def bar(a):
        return 2 * a

    result = esquilax.transforms.highest_weight(foo, default=-1, n=4)(
        None, None, x, None, f=bar, edge_idxs=edges, weights=weights
    )

    assert jnp.array_equal(result, jnp.array([6, -1, 0, -1]))


def test_highest_weight_edge_w_rng(rng_key: chex.PRNGKey):
    edges = jnp.array([[0, 0, 2], [2, 0, 1]])
    weights = jnp.array([0.1, 0.3, 0.1])

    def foo(_p, _a, _b, _e, *, key):
        return jax.random.choice(key, 10_000, ())

    result = esquilax.transforms.highest_weight(foo, default=-1, n=3)(
        None, None, None, None, edge_idxs=edges, weights=weights, key=rng_key
    )

    assert result.shape == (3,)
    assert jnp.unique(result).shape == (3,)
