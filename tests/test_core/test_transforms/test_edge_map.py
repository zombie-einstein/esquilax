import chex
import jax.numpy as jnp
import jax.random
import pytest
from jax import tree_util

from esquilax import transforms


@pytest.mark.parametrize(
    "args, expected",
    [
        (
            (jnp.arange(3), jnp.arange(3), jnp.arange(3)),
            jnp.array([3, 5, 5]),
        ),
        (
            (
                (jnp.arange(3), jnp.arange(1, 4)),
                (jnp.arange(3), jnp.arange(1, 4)),
                (jnp.arange(3), jnp.arange(1, 4)),
            ),
            (
                jnp.array([3, 5, 5]),
                jnp.array([6, 8, 8]),
            ),
        ),
        (
            (
                {"a": jnp.arange(3), "b": jnp.arange(1, 4)},
                {"a": jnp.arange(3), "b": jnp.arange(1, 4)},
                {"a": jnp.arange(3), "b": jnp.arange(1, 4)},
            ),
            {"a": jnp.array([3, 5, 5]), "b": jnp.array([6, 8, 8])},
        ),
    ],
)
def test_edge_map(args, expected):
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(p, x, y, z):
        return tree_util.tree_map(lambda a, b, c: p + a + b + c, x, y, z)

    results = foo(2, *args, edge_idxs=edges)

    chex.assert_trees_all_equal(expected, results)


def test_edge_map_with_none():
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(p, x, y, _z):
        return p + x + y

    results = foo(2, jnp.arange(3), jnp.arange(3), None, edge_idxs=edges)
    expected = jnp.array([3, 4, 3])

    assert jnp.array_equal(results, expected)

    @transforms.edge_map
    def bar(p, x, _y, z):
        return p + x + z

    results = bar(2, jnp.arange(3), None, jnp.arange(3), edge_idxs=edges)
    expected = jnp.array([2, 3, 5])

    assert jnp.array_equal(results, expected)


def test_edge_map_with_static():
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(p, x, y, z, *, func):
        return func(p, x, y, z)

    def bar(a, b, c, d):
        return a + b + c + d

    r = jnp.arange(3)
    results = foo(2, r, r, r, func=bar, edge_idxs=edges)
    expected = jnp.array([3, 5, 5])

    assert jnp.array_equal(results, expected)


def test_edge_map_with_rng(rng_key: chex.PRNGKey):
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(_p, _x, _y, _z, *, key):
        return jax.random.choice(key, 10_000, ())

    results = foo(None, None, None, None, edge_idxs=edges, key=rng_key)

    assert results.shape == (3,)
    assert jnp.unique(results).shape == (3,)
