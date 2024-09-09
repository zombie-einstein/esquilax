import chex
import jax
import jax.numpy as jnp
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
    key = jax.random.PRNGKey(101)

    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(_, p, x, y, z):
        return tree_util.tree_map(lambda a, b, c: p + a + b + c, x, y, z)

    results = foo(key, 2, *args, edge_idxs=edges)

    chex.assert_trees_all_equal(expected, results)


def test_edge_map_with_none():
    key = jax.random.PRNGKey(101)

    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(_, p, x, y, _z):
        return p + x + y

    results = foo(key, 2, jnp.arange(3), jnp.arange(3), None, edge_idxs=edges)
    expected = jnp.array([3, 4, 3])

    assert jnp.array_equal(results, expected)

    @transforms.edge_map
    def bar(_, p, x, _y, z):
        return p + x + z

    results = bar(key, 2, jnp.arange(3), None, jnp.arange(3), edge_idxs=edges)
    expected = jnp.array([2, 3, 5])

    assert jnp.array_equal(results, expected)


def test_edge_map_with_static():
    key = jax.random.PRNGKey(101)

    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    @transforms.edge_map
    def foo(_, p, x, y, z, *, func):
        return func(p, x, y, z)

    def bar(a, b, c, d):
        return a + b + c + d

    r = jnp.arange(3)
    results = foo(key, 2, r, r, r, func=bar, edge_idxs=edges)
    expected = jnp.array([3, 5, 5])

    assert jnp.array_equal(results, expected)
