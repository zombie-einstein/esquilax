import chex
import jax
import jax.numpy as jnp
from jax import tree_util

from esquilax import transforms


def test_graph_reduce_array():
    key = jax.random.PRNGKey(101)
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(_, p, x, y, z):
        return p + x + y + z

    results = transforms.graph_reduce(foo, reduction=jnp.add, default=0)(
        key, 2, jnp.arange(3), jnp.arange(3), jnp.arange(3), edge_idxs=edges
    )
    expected = jnp.array([8, 5, 0])

    assert jnp.array_equal(results, expected)


def test_graph_reduce_array_no_starts():
    key = jax.random.PRNGKey(101)
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(_, p, x, y, z):
        return p + y + z

    results = transforms.graph_reduce(foo, reduction=jnp.add, default=0, n=3)(
        key, 2, None, jnp.arange(3), jnp.arange(3), edge_idxs=edges
    )
    expected = jnp.array([8, 4, 0])

    assert jnp.array_equal(results, expected)


def test_graph_reduce_tuple():
    key = jax.random.PRNGKey(101)
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(_, p, x, y, z):
        return tree_util.tree_map(lambda a, b, c: p + a + b + c, x, y, z)

    results = transforms.graph_reduce(
        foo, reduction=(jnp.add, jnp.add), default=(0, 0)
    )(
        key,
        2,
        (jnp.arange(3), jnp.arange(3)),
        (jnp.arange(3), jnp.arange(3)),
        (jnp.arange(3), jnp.arange(3)),
        edge_idxs=edges,
    )
    expected = (jnp.array([8, 5, 0]), jnp.array([8, 5, 0]))

    chex.assert_trees_all_equal(expected, results)


def test_graph_reduce_w_static():
    key = jax.random.PRNGKey(101)
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(_, p, x, y, z, *, func):
        return func(p, x, y, z)

    def bar(a, b, c, d):
        return a + b + c + d

    results = transforms.graph_reduce(foo, reduction=jnp.add, default=0)(
        key, 2, jnp.arange(3), jnp.arange(3), jnp.arange(3), func=bar, edge_idxs=edges
    )
    expected = jnp.array([8, 5, 0])

    assert jnp.array_equal(results, expected)
