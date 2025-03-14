import chex
import jax
import jax.numpy as jnp
from jax import tree_util

from esquilax import reductions, transforms


def test_graph_reduce_array():
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(p, x, y, z):
        return p + x + y + z

    results = transforms.graph_reduce(foo, reduction=reductions.add(dtype=int))(
        2, jnp.arange(3), jnp.arange(3), jnp.arange(3), edge_idxs=edges
    )
    expected = jnp.array([8, 5, 0])

    assert jnp.array_equal(results, expected)


def test_graph_reduce_array_no_starts():
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(p, x, y, z):
        return p + y + z

    results = transforms.graph_reduce(foo, reduction=reductions.add(dtype=int), n=3)(
        2, None, jnp.arange(3), jnp.arange(3), edge_idxs=edges
    )
    expected = jnp.array([8, 4, 0])

    assert jnp.array_equal(results, expected)


def test_graph_reduce_tuple():
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(p, x, y, z):
        return tree_util.tree_map(lambda a, b, c: p + a + b + c, x, y, z)

    reduction = reductions.Reduction((jnp.add, jnp.add), (0, 0))

    results = transforms.graph_reduce(
        foo,
        reduction=reduction,
    )(
        2,
        (jnp.arange(3), jnp.arange(3)),
        (jnp.arange(3), jnp.arange(3)),
        (jnp.arange(3), jnp.arange(3)),
        edge_idxs=edges,
    )
    expected = (jnp.array([8, 5, 0]), jnp.array([8, 5, 0]))

    chex.assert_trees_all_equal(expected, results)


def test_graph_reduce_w_static():
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(p, x, y, z, *, func):
        return func(p, x, y, z)

    def bar(a, b, c, d):
        return a + b + c + d

    results = transforms.graph_reduce(foo, reduction=reductions.add(dtype=int))(
        2, jnp.arange(3), jnp.arange(3), jnp.arange(3), func=bar, edge_idxs=edges
    )
    expected = jnp.array([8, 5, 0])

    assert jnp.array_equal(results, expected)


def test_graph_reduce_w_rng(rng_key: chex.PRNGKey):
    edges = jnp.array([[0, 0, 1], [1, 2, 0]])

    def foo(_p, _x, _y, _z, *, key):
        return jax.random.choice(key, 10_000, ())

    results = transforms.graph_reduce(foo, reduction=reductions.add(dtype=int), n=3)(
        None, None, None, None, key=rng_key, edge_idxs=edges
    )

    assert results.shape == (3,)
    assert jnp.unique(results).shape == (3,)
