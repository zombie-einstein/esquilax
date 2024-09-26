import chex
import jax
import jax.numpy as jnp
import pytest

from esquilax import transforms


@pytest.mark.parametrize(
    "args, expected, default",
    [
        (
            [jnp.arange(3), jnp.arange(3), jnp.ones((3,))],
            jnp.array([4, 6, -1]),
            -1.0,
        ),
        (
            [
                (jnp.arange(3), jnp.arange(3)),
                (jnp.arange(3), jnp.arange(3)),
                (jnp.ones((3,)), 1 + jnp.ones((3,))),
            ],
            (jnp.array([4, 6, -1]), jnp.array([5, 7, -2])),
            (-1.0, -2.0),
        ),
        (
            [
                {"a": jnp.arange(3), "b": jnp.arange(3)},
                {"a": jnp.arange(3), "b": jnp.arange(3)},
                {"a": jnp.ones((3,)), "b": 1 + jnp.ones((3,))},
            ],
            {"a": jnp.array([4, 6, -1]), "b": jnp.array([5, 7, -2])},
            {"a": -1.0, "b": -2.0},
        ),
    ],
)
def test_graph_random_neighbour(args, expected, default):
    key = jax.random.PRNGKey(101)
    edges = jnp.array([[0, 0, 1], [1, 1, 2]])

    @transforms.random_edge(default)
    def foo(_, p, x, y, z):
        return jax.tree_util.tree_map(lambda a, b, c: p + a + b + c, x, y, z)

    results = foo(key, 2, *args, edge_idxs=edges)

    chex.assert_trees_all_equal(expected, results)


def test_graph_random_neighbour_w_static():
    key = jax.random.PRNGKey(101)
    edges = jnp.array([[0, 0, 1], [1, 1, 2]])

    args = jnp.arange(3), jnp.arange(3), jnp.ones((3,))
    expected = jnp.array([4, 6, -1])

    @transforms.random_edge(-1.0)
    def foo(_, p, x, y, z, *, func):
        return func(p, x, y, z)

    def bar(a, b, c, d):
        return a + b + c + d

    results = foo(key, 2, *args, func=bar, edge_idxs=edges)

    assert jnp.array_equal(results, expected)
