import chex
import jax
import jax.numpy as jnp
import pytest
from jax import tree_util

from esquilax import transforms


@pytest.mark.parametrize(
    "args, expected",
    [
        (jnp.arange(10), jnp.arange(2, 12)),
        (
            (jnp.arange(10), jnp.arange(1, 11)),
            (jnp.arange(2, 12), jnp.arange(3, 13)),
        ),
        (
            {"a": jnp.arange(10), "b": jnp.arange(1, 11)},
            {"a": jnp.arange(2, 12), "b": jnp.arange(3, 13)},
        ),
    ],
)
def test_self(args, expected):
    key = jax.random.PRNGKey(101)

    @transforms.amap
    def foo(_, x, y):
        return tree_util.tree_map(lambda z: x + z, y)

    results = foo(key, 2, args)

    chex.assert_trees_all_equal(results, expected)


def test_self_with_static():
    key = jax.random.PRNGKey(101)

    @transforms.amap
    def foo(_, x, y, *, func):
        return func(x, y)

    def bar(a, b):
        return 3 * (a + b)

    args = jnp.arange(5)
    results = foo(key, 2, args, func=bar)
    expected = 3 * (2 + args)

    assert jnp.array_equal(results, expected)
