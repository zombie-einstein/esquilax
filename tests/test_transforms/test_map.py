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
def test_map(args, expected):
    @transforms.amap
    def foo(x, y):
        return tree_util.tree_map(lambda z: x + z, y)

    results = foo(2, args)

    chex.assert_trees_all_equal(results, expected)


def test_map_with_static():
    @transforms.amap
    def foo(x, y, *, func):
        return func(x, y)

    def bar(a, b):
        return 3 * (a + b)

    args = jnp.arange(5)
    results = foo(2, args, func=bar)
    expected = 3 * (2 + args)

    assert jnp.array_equal(results, expected)


def test_map_w_key(rng_key: chex.PRNGKey):
    @transforms.amap
    def foo(_x, _y, *, key):
        return key

    args = jnp.arange(5)
    results = foo(None, args, key=rng_key)

    expected = jax.random.split(rng_key, 5)

    assert jnp.array_equal(results, expected)
