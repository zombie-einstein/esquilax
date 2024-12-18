import chex
import jax
import jax.numpy as jnp
import pytest

from esquilax import transforms


@pytest.mark.parametrize(
    "expected, include_self, topology",
    [
        ([6, 0, 0, 0, 6], False, "same-cell"),
        ([8, 4, 6, 8, 16], True, "same-cell"),
        ([10, 6, 12, 6, 14], False, "von-neumann"),
        ([12, 10, 18, 14, 24], True, "von-neumann"),
        ([18, 21, 24, 27, 30], False, "moore"),
        ([20, 25, 30, 35, 40], True, "moore"),
    ],
)
def test_grid_transform(expected: chex.Array, include_self: bool, topology: str):
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0, 0], [1, 1], [2, 0], [1, 2], [0, 0]])

    def foo(_, params, a, b):
        return params + a + b

    vals = jnp.arange(5)
    results = transforms.grid(
        foo,
        dims=(3, 3),
        reduction=jnp.add,
        default=0,
        include_self=include_self,
        topology=topology,
    )(k, 2, vals, vals, co_ords=x)

    assert jnp.array_equal(results, jnp.array(expected))


@pytest.mark.parametrize(
    "expected, include_self, topology",
    [
        ([[0, 0], [0, 0], [0, 0]], False, "same-cell"),
        ([[2, 4], [4, 6], [6, 8]], True, "same-cell"),
        ([[14, 22], [26, 38], [28, 40]], False, "moore"),
        ([[16, 26], [30, 44], [34, 48]], True, "moore"),
    ],
)
def test_grid_transform_w_array(
    expected: chex.Array, include_self: bool, topology: str
):
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0, 0], [1, 0], [0, 1]])

    def foo(_, params, a, b):
        return params + a + b

    vals = jnp.column_stack([jnp.arange(3), jnp.arange(3) + 1])

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=jnp.add,
        default=jnp.zeros(2, dtype=int),
        include_self=include_self,
        topology=topology,
    )(k, 2, vals, vals, co_ords=x)

    assert jnp.array_equal(results, jnp.array(expected))


def test_grid_w_static():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0, 0], [1, 1]])

    def foo(_, params, a, b, *, func):
        return func(params, a, b)

    def bar(a, b, c):
        return a + b + c

    vals = jnp.arange(2)

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=jnp.add,
        default=0,
        topology="moore",
        include_self=True,
    )(k, 2, vals, vals, co_ords=x, func=bar)

    expected = jnp.array([14, 16])
    assert jnp.array_equal(results, expected)


def test_grid_w_none():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0, 0], [1, 1]])

    def foo(_, params, a, b):
        return params + a

    vals = jnp.arange(2)

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=jnp.add,
        default=0,
        topology="moore",
        include_self=False,
    )(k, 2, vals, None, co_ords=x)

    expected = jnp.array([8, 12])
    assert jnp.array_equal(results, expected)

    def bar(_, params, a, b):
        return params + b

    results = transforms.grid(
        bar,
        dims=(2, 2),
        reduction=jnp.add,
        default=0,
        topology="moore",
        include_self=False,
    )(k, 2, None, vals, co_ords=x)

    expected = jnp.array([12, 8])
    assert jnp.array_equal(results, expected)


def test_grid_w_mixed_types():
    k = jax.random.PRNGKey(101)
    xa = jnp.array([[0, 0], [1, 1]])
    xb = jnp.array([[1, 0], [0, 1]])

    def foo(_, params, a, b):
        return params + a + b

    vals_a = jnp.arange(2)
    vals_b = 2 + vals_a

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=jnp.add,
        default=0,
        topology="von-neumann",
        include_self=False,
    )(k, 2, vals_a, vals_b, co_ords=xa, co_ords_b=xb)

    expected = jnp.array([18, 22])
    assert jnp.array_equal(results, expected)


def test_grid_non_square():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0, 0], [1, 1], [0, 2], [0, 1]])

    def foo(_, params, a, b):
        return params + a + b

    vals = jnp.arange(4)

    results = transforms.grid(
        foo,
        dims=(2, 3),
        reduction=jnp.add,
        default=0,
        topology="von-neumann",
        include_self=False,
    )(k, 2, vals, vals, co_ords=x)

    expected = jnp.array([9, 12, 11, 24])

    assert jnp.array_equal(results, expected)
