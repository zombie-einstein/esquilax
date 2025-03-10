import chex
import jax
import jax.numpy as jnp
import pytest

from esquilax import reductions, transforms


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
    x = jnp.array([[0, 0], [1, 1], [2, 0], [1, 2], [0, 0]])

    def foo(params, a, b):
        return params + a + b

    vals = jnp.arange(5)
    results = transforms.grid(
        foo,
        dims=(3, 3),
        reduction=reductions.add(dtype=int),
        include_self=include_self,
        topology=topology,
    )(2, vals, vals, co_ords=x)

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
    x = jnp.array([[0, 0], [1, 0], [0, 1]])

    def foo(params, a, b):
        return params + a + b

    vals = jnp.column_stack([jnp.arange(3), jnp.arange(3) + 1])

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=reductions.add(shape=(2,), dtype=int),
        include_self=include_self,
        topology=topology,
    )(2, vals, vals, co_ords=x)

    assert jnp.array_equal(results, jnp.array(expected))


def test_grid_w_static():
    x = jnp.array([[0, 0], [1, 1]])

    def foo(params, a, b, *, func):
        return func(params, a, b)

    def bar(a, b, c):
        return a + b + c

    vals = jnp.arange(2)

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=reductions.add(dtype=int),
        topology="moore",
        include_self=True,
    )(2, vals, vals, co_ords=x, func=bar)

    expected = jnp.array([14, 16])
    assert jnp.array_equal(results, expected)


def test_grid_w_none():
    x = jnp.array([[0, 0], [1, 1]])

    def foo(params, a, b):
        return params + a

    vals = jnp.arange(2)

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=reductions.add(dtype=int),
        topology="moore",
        include_self=False,
    )(2, vals, None, co_ords=x)

    expected = jnp.array([8, 12])
    assert jnp.array_equal(results, expected)

    def bar(params, a, b):
        return params + b

    results = transforms.grid(
        bar,
        dims=(2, 2),
        reduction=reductions.add(dtype=int),
        topology="moore",
        include_self=False,
    )(2, None, vals, co_ords=x)

    expected = jnp.array([12, 8])
    assert jnp.array_equal(results, expected)


def test_grid_w_mixed_types():
    xa = jnp.array([[0, 0], [1, 1]])
    xb = jnp.array([[1, 0], [0, 1]])

    def foo(params, a, b):
        return params + a + b

    vals_a = jnp.arange(2)
    vals_b = 2 + vals_a

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=reductions.add(dtype=int),
        topology="von-neumann",
        include_self=False,
    )(2, vals_a, vals_b, co_ords=xa, co_ords_b=xb)

    expected = jnp.array([18, 22])
    assert jnp.array_equal(results, expected)


def test_grid_non_square():
    x = jnp.array([[0, 0], [1, 1], [0, 2], [0, 1]])

    def foo(params, a, b):
        return params + a + b

    vals = jnp.arange(4)

    results = transforms.grid(
        foo,
        dims=(2, 3),
        reduction=reductions.add(dtype=int),
        topology="von-neumann",
        include_self=False,
    )(2, vals, vals, co_ords=x)

    expected = jnp.array([9, 12, 11, 24])

    assert jnp.array_equal(results, expected)


def test_grid_w_random(rng_key: chex.PRNGKey):
    x = jnp.array([[0, 0], [1, 1]])

    def foo(_params, _a, _b, *, key):
        return jax.random.choice(
            key,
            1_000,
            (),
        )

    results = transforms.grid(
        foo,
        dims=(2, 2),
        reduction=reductions.add(dtype=int),
        topology="moore",
        include_self=True,
    )(2, None, None, co_ords=x, key=rng_key)

    assert results.shape == (2,)
    assert results[0] != results[1]
