from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import jax.typing
import pytest

import esquilax


def test_reduction_assertion() -> None:
    with pytest.raises(AssertionError):
        esquilax.reductions.Reduction(fn=jnp.add, id=(0, 0))

    with pytest.raises(AssertionError):
        esquilax.reductions.Reduction(fn=(jnp.add, jnp.add), id=0)


@pytest.mark.parametrize(
    "shape, dtype",
    [((), int), ((), float), ((5,), int), ((5,), float)],
)
def test_add_reduce(
    rng_key: chex.PRNGKey, shape: Sequence[int], dtype: jax.typing.DTypeLike
) -> None:
    reduction = esquilax.reductions.add(shape, dtype)

    k1, k2 = jax.random.split(rng_key)
    a = jax.random.choice(k1, 10_000, shape=shape).astype(dtype)
    b = jax.random.choice(k2, 10_000, shape=shape).astype(dtype)

    result = reduction.fn(a, b)

    assert jnp.array_equal(result, a + b)
    assert jnp.array_equal(reduction.id, jnp.zeros_like(result))


@pytest.mark.parametrize("shape", [(), (5,)])
def test_logical_or_reduce(rng_key: chex.PRNGKey, shape: Sequence[int]) -> None:
    reduction = esquilax.reductions.logical_or(shape)

    k1, k2 = jax.random.split(rng_key)
    a = jax.random.choice(k1, 2, shape=shape).astype(bool)
    b = jax.random.choice(k2, 2, shape=shape).astype(bool)

    result = reduction.fn(a, b)

    assert jnp.array_equal(result, jnp.logical_or(a, b))
    assert jnp.array_equal(reduction.id, jnp.zeros_like(result))


@pytest.mark.parametrize(
    "shape, dtype",
    [((), int), ((), float), ((5,), int), ((5,), float)],
)
def test_min_and_max(
    rng_key: chex.PRNGKey, shape: Sequence[int], dtype: jax.typing.DTypeLike
) -> None:
    max_reduction = esquilax.reductions.add(shape, dtype)
    min_reduction = esquilax.reductions.add(shape, dtype)

    x = jax.random.choice(rng_key, 10_000, shape=shape).astype(dtype)

    max_result = max_reduction.fn(max_reduction.id, x)
    min_result = min_reduction.fn(min_reduction.id, x)

    assert jnp.array_equal(max_result, x)
    assert jnp.array_equal(min_result, x)
