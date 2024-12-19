from typing import List, Tuple, Union

import chex
import jax.numpy as jnp
import jax.random
import pytest

from esquilax import transforms


@pytest.mark.parametrize(
    "expected, topology, i_range",
    [
        ([3, 3, -1], "same-cell", 10.0),
        ([3, 3, 5], "von-neumann", 10.0),
        ([3, 3, 5], "moore", 10.0),
        ([-1, -1, -1], "moore", 0.00001),
    ],
)
def test_nearest_neighbour(expected: chex.Array, topology: str, i_range: float):
    x = jnp.array([[0.1, 0.1], [0.1, 0.2], [0.1, 0.6]])

    def foo(params, a, b):
        return params + a + b

    vals = jnp.arange(3)
    results = transforms.nearest_neighbour(
        foo, n_bins=2, default=-1, topology=topology, i_range=i_range
    )(2, vals, vals, pos=x)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


@pytest.mark.parametrize(
    "x, i_range, dims, expected",
    [
        ([[1.0, 0.95], [1.0, 1.05], [1.0, 0.2]], 0.2, 2.0, [3, 3, -1]),
        ([[1.0, 1.99], [1.0, 0.05], [1.0, 1.95]], 0.2, 2.0, [4, 3, 4]),
        ([[0.2, 1.99], [0.2, 0.05], [0.2, 1.95]], 0.2, (0.4, 2.0), [4, 3, 4]),
        ([[0.39, 1.0], [0.02, 1.0], [0.38, 1.0]], 0.2, (0.4, 2.0), [4, 3, 4]),
    ],
)
def test_nearest_neighbour_non_unit_region(
    x: List[List[float]],
    i_range: float,
    dims: Union[float, Tuple[float, float]],
    expected: List[int],
):
    x = jnp.array(x)

    def foo(params, a, b):
        return params + a + b

    vals = jnp.arange(x.shape[0])
    results = transforms.nearest_neighbour(
        foo, default=-1, topology="moore", i_range=i_range, dims=dims
    )(2, vals, vals, pos=x)

    assert jnp.array_equal(results, jnp.array(expected))


def test_nearest_neighbour_w_rng(rng_key: chex.PRNGKey):
    x = jnp.array([[0.1, 0.1], [0.1, 0.2]])

    def foo(_p, _a, _b, *, key):
        return jax.random.choice(key, 10_000, ())

    results = transforms.nearest_neighbour(foo, default=-1, i_range=0.3)(
        None, None, None, key=rng_key, pos=x
    )

    assert results.shape == (2,)
    assert results[0] != results[1]
