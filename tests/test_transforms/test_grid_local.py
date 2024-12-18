import chex
import jax.numpy as jnp

from esquilax import transforms


def test_grid_local():
    grid = jnp.arange(25).reshape(5, 5)
    x = jnp.array([[1, 1], [4, 4]])
    y = jnp.arange(2)

    def foo(p, a, g):
        return g + a + p

    result = transforms.grid_local(foo, topology="von-neumann")(2, y, grid, co_ords=x)

    assert result.shape == (2, 5)
    expected = jnp.array([[8, 9, 13, 7, 3], [27, 23, 7, 26, 22]])
    assert jnp.array_equal(result, expected)


def test_grid_local_return_tuple():
    grid = jnp.arange(25).reshape(5, 5)
    x = jnp.array([[1, 1], [4, 4]])
    y = jnp.arange(2)

    def foo(p, a, g):
        return p + a + g[0], p + a + g[-1]

    result = transforms.grid_local(foo, topology="von-neumann")(2, y, grid, co_ords=x)

    assert isinstance(result, tuple)
    expected_0 = jnp.array([8, 27])
    expected_1 = jnp.array([3, 22])

    assert jnp.array_equal(result[0], expected_0)
    assert jnp.array_equal(result[1], expected_1)


def test_grid_local_w_key(rng_key: chex.PRNGKey):
    grid = jnp.arange(25).reshape(5, 5)
    x = jnp.array([[1, 1], [4, 4]])

    def foo(p, a, g, *, key):
        return key

    result = transforms.grid_local(foo, topology="von-neumann")(
        None, None, grid, co_ords=x, key=rng_key
    )

    assert result.shape == (2, 2)
    assert not jnp.array_equal(result[0], result[1])
