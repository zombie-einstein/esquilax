import jax
import jax.numpy as jnp

from esquilax import transforms


def test_grid_local():
    k = jax.random.PRNGKey(101)

    grid = jnp.arange(25).reshape(5, 5)
    x = jnp.array([[1, 1], [4, 4]])
    y = jnp.arange(2)

    def foo(_, p, a, g):
        return g + a + p

    result = transforms.grid_local(foo, topology="von-neumann")(
        k, 2, y, grid, co_ords=x
    )

    assert result.shape == (2, 5)
    expected = jnp.array([[8, 9, 13, 7, 3], [27, 23, 7, 26, 22]])
    assert jnp.array_equal(result, expected)


def test_grid_local_return_tuple():
    k = jax.random.PRNGKey(101)

    grid = jnp.arange(25).reshape(5, 5)
    x = jnp.array([[1, 1], [4, 4]])
    y = jnp.arange(2)

    def foo(_, p, a, g):
        return p + a + g[0], p + a + g[-1]

    result = transforms.grid_local(foo, topology="von-neumann")(
        k, 2, y, grid, co_ords=x
    )

    assert isinstance(result, tuple)
    expected_0 = jnp.array([8, 27])
    expected_1 = jnp.array([3, 22])

    assert jnp.array_equal(result[0], expected_0)
    assert jnp.array_equal(result[1], expected_1)
