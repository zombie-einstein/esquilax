import jax
import jax.numpy as jnp
import pytest

from esquilax.runner import sim_runner


@pytest.mark.parametrize("rng", [jax.random.PRNGKey(101), 101])
def test_sim_runner(rng):
    def step(i, _, x, y):
        z = y + x * i
        return z, z

    _, history, _ = sim_runner(step, 2, 0, 10, rng, show_progress=False)

    expected = jnp.cumsum(2 * jnp.arange(10))

    assert jnp.array_equal(history, expected)


def test_sim_runner_w_static():
    k = jax.random.PRNGKey(101)

    def step(i, _, x, y, *, func):
        z = func(i, x, y)
        return z, z

    def foo(a, b, c):
        return c + b * a

    _, history, _ = sim_runner(step, 2, 0, 10, k, show_progress=False, func=foo)

    expected = jnp.cumsum(2 * jnp.arange(10))

    assert jnp.array_equal(history, expected)
