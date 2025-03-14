import jax
import pytest


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(101)
