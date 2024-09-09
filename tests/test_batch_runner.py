import jax.numpy as jnp
import pytest

import esquilax


@pytest.fixture
def sim():
    class TestSim(esquilax.Sim):
        def default_params(self):
            return 10

        def initial_state(self, k):
            return 10, 20

        def step(self, i, k, params, state):
            new_state = (params + state[0], params + state[1])
            return new_state, state

    return TestSim()


def test_single_params(sim):
    n_samples = 2
    n_steps = 10

    results = esquilax.batch_sim_runner(
        sim,
        n_samples,
        n_steps,
        101,
        show_progress=False,
    )

    assert isinstance(results, tuple)
    assert results[0].shape == (n_samples, n_steps)
    assert results[1].shape == (n_samples, n_steps)


def test_param_set(sim):
    n_params = 3
    n_samples = 2
    n_steps = 10

    param_set = jnp.arange(n_params)

    results = esquilax.batch_sim_runner(
        sim, n_samples, n_steps, 101, show_progress=False, param_samples=param_set
    )

    assert isinstance(results, tuple)
    assert results[0].shape == (n_params, n_samples, n_steps)
    assert results[1].shape == (n_params, n_samples, n_steps)
