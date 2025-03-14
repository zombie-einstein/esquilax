# import examples
from examples import forest_fire, game_of_life, opinion_dynamics
from examples.boids import hard_coded


def test_opinion_dynamics_example():
    n_steps = 100
    n_agents = 50

    results = opinion_dynamics.opinion_dynamics(
        n_agents, 250, n_steps, show_progress=False
    )

    assert results.shape == (n_steps, n_agents)


def test_hard_coded_boids_example():
    n_steps = 100
    n_agents = 50

    results = hard_coded.boids_sim(n_agents, n_steps, show_progress=False)

    assert results.shape == (n_steps, n_agents, 2)


def test_gol():
    state_hist = game_of_life.run_model(n_steps=20, show_progress=False)
    assert state_hist.shape == (20, 100, 100)


def test_forest_fire():
    state_hist = forest_fire.run_model(n_steps=20, show_progress=False)
    assert state_hist.shape == (20, 100, 100)
