import examples
from examples.boids import updates


def test_opinion_dynamics_example():
    n_steps = 100
    n_agents = 50

    results = examples.opinion_dynamics.opinion_dynamics(
        n_agents, 250, n_steps, show_progress=False
    )

    assert results.shape == (n_steps, n_agents)


def test_hard_coded_boids_example():
    n_steps = 100
    n_agents = 50

    results = examples.boids.hard_coded.boids_sim(
        n_agents, n_steps, show_progress=False
    )

    assert results.shape == (n_steps, n_agents, 2)


def test_evo_boids_a():
    pop_size = 2
    n_agents = 10
    n_generations = 3
    n_samples = 2
    n_steps = 8

    f = examples.boids.evolutionary
    params = updates.Params()
    _, scores, (test_paths, test_headings), test_rewards = f.evo_boids(
        params,
        n_agents,
        n_generations,
        n_samples,
        n_steps,
        True,
        show_progress=False,
        layer_width=4,
        pop_size=pop_size,
    )

    assert scores.shape == (n_generations, pop_size)
    assert test_paths.shape == (n_samples, pop_size, n_steps, n_agents, 2)
    assert test_headings.shape == (n_samples, pop_size, n_steps, n_agents)
    assert test_rewards.shape == (n_samples, pop_size, n_steps, n_agents)


def test_evo_boids_b():
    n_agents = 10
    n_generations = 3
    n_samples = 2
    n_steps = 25

    f = examples.boids.evolutionary
    params = updates.Params()
    _, scores, (test_paths, test_headings), test_rewards = f.evo_boids(
        params,
        n_agents,
        n_generations,
        n_samples,
        n_steps,
        False,
        show_progress=False,
        layer_width=4,
    )

    assert scores.shape == (n_generations, n_agents)
    assert test_paths.shape == (n_samples, n_steps, n_agents, 2)
    assert test_headings.shape == (n_samples, n_steps, n_agents)
    assert test_rewards.shape == (n_samples, n_steps, n_agents)


def test_rl_boids():
    n_agents = 10
    n_epochs = 3
    n_env = 2
    n_steps = 25

    agent, rewards = examples.boids.rl.rl_boids(
        updates.Params(),
        n_agents,
        n_epochs,
        n_env,
        n_steps,
        layer_width=4,
        show_progress=False,
    )


def test_gol():
    state_hist = examples.game_of_life.run_model(n_steps=20, show_progress=False)
    assert state_hist.shape == (20, 100, 100)


def test_forest_fire():
    state_hist = examples.forest_fire.run_model(n_steps=50, show_progress=False)
    assert isinstance(state_hist, examples.forest_fire.State)
