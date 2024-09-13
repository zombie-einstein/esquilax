from typing import Collection, Tuple

import chex
import jax
import jax.numpy as jnp
import jax_tqdm
from gymnax.environments.environment import Environment, EnvParams

from .agents import Agent


@chex.dataclass
class Trajectory:
    obs: chex.ArrayTree
    actions: chex.ArrayTree
    action_values: chex.ArrayTree
    rewards: chex.ArrayTree
    done: chex.ArrayTree


def sample_actions(
    key: chex.PRNGKey, agents: Collection[Agent], observations: chex.Array
):
    treedef = jax.tree.structure(agents, is_leaf=lambda x: isinstance(x, Agent))
    keys = jax.random.split(key, treedef.num_leaves)
    keys = jax.tree.unflatten(treedef, keys)
    actions, action_values = jax.tree.map(
        lambda agent, k, obs: agent.sample_actions(k, obs),
        agents,
        keys,
        observations,
        is_leaf=lambda x: isinstance(x, Agent),
    )

    return actions, action_values


def update_agents(
    key: chex.PRNGKey,
    agents: Collection[Agent],
    trajectories: Collection[Trajectory],
):
    treedef = jax.tree.structure(agents, is_leaf=lambda x: isinstance(x, Agent))
    keys = jax.random.split(key, treedef.num_leaves)
    keys = jax.tree.unflatten(treedef, keys)

    return jax.tree.map(
        lambda agent, k, t: agent.update(k, t),
        agents,
        keys,
        trajectories,
        is_leaf=lambda x: isinstance(x, Agent),
    )


def train(
    k: chex.PRNGKey,
    agents: Collection[Agent],
    env: Environment,
    env_params: EnvParams,
    n_epochs: int,
    n_env: int,
    n_env_steps: int,
    show_progress: bool = True,
):
    def step(carry, _) -> Tuple[Tuple, Trajectory]:
        _k, _env_state, _obs, _agents = carry
        _k, _k_act, _k_step = jax.random.split(_k, 3)
        _actions, _action_values = sample_actions(_k_act, _agents, _obs)
        _new_obs, _env_state, _rewards, _done, _ = env.step(
            _k_step, _env_state, _actions, env_params
        )
        return (
            (_k, _env_state, _new_obs, _agents),
            Trajectory(
                obs=_obs,
                actions=_actions,
                action_values=_action_values,
                rewards=_rewards,
                done=_done,
            ),
        )

    def sample_trajectories(_k, _agents):
        _obs, _state = env.reset_env(_k, env_params)
        _, trajectories = jax.lax.scan(
            step, (_k, _state, _obs, _agents), None, length=n_env_steps
        )
        return trajectories

    def epoch(carry, _):
        _k, _agents = carry
        _k, _k_sample, _k_train = jax.random.split(_k, 3)
        _k_sample = jax.random.split(_k_sample, n_env)
        _trajectories = jax.vmap(sample_trajectories, in_axes=(0, None))(
            _k_sample, _agents
        )
        _agents = update_agents(_k_train, _agents, _trajectories)
        return (_k, _agents), _trajectories.rewards

    if show_progress:
        epoch = jax_tqdm.scan_tqdm(n_epochs, desc="Epoch")(epoch)

    (_, agents), rewards = jax.lax.scan(
        epoch,
        (k, agents),
        jnp.arange(n_epochs),
    )

    return agents, rewards
