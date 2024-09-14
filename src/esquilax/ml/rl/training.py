from typing import Callable, Collection, Tuple

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


def key_tree_split(key: chex.PRNGKey, tree, typ) -> Collection[chex.PRNGKey]:
    treedef = jax.tree.structure(tree, is_leaf=lambda x: isinstance(x, typ))
    keys = jax.random.split(key, treedef.num_leaves)
    keys = jax.tree.unflatten(treedef, keys)
    return keys


def sample_actions(
    key: chex.PRNGKey, agents: Collection[Agent], observations: chex.Array
):
    keys = key_tree_split(key, agents, Agent)
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
    keys = key_tree_split(key, agents, Agent)
    return jax.tree.map(
        lambda agent, k, t: agent.update(k, t),
        agents,
        keys,
        trajectories,
        is_leaf=lambda x: isinstance(x, Agent),
    )


def scan_step(env_params: EnvParams, env: Environment) -> Callable:
    def inner_step(carry, _) -> Tuple[Tuple, Trajectory]:
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

    return inner_step


def train(
    key: chex.PRNGKey,
    agents: Collection[Agent],
    env: Environment,
    env_params: EnvParams,
    n_epochs: int,
    n_env: int,
    n_env_steps: int,
    show_progress: bool = True,
):
    step = scan_step(env_params, env)

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
        _agents, _train_data = update_agents(_k_train, _agents, _trajectories)
        return (_k, _agents), (_trajectories.rewards, _train_data)

    if show_progress:
        epoch = jax_tqdm.scan_tqdm(n_epochs, desc="Epoch")(epoch)

    (_, agents), (rewards, train_data) = jax.lax.scan(
        epoch,
        (key, agents),
        jnp.arange(n_epochs),
    )

    return agents, rewards, train_data


def test(
    key: chex.PRNGKey,
    agents: Collection[Agent],
    env: Environment,
    env_params: EnvParams,
    n_env: int,
    n_env_steps: int,
    show_progress: bool = True,
):
    step = scan_step(env_params, env)

    if show_progress:
        step = jax_tqdm.scan_tqdm(n_env_steps, desc="Step")(step)

    def sample_trajectories(_k, _agents):
        _obs, _state = env.reset_env(_k, env_params)
        _, _trajectories = jax.lax.scan(
            step, (_k, _state, _obs, _agents), None, length=n_env_steps
        )
        return _trajectories

    k_sample = jax.random.split(key, n_env)

    trajectories = jax.vmap(sample_trajectories, in_axes=(0, None))(k_sample, agents)

    return trajectories
