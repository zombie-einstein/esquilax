import chex


@chex.dataclass
class Trajectory:
    """
    Observation-action environment trajectories

    Parameters
    ----------
    obs: chex.ArrayTree
        Agent observations
    actions: chex.ArrayTree
        Agent actions
    action_values: chex.ArrayTree
        Any additional values associated with
        the action
    rewards: chex.ArrayTree
        Agent rewards
    done: chex.ArrayTree
        Terminal flag
    """

    obs: chex.ArrayTree
    actions: chex.ArrayTree
    action_values: chex.ArrayTree
    rewards: chex.ArrayTree
    done: chex.ArrayTree
