"""
RL related data classes
"""
import chex


@chex.dataclass
class Trajectory:
    """
    Observation-action environment trajectories

    Parameters
    ----------
    obs
        Agent observations
    actions
        Agent actions
    action_values
        Any additional values associated with
        the action
    rewards
        Agent rewards
    done
        Terminal flag
    """

    obs: chex.ArrayTree
    actions: chex.ArrayTree
    action_values: chex.ArrayTree
    rewards: chex.ArrayTree
    done: chex.ArrayTree
