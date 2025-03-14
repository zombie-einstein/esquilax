"""
Reinforcement learning utilities and training functionality

.. note::

   This module requires additional dependencies that can be installed
   using the installation extra ``rl``, for example with pip using

   .. code::

      pip install esquilax[rl]
"""
try:
    import flax  # noqa: F401
except ImportError as e:
    raise ImportError(
        "This functionality requires additional dependencies."
        "They can be installed with the extra `pip install esquilax[rl]`"
    ) from e
from .agent import Agent
from .agent_state import AgentState, BatchAgentState
from .environment import Environment
from .training import test, train, train_and_test
from .types import Trajectory
