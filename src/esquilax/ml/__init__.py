"""
RL and Neuro-evolution functionality and utilities

Functions and utilities for multi-agent reinforcement-learning
and neuro-evolution training, where Esquilax simulations
plays the role of training environments.

Training supports several configurations, including multiple RL agents
with individual policies, and training of multiple agent
types or strategies inside the same environment.
"""
from . import evo, rl
from .updates import get_actions, sample_actions
