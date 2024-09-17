"""
Reinforcement learning utilities and training functionality
"""
from .agents import Agent, BatchPolicyAgent, SharedPolicyAgent
from .environment import Environment
from .training import test, train
from .types import Trajectory
