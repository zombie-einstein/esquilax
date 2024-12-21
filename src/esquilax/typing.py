"""
Generic types
"""
from typing import Collection, TypeVar

import chex

TSimState = TypeVar("TSimState")
"""Generic simulation state"""
TSimParams = TypeVar("TSimParams")
"""Generic simulation parameters"""
TEnvState = TypeVar("TEnvState")
"""Generic environment state"""
TEnvParams = TypeVar("TEnvParams")
"""Generic environment parameters"""
T = TypeVar("T")
TypedPyTree = T | Collection[T]
"""Reduction function(s) type"""
Default = bool | int | float | chex.ArrayTree
"""Default reduction value types"""
