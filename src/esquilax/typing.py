"""
Generic types
"""
from typing import Collection, TypeVar

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
"""PyTree with leaves of a single type"""
