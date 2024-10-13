"""
Generic types
"""
from typing import Callable, Collection, TypeVar, Union

import chex
from jax._src.numpy.ufuncs import ufunc

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
Reduction = TypedPyTree[Union[Callable | ufunc]]
"""Reduction function(s) type"""
Default = bool | int | float | chex.ArrayTree
"""Default reduction value types"""
