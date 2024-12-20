from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp
from jax._src.numpy.ufuncs import ufunc

from esquilax.typing import TypedPyTree


class Reduction:
    __slots__ = ["fn", "id"]

    def __init__(self, fn: TypedPyTree[Callable | ufunc], id: chex.Numeric) -> None:
        chex.assert_trees_all_equal_structs(
            fn, id
        ), "Reduction and default PyTrees should have the same structure"
        self.fn = fn
        self.id = id


def add(shape: Sequence[int] = (), dtype: jax.typing.DTypeLike = float) -> Reduction:
    return Reduction(fn=jnp.add, id=jnp.zeros(shape, dtype=dtype))


def logical_or(shape: Sequence[int] = ()) -> Reduction:
    return Reduction(fn=jnp.logical_or, id=jnp.zeros(shape, dtype=bool))


def min(shape: Sequence[int] = (), dtype: jax.typing.DTypeLike = float) -> Reduction:
    val = jnp.finfo(dtype).max
    return Reduction(fn=jnp.min, id=jnp.full(shape, val, dtype=dtype))


def max(shape: Sequence[int] = (), dtype: jax.typing.DTypeLike = float) -> Reduction:
    val = jnp.finfo(dtype).min
    return Reduction(fn=jnp.max, id=jnp.full(shape, val, dtype=dtype))
