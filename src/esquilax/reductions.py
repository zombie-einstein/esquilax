"""
Reduction interface and common implementations
"""
from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp
from jax._src.numpy.ufuncs import ufunc

from esquilax.typing import TypedPyTree


class Reduction:
    """
    Class wrapping a reduction function and its default/identity value

    A reduction function and its associated default value. The
    reduction function and identity can also be PyTrees to
    allow reduction of multiple values in a single pass.
    """

    __slots__ = ["fn", "id"]

    def __init__(
        self, fn: TypedPyTree[Callable | ufunc], id: TypedPyTree[chex.Numeric]
    ) -> None:
        """
        Initialise a reduction

        Parameters
        ----------
        fn
            Binary reduction function.
        id
            Identity/default value of the reduction.
        """
        chex.assert_trees_all_equal_structs(
            fn, id
        ), "'fn' and 'id' PyTrees should have the same structure"
        self.fn = fn
        self.id = id


def add(shape: Sequence[int] = (), dtype: jax.typing.DTypeLike = float) -> Reduction:
    """
    Addition reduction

    Reduction that adds values, with a default value of ``0``.

    Parameters
    ----------
    shape
        Shape of reduced values.
    dtype
        Dtype of reduced values.

    Returns
    -------
    Reduction
        Initialised :py:class:`Reduction`.
    """
    return Reduction(fn=jnp.add, id=jnp.zeros(shape, dtype=dtype))


def logical_or(shape: Sequence[int] = ()) -> Reduction:
    """
    Logical or reduction

    Reduction that applies the or function to values, with a default value of ``False``.

    Parameters
    ----------
    shape
        Shape of reduced values.

    Returns
    -------
    Reduction
        Initialised :py:class:`Reduction`.
    """
    return Reduction(fn=jnp.logical_or, id=jnp.zeros(shape, dtype=bool))


def min(shape: Sequence[int] = (), dtype: jax.typing.DTypeLike = float) -> Reduction:
    """
    Minimum reduction

    Reduction that returns the minimum of values, with a default value
    of the maximum numerical values of the dtype.

    Parameters
    ----------
    shape
        Shape of reduced values.
    dtype
        Dtype of reduced values.

    Returns
    -------
    Reduction
        Initialised :py:class:`Reduction`.
    """
    val = jnp.finfo(dtype).max
    return Reduction(fn=jnp.min, id=jnp.full(shape, val, dtype=dtype))


def max(shape: Sequence[int] = (), dtype: jax.typing.DTypeLike = float) -> Reduction:
    """
    Maximum reduction

    Reduction that returns the maximum of values, with a default value
    of the minimum numerical values of the dtype.

    Parameters
    ----------
    shape
        Shape of reduced values.
    dtype
        Dtype of reduced values.

    Returns
    -------
    Reduction
        Initialised :py:class:`Reduction`.
    """
    val = jnp.finfo(dtype).min
    return Reduction(fn=jnp.max, id=jnp.full(shape, val, dtype=dtype))
