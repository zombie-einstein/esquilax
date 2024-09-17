"""
Function and data utilities
"""
import inspect
from typing import Callable, List

import chex
import jax


def get_size(tree: chex.ArrayTree) -> int:
    """
    Get the length of arrays in a PyTree

    Also checks that arrays are all the same length

    Examples
    --------

    .. testsetup:: get_size

       import jax.numpy as jnp
       from esquilax.utils.functions import get_size

    .. doctest:: get_size

       >>> x = (jnp.arange(10), jnp.arange(10))
       >>> get_size(x)
       10

    Parameters
    ----------
    tree
        PyTree of arrays

    Raises
    ------
    AssertionError
        Raised if arrays do not have the same length.

    Returns
    -------
    int
        Length of arrays in the tree
    """
    lens = jax.tree_util.tree_map(lambda x: x.shape[0], tree)
    lens = jax.tree_util.tree_flatten(lens)[0]
    assert all([lens[0] == n for n in lens[1:]])
    return lens[0]


def get_keyword_args(f: Callable) -> List[str]:
    """
    Get keyword-only argument names from a function signature

    Examples
    --------

    .. testsetup:: get_keyword_args

       import jax.numpy as jnp
       from esquilax.utils.functions import get_keyword_args

    .. doctest:: get_keyword_args

       >>> def foo(a, b, *, c, d): pass
       >>> get_keyword_args(foo)
       ['c', 'd']

    Parameters
    ----------
    f
        Function to inspect

    Returns
    -------
    list[str]
        List of keyword-only argument names.
    """
    sig = inspect.signature(f)
    key_word_args = [
        k for k, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    return key_word_args
