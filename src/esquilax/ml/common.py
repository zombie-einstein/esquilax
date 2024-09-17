from typing import Any, Optional, Tuple, Type

import chex
import jax

from esquilax.typing import TypedPyTree


def key_tree_split(
    key: chex.PRNGKey, tree: TypedPyTree[Any], typ: Optional[Type] = None
) -> TypedPyTree[chex.PRNGKey]:
    """
    Generate random keys for PyTree leaves

    Split a random key, generating a PyTree of keys matching
    the structure of the provided tree. A type can be
    provided to define an expected type for leaves of the
    tree.

    Parameters
    ----------
    key
        JAX random key.
    tree
        Pytree to map over.
    typ
        If provided, leaves of the tree
        will be those that match the provided type.

    Returns
    -------
    esquilax.typing.TypedPyTree[chex.PRNGKey]
        PyTree with random keys as leaves and the
        same structure as the argument tree
    """
    if typ is None:
        is_leaf = None
    else:

        def is_leaf(x):
            return isinstance(x, typ)

    tree_def = jax.tree.structure(tree, is_leaf=is_leaf)
    keys = jax.random.split(key, tree_def.num_leaves)
    keys = jax.tree.unflatten(tree_def, keys)
    return keys


def transpose_tree_of_tuples(
    tree_a: TypedPyTree[Any], tree_b, n: int, typ: Optional[Type] = None
) -> Tuple:
    """
    Transpose a tree containing tuples, into a tuple of the outer tree

    Parameters
    ----------
    tree_a
        Tree to take outer structure from.
    tree_b
        Tree to transpose
    n
        Number of values to unpack
    typ
        Optional type to use as leaf check in ``tree_a``.
    Returns
    -------
    tuple
        Tuple of PyTrees
    """
    if typ is None:
        is_leaf = None
    else:

        def is_leaf(x):
            return isinstance(x, typ)

    return tuple(
        jax.tree.map(
            lambda _, x: x[i],
            tree_a,
            tree_b,
            is_leaf=is_leaf,
        )
        for i in range(n)
    )
