from typing import Collection, Optional, Type, TypeVar

import chex
import jax

T = TypeVar("T")
TypedPyTree = T | Collection[T]


def key_tree_split(
    key: chex.PRNGKey, tree: TypedPyTree, typ: Optional[Type] = None
) -> TypedPyTree[chex.PRNGKey]:
    """
    Generate random keys for PyTree leaves

    Split a random key, generating a PyTree of keys matching
    the structure of the provided tree. A type can be
    provided to define an expected type for leaves of the
    tree.

    Parameters
    ----------
    key: jax.random.PRNGKey
        JAX random key.
    tree: TypedPyTree
        Pytree to map over.
    typ: type, optional
        If provided, leaves of the tree
        will be those that match the provided type.

    Returns
    -------
    TypedPyTree[jax.random.PRNGKey]

    """
    if typ is None:
        is_leaf = None
    else:

        def is_leaf(x):
            return isinstance(x, typ)

    treedef = jax.tree.structure(tree, is_leaf=is_leaf)
    keys = jax.random.split(key, treedef.num_leaves)
    keys = jax.tree.unflatten(treedef, keys)
    return keys
