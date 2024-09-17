"""
Graph/network utility functions
"""
from typing import Any, Tuple, Union

import chex
import jax
import jax.numpy as jnp


def sort_edges(x: chex.Array, *args) -> Union[chex.Array, Tuple[chex.Array, Any]]:
    """
    Sort graph edge indices and data for use in graph transformations

    Graph transformations expect graph edge indices to be sorted
    by the start index. Consequently, any data associated with
    edges should also be sorted in the same manner.

    Examples
    --------

    .. testsetup:: sort_edges

       import esquilax
       import jax

       k = jax.random.PRNGKey(101)
       n_nodes = 20
       n_edges = 10

    .. testcode:: sort_edges

       # Generate indices representing random graph edges
       edges = jax.random.choice(k, n_nodes, shape=(2, n_edges))
       sorted_edges = esquilax.utils.sort_edges(edges)

       # Include edge-weights
       edge_weights = jax.random.uniform(k, n_edges)
       sorted_edges, sorted_weights = esquilax.utils.sort_edges(
           edges, edge_weights
       )

    Parameters
    ----------
    x
        Array of edge indices in the shape :code:`[2, n_edges]`
    *args
        Optional arguments representing edge data. These arguments
        can also be PyTrees which will then sort all members
        of the tree.

    Returns
    -------
    tuple
        Sorted indices and sorted optional data
    """
    idxs = jnp.lexsort((x[1], x[0]))
    x = jnp.stack([x[0][idxs], x[1][idxs]])

    if args:
        args = jax.tree_util.tree_map(lambda a: a[idxs], args)
        return x, *args
    else:
        return x


def index_bins(indices: chex.Array, length: int) -> Tuple[chex.Array, chex.Array]:
    """
    Count indices frequencies and generate bins of contiguous indices

    This effectively generates a histogram of integer values, and
    corresponding start and end indices of the binned values if they
    are in sort order.

    Examples
    --------

    .. testsetup:: index_bins

       import esquilax
       import jax.numpy as jnp

       from esquilax.utils.graph import index_bins

    .. doctest:: index_bins

       >>> indices = jnp.array([0, 0, 1, 3, 3])
       >>> counts, bins = index_bins(indices, 5)
       >>> counts
       Array([2, 1, 0, 2, 0], dtype=int32)
       >>> bins
       Array([[0, 2],
              [2, 3],
              [3, 3],
              [3, 5],
              [5, 5]], dtype=int32)

    Parameters
    ----------
    indices
        1D array of integer indices
    length
        Number of indices to count, e.g. the
        number of bins to generate and count.

    Returns
    -------
    tuple[jax.numpy.ndarray, jax.numpy.ndarray]
        Count/frequency of each index, and bins boundaries
    """
    bin_counts = jnp.bincount(indices, length=length)
    cum_counts = jnp.cumsum(bin_counts)

    return bin_counts, jnp.column_stack([cum_counts - bin_counts, cum_counts])
