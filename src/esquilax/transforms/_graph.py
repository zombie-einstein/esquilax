"""
Graph transformations

Transformations representing interactions between
nodes (and edges) on a graph.
"""
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp

from esquilax import utils


def edge_map(f: Callable) -> Callable:
    """
    Map a function over graph edges and related nodes

    Maps a function over a set of edges (with data) and
    data corresponding to the edge start and end nodes,

    .. warning::

       Edge indices and any associated data should be
       sorted using :py:meth:`esquilax.utils.sort_edges`

    Examples
    --------

    .. testsetup:: edge_map

       import esquilax
       import jax
       import jax.numpy as jnp

    .. testcode:: edge_map

       @esquilax.transforms.edge_map
       def f(k, params, start, end, edge):
           return params + start + end + edge

       k = jax.random.PRNGKey(101)
       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       starts = jnp.array([0, 1, 2])
       ends = jnp.array([0, 1, 2])

       # Call transform with edge indexes
       f(k, 2, starts, ends, edges, edge_idxs=edge_idxs)
       # [3, 5, 5]

    .. testcode:: edge_map
       :hide:

       z = f(k, 2, starts, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [3, 5, 5]

    Arguments can also be PyTrees or ``None`` if unused

    .. testcode:: edge_map

       @esquilax.transforms.edge_map
       def f(_k, _params, _start, end, edge):
           return end[0] + end[1] + edge

       k = jax.random.PRNGKey(101)
       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       f(k, None, None, ends, edges, edge_idxs=edge_idxs)
       # [2, 5, 2]

    .. testcode:: edge_map
       :hide:

       z = f(k, None, None, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [2, 5, 2]

    Parameters
    ----------
    f
        Function with the signature

        .. code-block:: python

           def f(k, params, start, end, edge, **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``k``: JAX PRNGKey
        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.
    """
    keyword_args = utils.functions.get_keyword_args(f)

    @partial(jax.jit, static_argnames=keyword_args)
    def _edge_map(
        k: chex.PRNGKey,
        params: Any,
        starts: Any,
        ends: Any,
        edges: Any,
        *,
        edge_idxs: chex.Array,
        **static_kwargs,
    ) -> Any:
        n = edge_idxs.shape[1]
        keys = jax.random.split(k, n)
        starts = jax.tree_util.tree_map(
            lambda x: x.at[edge_idxs[0]].get(),
            starts,
        )
        ends = jax.tree_util.tree_map(
            lambda x: x.at[edge_idxs[1]].get(),
            ends,
        )

        return jax.vmap(partial(f, **static_kwargs), in_axes=(0, None, 0, 0, 0))(
            keys, params, starts, ends, edges
        )

    return _edge_map


def graph_reduce(
    reduction: chex.ArrayTree, default: chex.ArrayTree, n: int = -1
) -> Callable:
    """
    Map function over graph edges and reduce results to nodes

    Maps the update function over the graph edges and corresponding
    node data. Then aggregates the results back to the start nodes
    of the edge-set. Edges are treated as directional, with the
    first node observing the state of the second.

    .. warning::

       Edge indices and any associated data should be
       sorted using :py:meth:`esquilax.utils.sort_edges`

    Examples
    --------

    .. testsetup:: graph_reduce

       import esquilax
       import jax
       import jax.numpy as jnp

    .. testcode:: graph_reduce

       @esquilax.transforms.graph_reduce(jnp.add, 0)
       def f(k, params, start, end, edge):
           return params + start + end + edge

       k = jax.random.PRNGKey(101)
       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       starts = jnp.array([0, 1, 2])
       ends = jnp.array([0, 1, 2])

       # Call transform with edge indexes
       f(k, 2, starts, ends, edges, edge_idxs=edge_idxs)
       # [8, 5, 0]

    .. testcode:: graph_reduce
       :hide:

       z =  f(k, 2, starts, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [8, 5, 0]

    Arguments can also be PyTrees, or ``None`` if unused

    .. testcode:: graph_reduce

       @esquilax.transforms.graph_reduce(jnp.add, 0, n=3)
       def f(k, _params, _start, end, edge):
           return end[0] + end[1] + edge

       k = jax.random.PRNGKey(101)
       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       f(k, None, None, ends, edges, edge_idxs=edge_idxs)
       # [7, 2, 0]

    .. testcode:: graph_reduce
       :hide:

       z = f(k, None, None, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [7, 2, 0]

    Parameters
    ----------
    reduction
        Binary monoidal reduction function
    default
        Default/identity value result value
    n
        Number of nodes, should be provided if start-node data is ``None``
    f
        Function with the signature

        .. code-block:: python

           def f(k, params, start, end, edge,  **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``k``: JAX PRNGKey
        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.
    """

    def _graph_reduce_decorator(f: Callable) -> Callable:
        chex.assert_trees_all_equal_structs(
            reduction, default
        ), "Reduction and default PyTrees should have the same structure"

        _edge_map = edge_map(f)
        keyword_args = utils.functions.get_keyword_args(f)

        @partial(jax.jit, static_argnames=keyword_args)
        def _graph_reduce(
            k: chex.PRNGKey,
            params: Any,
            starts: Any,
            ends: Any,
            edges: Any,
            *,
            edge_idxs: chex.Array,
            **kwargs,
        ) -> Any:
            n_results = utils.functions.get_size(starts) if n < 0 else n

            edge_results = _edge_map(
                k, params, starts, ends, edges, **kwargs, edge_idxs=edge_idxs
            )
            bin_counts, bins = utils.graph.index_bins(edge_idxs[0], n_results)
            bins = bins.reshape(-1)

            def reduce(r, x, d):
                x = jnp.frompyfunc(r, 2, 1).reduceat(x, bins, axis=0)[::2]
                x = jnp.where(bin_counts > 0, x, d)
                return x

            results = jax.tree_util.tree_map(reduce, reduction, edge_results, default)

            return results

        return _graph_reduce

    return _graph_reduce_decorator


def random_neighbour(default: Any, n: int = -1) -> Callable:
    """
    Apply function to random selected graph neighbours

    For each start node select a random neighbour on
    the graph (i.e. a random edge starting at that node)
    and apply the update function. Returns a
    default value if the node has no outgoing edges.

    .. warning::

       Edge indices and any associated data should be
       sorted using :py:meth:`esquilax.utils.sort_edges`

    Examples
    --------

    .. testsetup:: random_neighbour

       import esquilax
       import jax
       import jax.numpy as jnp

    .. testcode:: random_neighbour

       @esquilax.transforms.random_neighbour(0, n=3)
       def f(k, _params, _start, end, edge):
           return end[0] + end[1] + edge

       k = jax.random.PRNGKey(101)
       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       f(k, None, None, ends, edges, edge_idxs=edge_idxs)
       # [5, 2, 0]

    .. testcode:: random_neighbour
       :hide:

       z = f(k, None, None, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [5, 2, 0]

    Parameters
    ----------
    default
        Value returned when a node has no neighbours
    n
        Number of start nodes, should be provided if
        start-node data is None
    f
        Function with the signature

        .. code-block:: python

           def f(k, params, start, end, edge, **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``k``: JAX PRNGKey
        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.
    """

    def random_neighbour_decorator(f: Callable) -> Callable:
        keyword_args = utils.functions.get_keyword_args(f)

        @partial(jax.jit, static_argnames=keyword_args)
        def _random_neighbour(
            k: chex.PRNGKey,
            params: Any,
            starts: Any,
            ends: Any,
            edges: Any,
            *,
            edge_idxs: chex.Array,
            **static_kwargs,
        ):
            n_results = utils.functions.get_size(starts) if n < 0 else n
            bin_counts, bins = utils.graph.index_bins(edge_idxs[0], n_results)
            keys = jax.random.split(k, n_results)

            def sample(_k, i, a, b):
                k1, k2 = jax.random.split(_k)
                j = jax.random.randint(k1, (), a, b)
                end_idx = edge_idxs[1][j]
                start = jax.tree_util.tree_map(lambda x: x[i], starts)
                end = jax.tree_util.tree_map(lambda x: x[end_idx], ends)
                edge = jax.tree_util.tree_map(lambda x: x[j], edges)
                return partial(f, **static_kwargs)(k2, params, start, end, edge)

            def select(_k, i, count: int, bin: chex.Array) -> Any:
                return jax.lax.cond(
                    count > 0, sample, lambda *_: default, _k, i, bin[0], bin[1]
                )

            return jax.vmap(select, in_axes=(0, 0, 0, 0))(
                keys,
                jnp.arange(n_results),
                bin_counts,
                bins,
            )

        return _random_neighbour

    return random_neighbour_decorator
