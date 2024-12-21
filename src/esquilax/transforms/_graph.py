"""
Graph transformations

Transformations representing interactions between
nodes (and edges) on a graph.
"""
from functools import partial
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp

from esquilax import utils
from esquilax.reductions import Reduction
from esquilax.typing import Default


def _check_edges(edges: chex.ArrayTree, edge_idxs: chex.Array):
    assert edge_idxs.ndim == 2 and edge_idxs.shape[0] == 2, (
        "edge_idxs should be a 2d array of node " f"index pairs, got {edge_idxs.shape}"
    )
    chex.assert_tree_shape_prefix(edges, edge_idxs.shape[1:])


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

       def f(params, start, end, edge):
           return params + start + end + edge

       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       starts = jnp.array([0, 1, 2])
       ends = jnp.array([0, 1, 2])

       # Call transform with edge indexes
       result = esquilax.transforms.edge_map(
           f
       )(
           2, starts, ends, edges, edge_idxs=edge_idxs
       )
       # result = [3, 5, 5]

    .. testcode:: edge_map
       :hide:

       assert result.tolist() == [3, 5, 5]

    The transform can also be used as a decorator. Arguments
    can also be PyTrees or ``None`` if unused

    .. testcode:: edge_map

       @esquilax.transforms.edge_map
       def f(_params, _start, end, edge):
           return end[0] + end[1] + edge

       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       f(None, None, ends, edges, edge_idxs=edge_idxs)
       # [2, 5, 2]

    .. testcode:: edge_map
       :hide:

       z = f(None, None, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [2, 5, 2]

    JAX random keys can be passed to the wrapped function
    by including the ``key`` keyword argumnt

    .. testcode:: edge_map

       @esquilax.transforms.edge_map
       def f(_params, _start, _end, _edge, *, key):
           # Sample a random integer for each edge
           return jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = f(None, None, None, None, edge_idxs=edge_idxs, key=k)

    Parameters
    ----------
    f
        Function with the signature

        .. code-block:: python

           def f(params, start, end, edge, **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.

        The keyword argument ``key`` can be included to pass
        a random key to the mapped function.
    """
    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _edge_map(
        params: Any,
        starts: chex.ArrayTree,
        ends: chex.ArrayTree,
        edges: chex.ArrayTree,
        *,
        edge_idxs: chex.Array,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> chex.ArrayTree:
        utils.functions.check_key(has_key, key)
        chex.assert_tree_has_only_ndarrays(starts)
        chex.assert_tree_has_only_ndarrays(ends)
        chex.assert_tree_has_only_ndarrays(edge_idxs)
        _check_edges(edges, edge_idxs)

        n = edge_idxs.shape[1]
        starts = jax.tree_util.tree_map(
            lambda x: x.at[edge_idxs[0]].get(),
            starts,
        )
        ends = jax.tree_util.tree_map(
            lambda x: x.at[edge_idxs[1]].get(),
            ends,
        )

        if has_key:
            keys = jax.random.split(key, n)
            results = jax.vmap(
                lambda k, pr, st, en, ed: f(pr, st, en, ed, key=k, **static_kwargs),
                in_axes=(0, None, 0, 0, 0),
            )(keys, params, starts, ends, edges)
        else:
            results = jax.vmap(partial(f, **static_kwargs), in_axes=(None, 0, 0, 0))(
                params, starts, ends, edges
            )

        return results

    return _edge_map


def graph_reduce(f: Callable, *, reduction: Reduction, n: int = -1) -> Callable:
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
       from functools import partial

    .. testcode:: graph_reduce

       def f(params, start, end, edge):
           return params + start + end + edge

       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       starts = jnp.array([0, 1, 2])
       ends = jnp.array([0, 1, 2])

       # Call transform with edge indexes
       result = esquilax.transforms.graph_reduce(
           f,
           reduction=esquilax.reductions.add(dtype=int)
       )(
           2, starts, ends, edges, edge_idxs=edge_idxs
       )
       # result = [8, 5, 0]

    .. testcode:: graph_reduce
       :hide:

       assert result.tolist() == [8, 5, 0]

    Using :py:meth`functools.partial` the transform
    can be used as a decorator. Arguments can also be
    PyTrees, or ``None`` if unused

    .. testcode:: graph_reduce

       @partial(
           esquilax.transforms.graph_reduce,
           reduction=esquilax.reductions.add(dtype=int),
           n=3,
        )
       def f(_params, _start, end, edge):
           return end[0] + end[1] + edge

       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       f(None, None, ends, edges, edge_idxs=edge_idxs)
       # [7, 2, 0]

    .. testcode:: graph_reduce
       :hide:

       z = f(None, None, ends, edges, edge_idxs=edge_idxs)
       assert z.tolist() == [7, 2, 0]

    Random keys can be passed to the wrapped function
    by including the ``key`` keyword argument

    .. testcode:: graph_reduce

       @partial(
           esquilax.transforms.graph_reduce,
           reduction=esquilax.reductions.add(dtype=int),
           n=3,
        )
       def f(_params, _start, _end, _edge, *, key):
           return jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = f(
           None, None, None, None, edge_idxs=edge_idxs, key=k
       )

    Parameters
    ----------
    f
        Function with the signature

        .. code-block:: python

           def f(params, start, end, edge,  **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.

        The ``key`` keyword argument can be included
        to pass a random key to the mapped function.
    reduction
        Binary monoidal reduction instance
    n
        Number of nodes, should be provided if start-node data is ``None``
    """

    _edge_map = edge_map(f)
    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _graph_reduce(
        params: Any,
        starts: chex.ArrayTree,
        ends: chex.ArrayTree,
        edges: chex.ArrayTree,
        *,
        edge_idxs: chex.Array,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> chex.ArrayTree:
        if starts is None:
            assert n > 0, "If starts is not provided, n should be provided"

        n_results = utils.functions.get_size(starts) if n < 0 else n

        edge_results = _edge_map(
            params,
            starts,
            ends,
            edges,
            **static_kwargs,
            edge_idxs=edge_idxs,
            key=key,
        )
        bin_counts, bins = utils.graph.index_bins(edge_idxs[0], n_results)
        bins = bins.reshape(-1)

        def reduce(r, x, d):
            x = jnp.frompyfunc(r, 2, 1).reduceat(x, bins, axis=0)[::2]
            x = jnp.where(bin_counts > 0, x, d)
            return x

        results = jax.tree_util.tree_map(
            reduce, reduction.fn, edge_results, reduction.id
        )

        return results

    return _graph_reduce


def random_edge(
    f: Callable,
    *,
    default: Default,
    n: int = -1,
) -> Callable:
    """
    Apply function to randomly selected edge from each node

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
       from functools import partial

    .. testcode:: random_neighbour

       @partial(
           esquilax.transforms.random_edge,
           default=0,
           n=3,
       )
       def f(_params, _start, end, edge):
           return end[0] + end[1] + edge

       k = jax.random.PRNGKey(101)
       edge_idxs = jnp.array([[0, 0, 1], [1, 2, 0]])
       edges = jnp.array([0, 1, 2])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       f(None, None, ends, edges, edge_idxs=edge_idxs, key=k)
       # [5, 2, 0]

    .. testcode:: random_neighbour
       :hide:

       z = f(None, None, ends, edges, edge_idxs=edge_idxs, key=k)
       assert z.tolist() == [5, 2, 0]

    Parameters
    ----------
    f
        Function with the signature

        .. code-block:: python

           def f(params, start, end, edge, **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.

        A random key can be passed to the wrapped function by
        including the ``key`` keyword argument.
    default
        Value returned when a node has no neighbours
    n
        Number of start nodes, should be provided if
        start-node data is None
    """

    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _random_edge(
        params: Any,
        starts: chex.ArrayTree,
        ends: chex.ArrayTree,
        edges: chex.ArrayTree,
        *,
        edge_idxs: chex.Array,
        key: chex.PRNGKey,
        **static_kwargs,
    ) -> chex.ArrayTree:
        chex.assert_tree_has_only_ndarrays(starts)
        chex.assert_tree_has_only_ndarrays(ends)
        chex.assert_tree_has_only_ndarrays(edges)
        _check_edges(edges, edge_idxs)
        if starts is None:
            assert n > 0, "If starts is not provided, n should be provided"
        n_results = utils.functions.get_size(starts) if n < 0 else n
        bin_counts, bins = utils.graph.index_bins(edge_idxs[0], n_results)

        def sample(
            _k: Optional[chex.PRNGKey], i: chex.Numeric, a: chex.Array, b: chex.Array
        ):
            k1, k2 = jax.random.split(_k)
            j = jax.random.randint(k1, (), a, b)
            end_idx = edge_idxs[1][j]
            start = jax.tree_util.tree_map(lambda x: x[i], starts)
            end = jax.tree_util.tree_map(lambda x: x[end_idx], ends)
            edge = jax.tree_util.tree_map(lambda x: x[j], edges)
            if has_key:
                result = f(params, start, end, edge, key=k2, **static_kwargs)
            else:
                result = f(params, start, end, edge, **static_kwargs)
            return result

        def select(_k, i, count: int, b: chex.Array) -> Any:
            return jax.lax.cond(
                count > 0, sample, lambda *_: default, _k, i, b[0], b[1]
            )

        keys = jax.random.split(key, n_results)
        results = jax.vmap(select, in_axes=(0, 0, 0, 0))(
            keys,
            jnp.arange(n_results),
            bin_counts,
            bins,
        )
        return results

    return _random_edge


def highest_weight(f: Callable, *, default: Default, n: int = -1) -> Callable:
    """
    Map function over graph edges with the highest weights

    For each node selects the outgoing edge with the highest
    weight, and then applies the update function on these edges
    and connected nodes. If a node has no neighbours the default value
    is returned instead.

    .. warning::

       Edge indices and any associated data should be
       sorted using :py:meth:`esquilax.utils.sort_edges`

    Examples
    --------

    .. testsetup:: graph_reduce

       import esquilax
       import jax
       import jax.numpy as jnp
       from functools import partial

    .. testcode:: graph_reduce

       def f(params, start, end, edge):
           return params + start + end + edge

       edge_idxs = jnp.array([[0, 0, 2, 2], [1, 2, 0, 1]])
       weights = jnp.array([0.1, 0.5, 0.3, 0.2])
       edges = jnp.array([0, 1, 2, 3])
       starts = jnp.array([0, 1, 2])
       ends = jnp.array([0, 1, 2])

       # Call transform with edge indexes
       result = esquilax.transforms.highest_weight(
           f, default=-1
       )(
           2,
           starts,
           ends,
           edges,
           edge_idxs=edge_idxs,
           weights=weights,
       )
       # result = [5, -1, 6]

    .. testcode:: graph_reduce
       :hide:

       assert result.tolist() == [5, -1, 6], f"{result}"

    Using :py:meth:`functools.partial` the transform can
    be used as a decorator. Arguments can also be PyTrees,
    or ``None`` if unused

    .. testcode:: graph_reduce

       @partial(
           esquilax.transforms.highest_weight,
           default=-1,
           n=3,
       )
       def f(_params, _start, end, edge):
           return end[0] + end[1] + edge

       edge_idxs = jnp.array([[0, 0, 2, 2], [1, 2, 0, 1]])
       weights = jnp.array([0.1, 0.5, 0.3, 0.2])
       edges = jnp.array([0, 1, 2, 3])
       ends = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))

       # Call transform with edge indexes
       result = f(
           None,
           None,
           ends,
           edges,
           edge_idxs=edge_idxs,
           weights=weights,
       )
       # [5, -1, 2]

    .. testcode:: graph_reduce
       :hide:

       assert result.tolist() == [5, -1, 2], f"{z}"

    Random keys can be passed to the mapped argument by
    including the ``key`` keyword argument

    .. testcode:: graph_reduce

       def f(params, start, end, edge, *, key):
           return jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = esquilax.transforms.highest_weight(
           f, default=-1, n=3,
       )(
           None,
           None,
           None,
           None,
           edge_idxs=edge_idxs,
           weights=weights,
           key=k,
       )

    Parameters
    ----------
    f
        Function with the signature

        .. code-block:: python

           def f(k, params, start, end, edge,  **static_kwargs):
               ...
               return x

        where the arguments are:

        - ``params``: Parameters (shared over the map)
        - ``start``: Start node data
        - ``end``: End node data
        - ``edge``: Edge data
        - ``**static_kwargs``: Any values required at compile-time
          by JAX can be passed as keyword arguments.

        Random keys can be passed to wrapped argument by including
        the ``key`` keyword argument.
    default
        Default/identity result value
    n
        Number of nodes, should be provided if start-node data is ``None``
    """

    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _highest_weight(
        params: Any,
        starts: chex.ArrayTree,
        ends: chex.ArrayTree,
        edges: chex.ArrayTree,
        *,
        edge_idxs: chex.Array,
        weights: chex.Array,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> Any:
        if starts is None:
            assert n > 0, "If starts is not provided, n should be provided"

        utils.functions.check_key(has_key, key)
        chex.assert_tree_has_only_ndarrays(starts)
        chex.assert_tree_has_only_ndarrays(ends)
        chex.assert_tree_has_only_ndarrays(edges)
        _check_edges(edges, edge_idxs)

        n_results = utils.functions.get_size(starts) if n < 0 else n
        start_nodes = edge_idxs[0]
        bin_counts, bins = utils.graph.index_bins(start_nodes, n_results)

        s0 = jnp.argsort(weights, descending=True)
        s1 = jnp.argsort(start_nodes[s0])
        s2 = s0[s1]

        max_idxs = s2[bins[:, 0]]
        max_idxs = jnp.where(bin_counts > 0, max_idxs, -1)

        def apply(k: Optional[chex.PRNGKey], i: chex.Numeric):
            e = edge_idxs[:, i]
            a = jax.tree.map(lambda x: x.at[e[0]].get(), starts)
            b = jax.tree.map(lambda x: x.at[e[1]].get(), ends)
            c = jax.tree.map(lambda x: x.at[i].get(), edges)
            if has_key:
                result = f(params, a, b, c, key=k, **static_kwargs)
            else:
                result = f(params, a, b, c, **static_kwargs)
            return result

        def check(k: Optional[chex.PRNGKey], i: chex.Numeric):
            return jax.lax.cond(i < 0, lambda *_: default, apply, k, i)

        if has_key:
            keys = jax.random.split(key, num=n_results)
            results = jax.vmap(check)(keys, max_idxs)
        else:
            results = jax.vmap(check, in_axes=(None, 0))(None, max_idxs)

        return results

    return _highest_weight
