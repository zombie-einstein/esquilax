from functools import partial
from math import prod
from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from esquilax import utils
from esquilax.reductions import Reduction


def _sort_agents(
    dims: Tuple[int, int], co_ords: chex.Array, agents: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.ArrayTree]:
    idxs = co_ords[:, 0] * dims[1] + co_ords[:, 1]
    sort_idxs = jnp.argsort(idxs)
    _, bins = utils.graph.index_bins(idxs, prod(dims))
    sorted_co_ords = co_ords[sort_idxs]
    sorted_idxs = idxs[sort_idxs]
    sorted_agents = jax.tree_util.tree_map(lambda y: y[sort_idxs], agents)

    return sorted_co_ords, sort_idxs, bins, sorted_idxs, sorted_agents


def _check_arguments(
    idxs: chex.Array,
    idxs_b: Optional[chex.Array],
    agents_a: Optional[chex.ArrayTree],
    agents_b: Optional[chex.ArrayTree],
) -> None:
    assert (
        idxs.ndim == 2 and idxs.shape[1] == 2
    ), f"idxs argument should be an array of 2d indices, got shape {idxs.shape}"

    if idxs_b is not None:
        assert idxs_b.ndim == 2 and idxs_b.shape[1] == 2, (
            "idxs_b argument should be an array of "
            f"2d coordinates, got shape {idxs_b.shape}"
        )
        n_b = idxs_b.shape[:1]
    else:
        n_b = idxs.shape[:1]

    if agents_a is not None:
        chex.assert_tree_has_only_ndarrays(agents_a)
        chex.assert_tree_shape_prefix(agents_a, idxs.shape[:1])

    if agents_b is not None:
        chex.assert_tree_has_only_ndarrays(agents_b)
        chex.assert_tree_shape_prefix(agents_b, n_b)


def grid(
    f: Callable,
    *,
    reduction: Reduction,
    dims: Tuple[int, int],
    include_self: bool = False,
    topology: str = "moore",
) -> Callable:
    """
    Apply a function to agents based on a grid neighbourhood

    Applies an interaction function between agents based
    on their proximity on a 2-dimensional grid.

    .. warning::

       This implementation assumes the grid is wrapped at its
       boundaries.

    Examples
    --------

    This example subdivides the space into 2 cells along
    each dimension (i.e. there are 4 cells total),
    then sums contributions from neighbours, excluding the
    active agent

    .. testsetup:: grid

       import esquilax
       import jax
       import jax.numpy as jnp
       from functools import partial

    .. testcode:: grid

       def foo(p, a, b):
           return p + a + b

       x = jnp.array([[0, 0], [0, 1], [2, 2]])
       a = jnp.arange(3)

       result = esquilax.transforms.grid(
           foo,
           reduction=esquilax.reductions.add(dtype=int),
           dims=(4, 4),
           include_self=False,
       )(
           2, a, a, co_ords=x
       )
       # result = [3, 3, 0]

    .. doctest:: grid
       :hide:

       >>> result
       Array([3, 3, 0], dtype=int32)

    in this case the first two agents are next
    to each other on the grid, but the third
    has no direct neighbours.

    The transform can also be used as a decorator using
    :py:meth:`functools.partial`. Arguments and return values
    can be PyTrees or multidimensional arrays. Arguments can
    also be ``None`` if not used

    .. testcode:: grid

       tuple_reduce = esquilax.reductions.Reduction(
           fn=(jnp.add, jnp.add), id=(0, 0),
       )

       @partial(
           esquilax.transforms.grid,
           reduction=tuple_reduce,
           dims=(4, 4),
           include_self=False,
       )
       def foo(p, _, b):
           return p + b[0], p + b[1]

       x = jnp.array([[0, 0], [0, 1], [2, 2]])
       a = jnp.arange(6).reshape(3, 2)

       result = foo(2, None, a, co_ords=x)
       # result = ([4, 2, 0], [5, 3, 0])

    .. doctest:: grid

       >>> result
       (Array([4, 2, 0], dtype=int32), Array([5, 3, 0], dtype=int32))

    You can also pass different agent types (i.e. two sets
    of agents with different positions) using the ``co_ords_b``
    keyword argument

    .. testcode:: grid

       def foo(p, a, b):
           return p + a + b

       xa = jnp.array([[0, 0], [2, 2]])
       xb = jnp.array([[0, 0], [2, 2]])
       a = jnp.arange(2)
       b = 2 + a

       result = esquilax.transforms.grid(
           foo,
           reduction=esquilax.reductions.add(dtype=int),
           dims=(4, 4),
           include_self=False,
       )(
           2, a, b, co_ords=xa, co_ords_b=xb
       )
       # result = [4, 6]

    .. doctest:: grid
       :hide:

       >>> result
       Array([4, 6], dtype=int32)

    .. testcode:: grid

       def foo(p, a, b, *, key):
           return jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = esquilax.transforms.grid(
           foo,
           reduction=esquilax.reductions.add(dtype=int),
           dims=(4, 4),
           include_self=False,
       )(
           None, None, None, co_ords=x, key=k
       )

    Parameters
    ----------
    f
        Interaction to apply to in-proximity pairs, should
        have the signature

        .. code-block:: python

           def f(
               params: Any,
               a: Any,
               b: Any,
               **static_kwargs,
           ):
               ...
               return x

        where

        - ``params``: Parameters broadcast over all interactions
        - ``a``: Start agent in the interaction
        - ``b``: End agent in the interaction
        - ``**static_kwargs``: Any arguments required at compile
          time by JAX can be passed as keyword arguments.

        Random keys can be passed to the wrapped function by
        including the ``key`` keyword argument.
    reduction
        Binary monoidal reduction.
    dims
        Number of cells along each dimension
    include_self
        if ``True`` each agent will include itself in the
        gathered values.
    topology
        Topology of cells, default ``"moore"``. Since cells
        interact with their neighbours, topologies with
        fewer neighbours can increase performance at the
        cost of fidelity. Should be one of ``"same-cell"``,
        ``"von-neumann"`` or ``"moore"``.
    """
    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _grid(
        params: Any,
        agents_a: Any,
        agents_b: Any,
        *,
        co_ords: chex.Array,
        co_ords_b: Optional[chex.Array] = None,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> Any:
        utils.functions.check_key(has_key, key)
        _check_arguments(co_ords, co_ords_b, agents_a, agents_b)
        same_types = co_ords_b is None

        (
            co_ords_a,
            sort_idxs_a,
            bins_a,
            sorted_idxs_a,
            sorted_agents_a,
        ) = _sort_agents(dims, co_ords, agents_a)

        if same_types:
            bins_b = bins_a
            sorted_agents_b = jax.tree_util.tree_map(lambda y: y[sort_idxs_a], agents_b)
        else:
            _, _, bins_b, _, sorted_agents_b = _sort_agents(dims, co_ords_b, agents_b)

        def cell(k: Optional[chex.PRNGKey], i: int, bin_range: chex.Array) -> Any:
            agent_a = jax.tree_util.tree_map(lambda y: y[i], sorted_agents_a)

            def interact(
                j: int, carry: Tuple[Optional[chex.PRNGKey], Any]
            ) -> Tuple[Optional[chex.PRNGKey], Any]:
                _k, _r = carry
                agent_b = jax.tree_util.tree_map(lambda z: z[j], sorted_agents_b)

                if has_key:
                    _k, fk = jax.random.split(_k, 2)
                    r = f(params, agent_a, agent_b, key=fk, **static_kwargs)
                else:
                    r = f(params, agent_a, agent_b, **static_kwargs)

                r = jax.tree_util.tree_map(
                    lambda red, a, b: red(a, b), reduction.fn, _r, r
                )
                return _k, r

            if (not same_types) or include_self:
                _, _results = jax.lax.fori_loop(
                    bin_range[0], bin_range[1], interact, (k, reduction.id)
                )
            else:
                k, _results = jax.lax.fori_loop(
                    bin_range[0],
                    jnp.minimum(i, bin_range[1]),
                    interact,
                    (k, reduction.id),
                )
                _, _results = jax.lax.fori_loop(
                    jnp.maximum(i + 1, bin_range[0]),
                    bin_range[1],
                    interact,
                    (k, _results),
                )

            return _results

        def agent_reduce(
            k: Optional[chex.PRNGKey], i: chex.Numeric, _co_ords: chex.Array
        ):
            nbs = utils.space.neighbour_indices(_co_ords, offsets, dims)
            nb_bins = bins_b[nbs]

            if has_key:
                _keys = jax.random.split(k, nbs.shape[0])
                _results = jax.vmap(cell, in_axes=(0, None, 0))(_keys, i, nb_bins)
            else:
                _results = jax.vmap(cell, in_axes=(None, None, 0))(None, i, nb_bins)

            def red(a, _, c):
                return jnp.frompyfunc(c, 2, 1).reduce(a)

            return jax.tree_util.tree_map(red, _results, reduction.id, reduction.fn)

        n_agents = co_ords.shape[0]

        if has_key:
            keys = jax.random.split(key, n_agents)
            results = jax.vmap(agent_reduce, in_axes=(0, 0, 0))(
                keys, jnp.arange(n_agents), co_ords_a
            )
        else:
            results = jax.vmap(agent_reduce, in_axes=(None, 0, 0))(
                None, jnp.arange(n_agents), co_ords_a
            )

        inv_sort = jnp.argsort(sort_idxs_a)
        results = jax.tree_util.tree_map(lambda y: y[inv_sort], results)

        return results

    return _grid


def grid_local(
    f: Callable,
    *,
    topology: str = "moore",
) -> Callable:
    """
    Apply a function to an agent local cells

    Applies an interaction function between an agent and
    its surrounding cells. The transform slices the
    cell saround the agent, and passes them
    to the mapped function.

    .. warning::

       This implementation assumes the grid is wrapped at its
       boundaries.

    Examples
    --------

    This example subdivides the space into 3 cells along
    each dimension (i.e. there are 9 cells total),
    then applies the function to an agent at ``[1, 1]``

    .. testsetup:: grid

       import esquilax
       import jax
       import jax.numpy as jnp
       from functools import partial

    .. testcode:: grid

       def foo(p, _, grid):
           # grid here is an array of values
           return p + grid

       x = jnp.array([[1, 1]])
       grid = jnp.arange(9).reshape(3, 3)

       result = esquilax.transforms.grid_local(
           foo
       )(
           2, None, grid, co_ords=x
       )
       # result = [[ 2,  3,  4,  5,  6,  7,  8,  9, 10]]

    .. doctest:: grid
       :hide:

       >>> result
       Array([[ 2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=int32)

    The transform can also be used as a decorator using
    :py:meth:`functools.partial`. Arguments and return values
    can be PyTrees or multidimensional arrays. Arguments can
    also be ``None`` if not used

    .. testcode:: grid

       @partial(
           esquilax.transforms.grid_local,
           topology="von-neumann",
       )
       def foo(p, _, g):
           return p[0] + g[0], p[1] + g[1]

       x = jnp.array([[1, 1]])
       grid = jnp.arange(9).reshape(3, 3)
       grids = (grid, grid + 1)

       result = foo((1, 2), None, grids, co_ords=x)
       # result = ([[5, 6, 8, 4, 2]], [[ 7,  8, 10,  6,  4]])

    .. doctest:: grid
       :hide:

       >>> result[0]
       Array([[5, 6, 8, 4, 2]], dtype=int32)
       >>> result[1]
       Array([[ 7,  8, 10,  6,  4]], dtype=int32)

    Random keys can be passed to the wrapped function by
    including the ``key`` keyword argument

    .. testcode:: grid

       def foo(_p, _a, grid, *, key):
           # grid here is an array of values
           return grid + jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = esquilax.transforms.grid_local(
           foo
       )(
           2, None, grid, co_ords=x, key=k
       )

    Parameters
    ----------
    f
        Interaction to apply to in-proximity pairs, should
        have the signature

        .. code-block:: python

           def f(
               params: Any,
               a: Any,
               grid: Any,
               **static_kwargs,
           ):
               ...
               return x

        where

        - ``params``: Parameters broadcast over all interactions
        - ``a``: Agent in the interaction
        - ``grid``: Pytree of arrays gathered from local cells.
        - ``**static_kwargs``: Any arguments required at compile
          time by JAX can be passed as keyword arguments.

        JAX random keys can be passed to the wrapped argument by
        including the ``key`` keyword argument.
    topology
        Topology of the local neighbourhood, default ``"moore"``.
        The cells passed to the interaction function are gathered
        according to this topology, e.g. a moore neighbourhood
        will pass an array of size ``(9,)`` to the function.
    """
    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _grid_local(
        params: Any,
        agents: chex.ArrayTree,
        grids: chex.ArrayTree,
        key: Optional[chex.PRNGKey] = None,
        *,
        co_ords: chex.Array,
        **static_kwargs,
    ) -> chex.ArrayTree:
        assert grids is not None, "'grids' argument should not be None"
        utils.functions.check_key(has_key, key)
        dims = jax.tree.flatten(grids)[0][0].shape[:2]
        chex.assert_tree_shape_prefix(grids, dims)
        dims = jnp.array(dims)
        _check_arguments(co_ords, None, agents, None)

        def inner(k: Optional[chex.PRNGKey], _co_ords: chex.Array, agent):
            nbs = (_co_ords[jnp.newaxis] + offsets) % dims
            grid_vals = jax.tree.map(lambda x: x[nbs[:, 0], nbs[:, 1]], grids)
            if has_key:
                result = f(params, agent, grid_vals, key=k, **static_kwargs)
            else:
                result = f(params, agent, grid_vals, **static_kwargs)
            return result

        n_agents = co_ords.shape[0]

        if has_key:
            keys = jax.random.split(key, n_agents)
            results = jax.vmap(inner, in_axes=(0, 0, 0))(keys, co_ords, agents)
        else:
            results = jax.vmap(inner, in_axes=(None, 0, 0))(None, co_ords, agents)

        return results

    return _grid_local
