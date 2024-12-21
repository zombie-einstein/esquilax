from functools import partial
from math import floor, isclose, prod
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp

from esquilax import utils
from esquilax.reductions import Reduction
from esquilax.typing import Default


def _sort_agents(
    n_bins: Tuple[int, int], width: float, pos: chex.Array, agents: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.ArrayTree]:
    co_ords, idxs = utils.space.get_bins(pos, n_bins, width)
    sort_idxs = jnp.argsort(idxs)
    _, bins = utils.graph.index_bins(idxs, prod(n_bins))
    sorted_co_ords = co_ords[sort_idxs]
    sorted_idxs = idxs[sort_idxs]
    sorted_pos = pos[sort_idxs]
    sorted_agents = jax.tree_util.tree_map(lambda y: y[sort_idxs], agents)

    return sorted_co_ords, sort_idxs, bins, sorted_idxs, sorted_pos, sorted_agents


def _process_parameters(
    i_range: float,
    dims: Union[float, Sequence[float]],
    n_bins: Optional[int | Sequence[int]],
) -> Tuple[Tuple[int, int], float, chex.Array]:
    if isinstance(dims, Sequence):
        assert (
            len(dims) == 2
        ), f"2 spatial dimensions should be provided got {len(dims)}"

        if n_bins is not None:
            assert isinstance(
                n_bins, Sequence
            ), f"n_bins should be a sequence if dims is a sequence, got {type(n_bins)}"
            assert (
                len(n_bins) == 2
            ), f"Number of bins should be provided for 2 dimensions, got {len(n_bins)}"
            assert (
                n_bins[0] > 0 and n_bins[1] > 0
            ), f"n_bins should all be greater than 0, got {n_bins}"
            w1, w2 = dims[0] / n_bins[0], dims[1] / n_bins[1]
            assert w1 == w2, (
                "Dimensions of cells should be equal in "
                f"both dimensions got {w1} and {w2}"
            )
            n_bins = (n_bins[0], n_bins[1])
        else:
            assert (
                i_range is not None
            ), "If n_bins is not provided, i_range should be provided"
            n0 = dims[0] / i_range
            n1 = dims[1] / i_range
            assert isclose(round(n0), n0) and isclose(
                round(n1), n1
            ), "Dimensions should be a multiple of i_range"
            n_bins = (round(n0), round(n1))

        width = dims[0] / n_bins[0]
        dims = jnp.array(dims)

    else:
        if n_bins is not None:
            assert isinstance(
                n_bins, int
            ), "n_bins should be an integer value if dims is a float"
            assert n_bins > 0, f"n_bins should be greater than 0, got {n_bins}"
            n_bins = (n_bins, n_bins)
        else:
            n_bins = floor(dims / i_range)
            n_bins = (n_bins, n_bins)

        width = dims / n_bins[0]
        dims = jnp.array([dims, dims])

    return n_bins, width, dims


def _check_arguments(
    pos: chex.Array,
    pos_b: Optional[chex.Array],
    agent_a: chex.ArrayTree,
    agent_b: chex.ArrayTree,
) -> None:
    assert (
        pos.ndim == 2 and pos.shape[1] == 2
    ), f"pos argument should be an array of 2d coordinates, got shape {pos.shape}"

    if pos_b is not None:
        assert pos.ndim == 2 and pos.shape[1] == 2, (
            "pos_b argument should be an array of "
            f"2d coordinates, got shape {pos.shape}"
        )
        n_b = pos_b.shape[:1]
    else:
        n_b = pos.shape[:1]

    if agent_a is not None:
        chex.assert_tree_has_only_ndarrays(agent_a)
        chex.assert_tree_shape_prefix(agent_a, pos.shape[:1])

    if agent_b is not None:
        chex.assert_tree_has_only_ndarrays(agent_b)
        chex.assert_tree_shape_prefix(agent_b, n_b)


def spatial(
    f: Callable,
    *,
    reduction: Reduction,
    include_self: bool = False,
    topology: str = "moore",
    n_bins: Optional[int | Sequence[int]] = None,
    i_range: Optional[float] = None,
    dims: Union[float, Sequence[float]] = 1.0,
) -> Callable:
    """
    Apply a function between agents based on spatial proximity

    This transformation efficiently checks
    for spatial proximity between agents before applying
    the interaction function to pairs of agents
    below a threshold range of each other.

    .. warning::

       This implementation currently assumes a 2-dimensional
       space with continues boundary conditions (i.e. wrapped
       on a torus). The shape/dimensions of the space
       can be controlled with the `dims` parameter, by default
       it is a unit square region.

    .. note::

       Performance is dependent on the proximity threshold
       and density of agents. Worse case it will perform
       :math:`N^2` checks if all agents are in the same
       neighbourhood.

    Examples
    --------

    This example subdivides the space into 2 cells along
    each dimension (i.e. there are 4 cells total),
    then sums contributions from neighbours, excluding the
    active agent

    .. testsetup:: spatial

       import esquilax
       import jax
       import jax.numpy as jnp
       from functools import partial

    .. testcode:: spatial

       def foo(p, a, b):
           return p + a + b

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(3)

       result = esquilax.transforms.spatial(
           foo,
           i_range=0.5,
           reduction=esquilax.reductions.add(dtype=int),
           include_self=False,
       )(
           2, a, a, pos=x
       )
       # result = [0, 5, 5]

    .. doctest:: spatial
       :hide:

       >>> result
       Array([0, 5, 5], dtype=int32)

    so in this case agent ``0`` does not have any neighbours in
    its cell, but agents ``1`` and ``2`` observe each other.

    The transform can also be used as a decorator using
    :py:meth:`functools.partial`. Arguments and return values
    can be PyTrees or multidimensional arrays. Arguments can
    also be ``None`` if not used

    .. testcode:: spatial

       tuple_reduce = esquilax.reductions.Reduction(
           fn=(jnp.add, jnp.add), id=(0, 0),
       )

       @partial(
           esquilax.transforms.spatial,
           i_range=0.5,
           reduction=tuple_reduce,
           include_self=False,
           topology="same-cell",
       )
       def foo(p, _, b):
           return p + b[0],  p + b[1]

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(6).reshape(3, 2)

       result = foo(2, None, a, pos=x)
       # result = ([0, 6, 4], [0, 7, 5])

    .. doctest:: spatial
       :hide:

       >>> result
       (Array([0, 6, 4], dtype=int32), Array([0, 7, 5], dtype=int32))

    You can also pass different agent types (i.e. two sets
    of agents with different positions) using the ``pos_b``
    keyword argument

    .. testcode:: spatial

       @partial(
           esquilax.transforms.spatial,
           i_range=0.5,
           reduction=esquilax.reductions.add(dtype=int),
           topology="moore",
       )
       def foo(params, a, b):
           return params + a + b

       # A consists of 3 agents, and b 2 agents
       xa = jnp.array([[0.1, 0.1], [0.1, 0.7], [0.75, 0.2]])
       xb = jnp.array([[0.7, 0.1], [0.75, 0.75]])
       vals_a = jnp.arange(3)
       vals_b = jnp.arange(1, 3)

       result = foo(2, vals_a, vals_b, pos=xa, pos_b=xb)
       # result = [22, 10, 17]

    .. doctest:: spatial
       :hide:

       >>> result
       Array([22, 10, 17], dtype=int32)

    Random keys can be passed to the wrapped function
    by including the ``key`` keyword argument

    .. testcode:: spatial

       def foo(_p, _a, _b, *, key):
           return jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = esquilax.transforms.spatial(
           foo,
           i_range=0.5,
           reduction=esquilax.reductions.add(dtype=int),
           include_self=False,
       )(
           None, None, None, pos=x, key=k
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

        JAX random keys can be passed to the function by including
        the ``key`` keyword argument.
    reduction
        Binary monoidal reduction function.
    include_self
        if ``True`` each agent will include itself in the
        gathered values.
    topology
        Topology of cells, default ``"moore"``. Since cells
        interact with their neighbours, topologies with
        fewer neighbours can increase performance at the
        cost of fidelity. Should be one of ``"same-cell"``,
        ``"von-neumann"`` or ``"moore"``.
    i_range
        Range at which agents interact. Can be ommited, in which
        case the width of a cell is used as the interaction range
        (derived from ``n_bins``), but this can be increased/decreased
        using ``i_range`` dependent on the use-case.
    n_bins
        Optional number of bins each dimension is subdivided
        into. Assumes that each dimension contains the
        same number of cells. Each cell can only interact
        with adjacent cells, so this value also consequently
        also controls the number of interactions. If not provided
        the minimum number of bins if derived from ``i_range``. For a square
        space ``n_bins`` can be a single intiger, or a pair of integers
        for the number of bins along each cell. The number of cells for
        each dimension should result in square cells.
    dims
        Dimensions of the space, either a float edge length for a
        square space, or a pait (tuple or list) of dimension.
        Default value is a square space of size 1.0.
    """

    n_bins, width, dims = _process_parameters(i_range, dims, n_bins)
    i_range = width if i_range is None else i_range
    i_range = i_range**2
    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _spatial(
        params: Any,
        agents_a: Any,
        agents_b: Any,
        *,
        pos: chex.Array,
        pos_b: Optional[chex.Array] = None,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> Any:
        _check_arguments(pos, pos_b, agents_a, agents_b)
        utils.functions.check_key(has_key, key)

        same_types = pos_b is None

        (
            co_ords_a,
            sort_idxs_a,
            bins_a,
            sorted_idxs_a,
            sorted_pos_a,
            sorted_agents_a,
        ) = _sort_agents(n_bins, width, pos, agents_a)

        if same_types:
            bins_b = bins_a
            sorted_pos_b = sorted_pos_a
            sorted_agents_b = jax.tree_util.tree_map(lambda y: y[sort_idxs_a], agents_b)
        else:
            _, _, bins_b, _, sorted_pos_b, sorted_agents_b = _sort_agents(
                n_bins, width, pos_b, agents_b
            )

        def cell(k: Optional[chex.PRNGKey], i: int, bin_range: chex.Array) -> Any:
            agent_a = jax.tree_util.tree_map(lambda y: y[i], sorted_agents_a)
            pos_a = sorted_pos_a[i]

            def interact(
                j: int, _k: Optional[chex.PRNGKey], _r: Any
            ) -> Tuple[Optional[chex.PRNGKey], Any]:
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

            def inner(
                j: int, carry: Tuple[Optional[chex.PRNGKey], Any]
            ) -> Tuple[Optional[chex.PRNGKey], Any]:
                _k, _r = carry
                _pos_b = sorted_pos_b[j]
                d = utils.space.shortest_distance(
                    pos_a, _pos_b, length=dims, norm=False
                )
                return jax.lax.cond(
                    d <= i_range, interact, lambda _, _x, _z: (_x, _z), j, _k, _r
                )

            if (not same_types) or include_self:
                _, _results = jax.lax.fori_loop(
                    bin_range[0], bin_range[1], inner, (k, reduction.id)
                )
            else:
                k, _results = jax.lax.fori_loop(
                    bin_range[0],
                    jnp.minimum(i, bin_range[1]),
                    inner,
                    (k, reduction.id),
                )
                _, _results = jax.lax.fori_loop(
                    jnp.maximum(i + 1, bin_range[0]),
                    bin_range[1],
                    inner,
                    (k, _results),
                )

            return _results

        def agent_reduce(
            k: Optional[chex.PRNGKey], i: chex.Numeric, co_ords: chex.Array
        ) -> chex.ArrayTree:
            nbs = utils.space.neighbour_indices(co_ords, offsets, n_bins)
            nb_bins = bins_b[nbs]

            if has_key:
                _keys = jax.random.split(k, nbs.shape[0])
                _results = jax.vmap(cell, in_axes=(0, None, 0))(_keys, i, nb_bins)
            else:
                _results = jax.vmap(cell, in_axes=(None, None, 0))(None, i, nb_bins)

            def red(a, _, c):
                return jnp.frompyfunc(c, 2, 1).reduce(a)

            return jax.tree_util.tree_map(red, _results, reduction.id, reduction.fn)

        n_agents = pos.shape[0]

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

    return _spatial


def nearest_neighbour(
    f: Callable,
    *,
    default: Default,
    topology: str = "moore",
    n_bins: Optional[int | Sequence[int]] = None,
    i_range: Optional[float] = None,
    dims: Union[float, Sequence[float]] = 1.0,
) -> Callable:
    """
    Apply a function between an agent and its closest neighbour

    This transformation locates the nearest neighbour
    to each agent and then applies the update function
    to this pair. If no nearest neighbour then the
    default value is returned.

    .. warning::

       This implementation currently assumes a 2-dimensional
       space with continues boundary conditions (i.e. wrapped
       on a torus). The shape/dimensions of the space
       can be controlled with the `dims` parameter, by default
       it is a unit square region.

    .. note::

       Performance is dependent on the proximity threshold
       and density of agents. Worse case it will perform
       :math:`N^2` checks if all agents are in the same
       neighbourhood.

    Examples
    --------

    This example subdivides the space into 2 cells along
    each dimension (i.e. there are 4 cells total) then
    locate closest neighbours

    .. testsetup:: spatial

       import esquilax
       import jax
       import jax.numpy as jnp
       from functools import partial

    .. testcode:: spatial

       def foo(p, a, b):
           return p + a + b

       x = jnp.array([[0.4, 0.4], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(3)

       result = esquilax.transforms.nearest_neighbour(
           foo,
           i_range=0.5,
           default=-1,
           topology="moore"
       )(
           2, a, a, pos=x
       )
       # result = [3, 5, 5]

    .. doctest:: spatial
       :hide:

       >>> result
       Array([3, 5, 5], dtype=int32)

    so in this case agents ``1`` and ``2`` are nearest neighbours
    and ``0`` is closest to ``1``.

    The transform can be used as a decorator using
    :py:meth:`functools.partial`. Arguments and return values
    can be PyTrees or multidimensional arrays. Arguments can also
    be ``None`` if not used

    .. testcode:: spatial

       @partial(
           esquilax.transforms.nearest_neighbour,
           i_range=0.5,
           default=(-1, -2),
           topology="moore",
       )
       def foo(p, _, b):
           return p + b[0],  p + b[1]

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(6).reshape(3, 2)

       result = foo(2, None, a, pos=x)
       # result = ([-1, 6, 4], [-2, 7, 5])

    .. doctest:: spatial
       :hide:

       >>> result
       (Array([-1,  6,  4], dtype=int32), Array([-2,  7,  5], dtype=int32))

    You can also pass different agent types (i.e. two sets
    of agents with different positions) using the ``pos_b``
    keyword argument

    .. testcode:: spatial

       @partial(
           esquilax.transforms.nearest_neighbour,
           i_range=0.5,
           default=-1,
           topology="moore",
       )
       def foo(params, a, b):
           return params + a + b

       # A consists of a single agents, and b 2 agents
       xa = jnp.array([[0.1, 0.1]])
       xb = jnp.array([[0.2, 0.2], [0.75, 0.75]])
       vals_a = jnp.array([1])
       vals_b = jnp.array([2, 12])

       result = foo(2, vals_a, vals_b, pos=xa, pos_b=xb)
       # result = [5]

    .. doctest:: spatial
       :hide:

       >>> result
       Array([5], dtype=int32)

    Random keys can be passed to the wrapped function by
    using the ``key`` keyword argument

    .. testcode:: spatial

       def foo(p, a, b, *, key):
           return jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = esquilax.transforms.nearest_neighbour(
           foo,
           i_range=0.5,
           default=-1,
           topology="moore"
       )(
           None, None, None, pos=x, key=k
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

        - ``k``: Is a JAX random key
        - ``params``: Parameters broadcast over all interactions
        - ``a``: Start agent in the interaction
        - ``b``: End agent in the interaction
        - ``**static_kwargs``: Any arguments required at compile
          time by JAX can be passed as keyword arguments.

        Random keys can be passed to the function by including
        the ``key`` keyword argument.
    default
        Default value(s) returned if no-neighbours are in
        range of an agent.
    topology
        Topology of cells, default ``"moore"``. Since cells
        interact with their neighbours, topologies with
        fewer neighbours can increase performance at the
        cost of fidelity. Should be one of ``"same-cell"``,
        ``"von-neumann"`` or ``"moore"``.
    i_range
        Range at which agents interact. Can be ommited, in which
        case the width of a cell is used as the interaction range
        (derived from ``n_bins``), but this can be increased/decreased
        using ``i_range`` dependent on the use-case.
    n_bins
        Optional number of bins each dimension is subdivided
        into. Assumes that each dimension contains the
        same number of cells. Each cell can only interact
        with adjacent cells, so this value also consequently
        also controls the number of interactions. If not provided
        the minimum number of bins if derived from ``i_range``.
    dims
        Dimensions of the space, either a float edge length for a
        square space, or a pait (tuple or list) of dimension.
        Default value is a square space of size 1.0.
    """
    n_bins, width, dims = _process_parameters(i_range, dims, n_bins)
    i_range = width if i_range is None else i_range
    i_range = i_range**2

    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _nearest_neighbour(
        params: Any,
        agents_a: Any,
        agents_b: Any,
        *,
        pos: chex.Array,
        pos_b: Optional[chex.Array] = None,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> Any:
        _check_arguments(pos, pos_b, agents_a, agents_b)
        utils.functions.check_key(has_key, key)

        same_types = pos_b is None

        (
            co_ords_a,
            sort_idxs_a,
            bins_a,
            sorted_idxs_a,
            sorted_pos_a,
            sorted_agents_a,
        ) = _sort_agents(n_bins, width, pos, agents_a)

        if same_types:
            bins_b = bins_a
            sorted_pos_b = sorted_pos_a
            sorted_agents_b = jax.tree_util.tree_map(lambda y: y[sort_idxs_a], agents_b)
        else:
            _, _, bins_b, _, sorted_pos_b, sorted_agents_b = _sort_agents(
                n_bins, width, pos_b, agents_b
            )

        def cell(i: int, bin_range: chex.Array) -> Tuple[int, float]:
            pos_a = sorted_pos_a[i]

            def inner(j: int, carry: Tuple[int, float]) -> Tuple[int, float]:
                _best_idx, _best_d = carry
                _pos_b = sorted_pos_b[j]
                _d = utils.space.shortest_distance(
                    pos_a, _pos_b, length=dims, norm=False
                )
                return jax.lax.cond(
                    jnp.logical_and(_d <= i_range, _d < _best_d),
                    lambda: (j, _d),
                    lambda: (_best_idx, _best_d),
                )

            if not same_types:
                best_idx, best_d = jax.lax.fori_loop(
                    bin_range[0], bin_range[1], inner, (-1, jnp.inf)
                )
            else:
                best_idx, best_d = jax.lax.fori_loop(
                    bin_range[0],
                    jnp.minimum(i, bin_range[1]),
                    inner,
                    (-1, jnp.inf),
                )
                best_idx, best_d = jax.lax.fori_loop(
                    jnp.maximum(i + 1, bin_range[0]),
                    bin_range[1],
                    inner,
                    (best_idx, best_d),
                )

            return best_idx, best_d

        def agent_reduce(i: chex.Numeric, co_ords: chex.Array) -> chex.Numeric:
            nbs = utils.space.neighbour_indices(co_ords, offsets, n_bins)
            nb_bins = bins_b[nbs]
            best_idx, best_d = jax.vmap(cell, in_axes=(None, 0))(i, nb_bins)
            min_idx = jnp.argmin(best_d)
            min_idx = best_idx[min_idx]
            return min_idx

        n_agents = pos.shape[0]
        nearest_idxs = jax.vmap(agent_reduce, in_axes=(0, 0))(
            jnp.arange(n_agents), co_ords_a
        )
        inv_sort = jnp.argsort(sort_idxs_a)
        nearest_idxs = nearest_idxs[inv_sort]

        def apply(
            k: Optional[chex.PRNGKey], a: chex.ArrayTree, idx_b: chex.Numeric
        ) -> chex.ArrayTree:
            b = jax.tree.map(lambda x: x.at[idx_b].get(), sorted_agents_b)
            if has_key:
                result = f(params, a, b, key=k, **static_kwargs)
            else:
                result = f(params, a, b, **static_kwargs)
            return result

        def check(
            k: Optional[chex.PRNGKey], a: chex.ArrayTree, idx_b: chex.Numeric
        ) -> chex.ArrayTree:
            return jax.lax.cond(
                idx_b < 0,
                lambda *_: default,
                apply,
                k,
                a,
                idx_b,
            )

        if has_key:
            keys = jax.random.split(key, n_agents)
            results = jax.vmap(check)(keys, agents_a, nearest_idxs)
        else:
            results = jax.vmap(check, in_axes=(None, 0, 0))(
                None, agents_a, nearest_idxs
            )

        return results

    return _nearest_neighbour
