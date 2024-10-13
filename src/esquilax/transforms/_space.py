from functools import partial
from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from esquilax import utils
from esquilax.typing import Default, Reduction


def _sort_agents(
    n_bins: int, width: float, pos: chex.Array, agents: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.ArrayTree]:
    co_ords, idxs = utils.space.get_bins(pos, n_bins, width)
    sort_idxs = jnp.argsort(idxs)
    _, bins = utils.graph.index_bins(idxs, n_bins**2)
    sorted_co_ords = co_ords[sort_idxs]
    sorted_idxs = idxs[sort_idxs]
    sorted_pos = pos[sort_idxs]
    sorted_agents = jax.tree_util.tree_map(lambda y: y[sort_idxs], agents)

    return sorted_co_ords, sort_idxs, bins, sorted_idxs, sorted_pos, sorted_agents


def spatial(
    f: Callable,
    *,
    n_bins: int,
    reduction: Reduction,
    default: Default,
    include_self: bool = False,
    topology: str = "moore",
    i_range: Optional[float] = None,
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
       on a torus).

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

       def foo(_k, p, a, b):
           return p + a + b

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(3)
       k = jax.random.PRNGKey(101)

       result = esquilax.transforms.spatial(
           foo,
           n_bins=2,
           reduction=jnp.add,
           default=0,
           include_self=False,
       )(
           k, 2, a, a, pos=x
       )
       # result = [0, 5, 5]

    .. doctest:: spatial
       :hide:

       >>> result
       Array([0, 5, 5], dtype=int32)

    so in this case agent ``0`` does not have any neighbours in
    its cell, but agents ``1`` and ``2`` observe each other.

    The transform can also be used as a decoratot using
    :py:meth:`functools.partial`. Arguments and return values
    can be PyTrees or multidimensional arrays. Arguments can
    also be ``None`` if not used

    .. testcode:: spatial

       @partial(
           esquilax.transforms.spatial,
           n_bins=2,
           reduction=(jnp.add, jnp.add),
           default=(0, 0),
           include_self=False,
           topology="same-cell",
       )
       def foo(_k, p, _, b):
           return p + b[0],  p + b[1]

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(6).reshape(3, 2)
       k = jax.random.PRNGKey(101)

       foo(k, 2, None, a, pos=x)
       # ([0, 6, 4], [0, 7, 5])

    .. doctest:: spatial
       :hide:

       >>> foo(k, 2, None, a, pos=x)
       (Array([0, 6, 4], dtype=int32), Array([0, 7, 5], dtype=int32))

    You can also pass different agent types (i.e. two sets
    of agents with different positions) using the ``pos_b``
    keyword argument

    .. testcode:: spatial

       @partial(
           esquilax.transforms.spatial,
           n_bins=2,
           reduction=jnp.add,
           default=0,
           topology="moore",
       )
       def foo(_, params, a, b):
           return params + a + b

       # A consists of 3 agents, and b 2 agents
       xa = jnp.array([[0.1, 0.1], [0.1, 0.7], [0.75, 0.2]])
       xb = jnp.array([[0.7, 0.1], [0.75, 0.75]])
       vals_a = jnp.arange(3)
       vals_b = jnp.arange(1, 3)

       foo(k, 2, vals_a, vals_b, pos=xa, pos_b=xb)
       # [22, 10, 17]

    .. doctest:: spatial
       :hide:

       >>> foo(k, 2, vals_a, vals_b, pos=xa, pos_b=xb)
       Array([22, 10, 17], dtype=int32)

    Parameters
    ----------
    f
        Interaction to apply to in-proximity pairs, should
        have the signature

        .. code-block:: python

           def f(
               k: chex.PRNGKey,
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
    n_bins
        Number of bins each dimension is subdivided
        into. Assumes that each dimension contains the
        same number of cells. Each cell can only interact
        with adjacent cells, so this value also consequently
        also controls the number of interactions.
    reduction
        Binary monoidal reduction function, eg ``jax.numpy.add``.
    default
        Default/identity reduction value
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
        Optional interaction range. By default, the width
        of a cell is used as the interaction range, but this
        can be increased/decreased using ``i_range`` dependent
        on the use-case.
    """
    width = 1.0 / n_bins
    i_range = width if i_range is None else i_range
    i_range = i_range**2

    chex.assert_trees_all_equal_structs(
        reduction, default
    ), "Reduction and default PyTrees should have the same structure"

    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)

    @partial(jax.jit, static_argnames=keyword_args)
    def _spatial(
        key: chex.PRNGKey,
        params: Any,
        agents_a: Any,
        agents_b: Any,
        *,
        pos: chex.Array,
        pos_b: Optional[chex.Array] = None,
        **static_kwargs,
    ) -> Any:
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

        def cell(k: chex.PRNGKey, i: int, bin_range: chex.Array) -> Any:
            agent_a = jax.tree_util.tree_map(lambda y: y[i], sorted_agents_a)
            pos_a = sorted_pos_a[i]

            def interact(j: int, _k: chex.PRNGKey, _r: Any) -> Tuple[chex.PRNGKey, Any]:
                _k, fk = jax.random.split(_k, 2)
                agent_b = jax.tree_util.tree_map(lambda z: z[j], sorted_agents_b)
                r = partial(f, **static_kwargs)(fk, params, agent_a, agent_b)
                r = jax.tree_util.tree_map(
                    lambda red, a, b: red(a, b), reduction, _r, r
                )
                return _k, r

            def inner(
                j: int, carry: Tuple[chex.PRNGKey, Any]
            ) -> Tuple[chex.PRNGKey, Any]:
                _k, _r = carry
                pos_b = sorted_pos_b[j]
                d = utils.space.shortest_distance(pos_a, pos_b, 1.0, norm=False)
                return jax.lax.cond(
                    d < i_range, interact, lambda _, _x, _z: (_x, _z), j, _k, _r
                )

            if (not same_types) or include_self:
                _, _results = jax.lax.fori_loop(
                    bin_range[0], bin_range[1], inner, (k, default)
                )
            else:
                k, _results = jax.lax.fori_loop(
                    bin_range[0],
                    jnp.minimum(i, bin_range[1]),
                    inner,
                    (k, default),
                )
                _, _results = jax.lax.fori_loop(
                    jnp.maximum(i + 1, bin_range[0]),
                    bin_range[1],
                    inner,
                    (k, _results),
                )

            return _results

        def agent_reduce(k: chex.PRNGKey, i: int, co_ords: chex.Array):
            nbs = utils.space.neighbour_indices(co_ords, offsets, n_bins)
            nb_bins = bins_b[nbs]
            _keys = jax.random.split(k, nbs.shape[0])
            _results = jax.vmap(cell, in_axes=(0, None, 0))(_keys, i, nb_bins)

            def red(a, _, c):
                return jnp.frompyfunc(c, 2, 1).reduce(a)

            return jax.tree_util.tree_map(red, _results, default, reduction)

        n_agents = pos.shape[0]
        keys = jax.random.split(key, n_agents)
        results = jax.vmap(agent_reduce, in_axes=(0, 0, 0))(
            keys, jnp.arange(n_agents), co_ords_a
        )

        inv_sort = jnp.argsort(sort_idxs_a)
        results = jax.tree_util.tree_map(lambda y: y[inv_sort], results)

        return results

    return _spatial


def nearest_neighbour(
    f: Callable,
    *,
    n_bins: int,
    default: Default,
    topology: str = "moore",
    i_range: Optional[float] = None,
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
       on a torus).

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

       def foo(_k, p, a, b):
           return p + a + b

       x = jnp.array([[0.4, 0.4], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(3)
       k = jax.random.PRNGKey(101)

       result = esquilax.transforms.nearest_neighbour(
           foo,
           n_bins=2,
           default=-1,
           topology="moore"
       )(
           k, 2, a, a, pos=x
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
           n_bins=2,
           default=(-1, -2),
           topology="moore",
       )
       def foo(_k, p, _, b):
           return p + b[0],  p + b[1]

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(6).reshape(3, 2)
       k = jax.random.PRNGKey(101)

       foo(k, 2, None, a, pos=x)
       # ([-1, 6, 4], [-2, 7, 5])

    .. doctest:: spatial
       :hide:

       >>> foo(k, 2, None, a, pos=x)
       (Array([-1,  6,  4], dtype=int32), Array([-2,  7,  5], dtype=int32))

    You can also pass different agent types (i.e. two sets
    of agents with different positions) using the ``pos_b``
    keyword argument

    .. testcode:: spatial

       @partial(
           esquilax.transforms.nearest_neighbour,
           n_bins=2,
           default=-1,
           topology="moore",
       )
       def foo(_, params, a, b):
           return params + a + b

       # A consists of a single agents, and b 2 agents
       xa = jnp.array([[0.1, 0.1]])
       xb = jnp.array([[0.2, 0.2], [0.75, 0.75]])
       vals_a = jnp.array([1])
       vals_b = jnp.array([2, 12])

       foo(k, 2, vals_a, vals_b, pos=xa, pos_b=xb)
       # [5]

    .. doctest:: spatial
       :hide:

       >>> foo(k, 2, vals_a, vals_b, pos=xa, pos_b=xb)
       Array([5], dtype=int32)

    Parameters
    ----------
    n_bins
        Number of bins each dimension is subdivided
        into. Assumes that each dimension contains the
        same number of cells. Each cell can only interact
        with adjacent cells, so this value also consequently
        also controls the number of interactions.
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
        Optional interaction range. By default, the width
        of a cell is used as the interaction range, but this
        can be increased/decreased using ``i_range`` dependent
        on the use-case.
    f
        Interaction to apply to in-proximity pairs, should
        have the signature

        .. code-block:: python

           def f(
               k: chex.PRNGKey,
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
    """
    width = 1.0 / n_bins
    i_range = width if i_range is None else i_range
    i_range = i_range**2

    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)

    @partial(jax.jit, static_argnames=keyword_args)
    def _nearest_neighbour(
        key: chex.PRNGKey,
        params: Any,
        agents_a: Any,
        agents_b: Any,
        *,
        pos: chex.Array,
        pos_b: Optional[chex.Array] = None,
        **static_kwargs,
    ) -> Any:
        same_types = pos_b is None

        (
            co_ords_a,
            sort_idxs_a,
            bins_a,
            sorted_idxs_a,
            sorted_pos_a,
            _,
        ) = _sort_agents(n_bins, width, pos, agents_a)

        if same_types:
            bins_b = bins_a
            sorted_pos_b = sorted_pos_a
        else:
            _, _, bins_b, _, sorted_pos_b, _ = _sort_agents(
                n_bins, width, pos_b, agents_b
            )

        def cell(i: int, bin_range: chex.Array) -> Tuple[int, float]:
            pos_a = sorted_pos_a[i]

            def inner(j: int, carry: Tuple[int, float]) -> Tuple[int, float]:
                _best_idx, _best_d = carry
                _pos_b = sorted_pos_b[j]
                _d = utils.space.shortest_distance(pos_a, _pos_b, 1.0, norm=False)
                return jax.lax.cond(
                    jnp.logical_and(_d < i_range, _d < _best_d),
                    lambda: (j, _d),
                    lambda: (_best_idx, _best_d),
                )

            if not same_types:
                best_idx, best_d = jax.lax.fori_loop(
                    bin_range[0], bin_range[1], inner, (-1, 1.0)
                )
            else:
                best_idx, best_d = jax.lax.fori_loop(
                    bin_range[0],
                    jnp.minimum(i, bin_range[1]),
                    inner,
                    (-1, 1.0),
                )
                best_idx, best_d = jax.lax.fori_loop(
                    jnp.maximum(i + 1, bin_range[0]),
                    bin_range[1],
                    inner,
                    (best_idx, best_d),
                )

            return best_idx, best_d

        def agent_reduce(i: int, co_ords: chex.Array) -> int:
            nbs = utils.space.neighbour_indices(co_ords, offsets, n_bins)
            nb_bins = bins_b[nbs]
            best_idx, best_d = jax.vmap(cell, in_axes=(None, 0))(i, nb_bins)
            min_idx = jnp.argmin(best_d)
            min_idx = best_idx[min_idx]
            return min_idx

        n_agents = pos.shape[0]
        keys = jax.random.split(key, n_agents)
        nearest_idxs = jax.vmap(agent_reduce, in_axes=(0, 0))(
            jnp.arange(n_agents), co_ords_a
        )
        inv_sort = jnp.argsort(sort_idxs_a)
        nearest_idxs = nearest_idxs[inv_sort]

        def apply(k, a, idx_b):
            b = jax.tree.map(lambda x: x.at[idx_b].get(), agents_b)
            return partial(f, **static_kwargs)(k, params, a, b)

        def check(k, a, idx_b):
            return jax.lax.cond(
                idx_b < 0,
                lambda *_: default,
                apply,
                k,
                a,
                idx_b,
            )

        results = jax.vmap(check)(keys, agents_a, nearest_idxs)

        return results

    return _nearest_neighbour
