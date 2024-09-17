from functools import partial
from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from esquilax import utils


def spatial(
    n_bins: int,
    reduction: chex.ArrayTree,
    default: chex.ArrayTree,
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

    .. testcode:: spatial

       @esquilax.transforms.spatial(
           2, jnp.add, 0, include_self=False, topology="same-cell"
       )
       def foo(_k, p, a, b):
           return p + a + b

       x = jnp.array([[0.1, 0.1], [0.6, 0.6], [0.7, 0.7]])
       a = jnp.arange(3)
       k = jax.random.PRNGKey(101)

       foo(k, 2, a, a, pos=x)
       # [0, 5, 5]

    .. doctest:: spatial
       :hide:

       >>> foo(k, 2, a, a, pos=x).tolist()
       [0, 5, 5]

    so in this case agent ``0`` does not have any neighbours in
    its cell, but agents ``1`` and ``2`` observe each other.

    Arguments and return values can be PyTrees or multidimensional
    arrays. Arguments can also be ``None`` if not used

    .. testcode:: spatial

       @esquilax.transforms.spatial(
           2,
           (jnp.add, jnp.add),
           (0, 0),
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

    Parameters
    ----------
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
    w = 1.0 / n_bins
    i_range = w if i_range is None else i_range

    chex.assert_trees_all_equal_structs(
        reduction, default
    ), "Reduction and default PyTrees should have the same structure"

    def spatial_decorator(f: Callable) -> Callable:
        t = utils.space.get_cell_neighbours(n_bins, topology)
        keyword_args = utils.functions.get_keyword_args(f)

        @partial(jax.jit, static_argnames=keyword_args)
        def _spatial(
            key: chex.PRNGKey,
            params: Any,
            agents_a: Any,
            agents_b: Any,
            *,
            pos: chex.Array,
            **static_kwargs,
        ) -> Any:
            idxs = utils.space.get_bins(pos, n_bins, w)
            sort_idxs = jnp.argsort(idxs)

            _, bins = utils.graph.index_bins(idxs, n_bins**2)

            sorted_idxs = idxs[sort_idxs]
            sorted_x = pos[sort_idxs]
            sorted_agents_a = jax.tree_util.tree_map(lambda y: y[sort_idxs], agents_a)
            sorted_agents_b = jax.tree_util.tree_map(lambda y: y[sort_idxs], agents_b)

            def cell(k: chex.PRNGKey, i: int, bin_range: chex.Array) -> Any:
                agent_a = jax.tree_util.tree_map(lambda y: y[i], sorted_agents_a)
                pos_a = sorted_x[i]

                def interact(
                    j: int, _k: chex.PRNGKey, _r: Any
                ) -> Tuple[chex.PRNGKey, Any]:
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
                    pos_b = sorted_x[j]
                    d = utils.space.shortest_distance(pos_a, pos_b, 1.0, norm=False)
                    return jax.lax.cond(
                        d < i_range, interact, lambda _, _x, _z: (_x, _z), j, _k, _r
                    )

                if include_self:
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

            def agent_reduce(k: chex.PRNGKey, i: int, bin_idx: int):
                nbs = t[bin_idx]
                nb_bins = bins[nbs]
                _keys = jax.random.split(k, nbs.shape[0])
                _results = jax.vmap(cell, in_axes=(0, None, 0))(_keys, i, nb_bins)

                def red(a, _, c):
                    return jnp.frompyfunc(c, 2, 1).reduce(a)

                return jax.tree_util.tree_map(red, _results, default, reduction)

            n_agents = pos.shape[0]
            keys = jax.random.split(key, n_agents)
            results = jax.vmap(agent_reduce, in_axes=(0, 0, 0))(
                keys, jnp.arange(n_agents), sorted_idxs
            )

            inv_sort = jnp.argsort(sort_idxs)
            results = jax.tree_util.tree_map(lambda y: y[inv_sort], results)

            return results

        return _spatial

    return spatial_decorator
