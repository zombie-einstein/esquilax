from functools import partial
from math import prod
from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from esquilax import utils
from esquilax.typing import Default, Reduction


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
    agents_a: chex.ArrayTree,
    agents_b: chex.ArrayTree,
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
    default: Default,
    dims: Tuple[int, int],
    include_self: bool = False,
    topology: str = "moore",
) -> Callable:
    offsets = utils.space.get_neighbours_offsets(topology)
    keyword_args = utils.functions.get_keyword_args(f)

    @partial(jax.jit, static_argnames=keyword_args)
    def _grid(
        key: chex.PRNGKey,
        params: Any,
        agents_a: Any,
        agents_b: Any,
        *,
        co_ords: chex.Array,
        co_ords_b: Optional[chex.Array] = None,
        **static_kwargs,
    ) -> Any:
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

        def cell(k: chex.PRNGKey, i: int, bin_range: chex.Array) -> Any:
            agent_a = jax.tree_util.tree_map(lambda y: y[i], sorted_agents_a)

            def interact(
                j: int, carry: Tuple[chex.PRNGKey, Any]
            ) -> Tuple[chex.PRNGKey, Any]:
                _k, _r = carry
                _k, fk = jax.random.split(_k, 2)
                agent_b = jax.tree_util.tree_map(lambda z: z[j], sorted_agents_b)
                r = partial(f, **static_kwargs)(fk, params, agent_a, agent_b)
                r = jax.tree_util.tree_map(
                    lambda red, a, b: red(a, b), reduction, _r, r
                )
                return _k, r

            if (not same_types) or include_self:
                _, _results = jax.lax.fori_loop(
                    bin_range[0], bin_range[1], interact, (k, default)
                )
            else:
                k, _results = jax.lax.fori_loop(
                    bin_range[0],
                    jnp.minimum(i, bin_range[1]),
                    interact,
                    (k, default),
                )
                _, _results = jax.lax.fori_loop(
                    jnp.maximum(i + 1, bin_range[0]),
                    bin_range[1],
                    interact,
                    (k, _results),
                )

            return _results

        def agent_reduce(k: chex.PRNGKey, i: chex.Numeric, _co_ords: chex.Array):
            nbs = utils.space.neighbour_indices(_co_ords, offsets, dims)
            nb_bins = bins_b[nbs]
            _keys = jax.random.split(k, nbs.shape[0])
            _results = jax.vmap(cell, in_axes=(0, None, 0))(_keys, i, nb_bins)

            def red(a, _, c):
                return jnp.frompyfunc(c, 2, 1).reduce(a)

            return jax.tree_util.tree_map(red, _results, default, reduction)

        n_agents = co_ords.shape[0]
        keys = jax.random.split(key, n_agents)
        results = jax.vmap(agent_reduce, in_axes=(0, 0, 0))(
            keys, jnp.arange(n_agents), co_ords_a
        )
        inv_sort = jnp.argsort(sort_idxs_a)
        results = jax.tree_util.tree_map(lambda y: y[inv_sort], results)

        return results

    return _grid
