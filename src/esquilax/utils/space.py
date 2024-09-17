"""
Functions used in spatial transforms
"""
from itertools import product
from typing import Union

import chex
import jax
import jax.numpy as jnp


def get_bins(x: chex.Array, n_cells: int, width: float) -> chex.Array:
    """
    Assign co-ordinates to a grid-cell

    .. warning::

       This implementation currently assumes that the width
       of cells and number of cells is the same in both
       dimensions.

    Parameters
    ----------
    x
        2d array of co-ordinates, in shape ``[n, 2]``.
    n_cells
        Number of cells along dimensions.
    width
        Width of space.

    Returns
    -------
    jax.numpy.ndarray
        Array of cell indices for each position
    """
    y = jnp.floor_divide(x, width).astype(jnp.int32)
    i = y[:, 0] * n_cells + y[:, 1]
    return i


def get_cell_neighbours(n_bins: int, topology: str) -> chex.Array:
    """
    Get indices of adjacent cells for each cell on a grid

    Returns array where each row represents a cell
    and indices of cells adjacent to it given
    a desired topology.

    .. warning::

       This implementation currently assumes that the
       space is wrapped at the edges (i.e. on a torus).

    Parameters
    ----------
    n_bins
        Number of bins
    topology
        Topology of the neighbouring cells, one of:

        - ``same-cell``: Only consider cells in isolation
        - ``von-neumann``: Use a `Von-Neumann neighbourhood \
        <https://en.wikipedia.org/wiki/Von_Neumann_neighborhood>`_
        - ``moore``: Use the `Moore neighbourhood \
        <https://en.wikipedia.org/wiki/Moore_neighborhood>`_

    Returns
    -------
    jax.numpy.ndarray
        2d array where each row contains a cells index
        and indices of its neighbours.
    """
    n = n_bins**2

    if topology == "same-cell":
        offsets = jnp.array([[0, 0]])
    elif topology == "von-neumann":
        offsets = jnp.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
    elif topology == "moore":
        offsets = list(product([-1, 0, 1], [-1, 0, 1]))
        offsets = jnp.array(offsets)
    else:
        raise ValueError(
            "Topology should be one of 'same-cell', 'von-neumann' or 'moore'"
        )

    grid = jnp.arange(n).reshape(n_bins, n_bins)
    id_x, id_y = jnp.meshgrid(jnp.arange(n_bins), jnp.arange(n_bins))

    def slice(off):
        off_x, off_y = id_x + off[1], id_y + off[0]
        off_x, off_y = off_x % n_bins, off_y % n_bins
        x = grid[off_y, off_x].reshape(-1)
        return x

    idxs = jax.vmap(slice, out_axes=1)(offsets)

    return idxs


def shortest_vector(a: chex.Array, b: chex.Array, length: float = 1.0) -> chex.Array:
    """
    Get the shortest vector between points on a torus

    Parameters
    ----------
    a
        Co-ordinate vector.
    b
        Co-ordinate vector.
    length
        Length of the space. Default ``1.0``.

    Returns
    -------
    jax.numpy.ndarray
        Array representing the shortest vector between the points.
    """
    x = b - a
    x_ = jnp.sign(x) * (jnp.abs(x) - length)
    return jnp.where(jnp.abs(x) < jnp.abs(x_), x, x_)


def shortest_distance(
    a: Union[float, chex.Array],
    b: Union[float, chex.Array],
    length: float = 1.0,
    norm: bool = True,
) -> Union[float, chex.Array]:
    """
    Get the shortest distance between points on a torus

    Parameters
    ----------
    a
        Co-ordinate vector
    b
        Co-ordinate vector
    length
        Length of the space. Default is ``1.0``.
    norm
        If ``True`` the distance will be normalised
        otherwise the square of the distance will be
        returned. Default ``True``.

    Returns
    -------
    float
        Distance between the points
    """
    x = jnp.abs(a - b)
    d = jnp.where(x > 0.5 * length, length - x, x)
    d = jnp.sum(jnp.square(d), axis=-1)
    if norm:
        d = jnp.sqrt(d)
    return d
