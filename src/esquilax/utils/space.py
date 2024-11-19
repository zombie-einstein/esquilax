"""
Functions used in spatial transforms
"""
from itertools import product
from typing import Tuple, Union

import chex
import jax.numpy as jnp


def get_bins(
    x: chex.Array, n_cells: Tuple[int, int], width: float
) -> Tuple[chex.Array, chex.Array]:
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
        Number of cells along each dimensions.
    width
        Width of a cell.

    Returns
    -------
    jax.numpy.ndarray
        Array of cell indices for each position
    """
    y = jnp.floor_divide(x, width).astype(jnp.int32)
    i = y[:, 0] * n_cells[1] + y[:, 1]
    return y, i


def neighbour_indices(
    x: chex.Array, offsets: chex.Array, n_bins: Tuple[int, int]
) -> chex.Array:
    """
    Apply offsets to co-ordinates to get neighbouring bin indices

    Parameters
    ----------
    x
        Cell co-ordinates.
    offsets
        Index offsets.
    n_bins
        Number of bins across dimensions.

    Returns
    -------
    chex.Array
        Bin indices of neighbouring cells.
    """
    offset_x = x + offsets
    return (offset_x[:, 0] % n_bins[0]) * n_bins[1] + (offset_x[:, 1] % n_bins[1])


def get_neighbours_offsets(topology: str) -> chex.Array:
    """
    Get offset co-ords of adjacent cells for a given topology

    Returns array containing co-ordinate offsets of neighbours
    for a desired topology.

    Parameters
    ----------
    topology
        Topology of the neighbouring cells, one of:

        - ``same-cell``: Only consider cells in isolation
        - ``von-neumann``: Use a `Von-Neumann neighbourhood \
        <https://en.wikipedia.org/wiki/Von_Neumann_neighborhood>`_
        - ``moore``: Use the `Moore neighbourhood \
        <https://en.wikipedia.org/wiki/Moore_neighborhood>`_

    Returns
    -------
    chex.Array
        2d array of offsets of neighbours from a central cell
        (also containing the central cell).
    """
    if topology == "same-cell":
        offsets = jnp.array([[0, 0]])
    elif topology == "von-neumann":
        offsets = jnp.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
    elif topology == "moore":
        offsets = list(product([-1, 0, 1], [-1, 0, 1]))
        offsets = jnp.array(offsets)
    else:
        raise ValueError(
            (
                "Topology should be one of 'same-cell', "
                f"'von-neumann' or 'moore' got {topology}"
            )
        )

    return offsets


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
    length: Union[float, chex.Array] = 1.0,
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
