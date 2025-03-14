from typing import List, Tuple

import jax
import jax.numpy as jnp
import pytest

from esquilax import utils


@pytest.mark.parametrize(
    "tree",
    [
        jnp.zeros(10),
        jnp.zeros((10, 2)),
        (jnp.zeros(10), jnp.zeros(10)),
        [jnp.zeros(10), jnp.zeros(10)],
        {"a": jnp.zeros(10), "b": jnp.zeros(10)},
    ],
)
def test_tree_size(tree):
    assert utils.functions.get_size(tree) == 10


def test_edge_sort():
    edges = jnp.array(
        [
            [0, 1, 2, 1, 0],
            [2, 0, 0, 2, 1],
        ]
    )

    sorted_edges = utils.sort_edges(edges)

    expected = jnp.array(
        [
            [0, 0, 1, 1, 2],
            [1, 2, 0, 2, 0],
        ]
    )

    assert jnp.array_equal(sorted_edges, expected)

    data = jnp.arange(5)
    expected_data = jnp.array([4, 0, 1, 3, 2])

    sorted_edges, sorted_data = utils.sort_edges(edges, data)

    assert jnp.array_equal(sorted_edges, expected)
    assert jnp.array_equal(sorted_data, expected_data)

    sorted_edges, sorted_data_0, sorted_data_1 = utils.sort_edges(edges, data, data)

    assert jnp.array_equal(sorted_edges, expected)
    assert jnp.array_equal(sorted_data_0, expected_data)
    assert jnp.array_equal(sorted_data_1, expected_data)


def test_index_bins():
    x = jnp.array([0, 0, 0, 2, 2, 5])
    counts, bins = utils.graph.index_bins(x, 7)

    expected_counts = jnp.array([3, 0, 2, 0, 0, 1, 0])
    expected_bins = jnp.array([[0, 3], [3, 3], [3, 5], [5, 5], [5, 5], [5, 6], [6, 6]])

    assert jnp.array_equal(counts, expected_counts)
    assert jnp.array_equal(bins, expected_bins)


@pytest.mark.parametrize(
    "n_cells, width, expected",
    [
        ((2, 2), 0.5, [0, 1, 2, 3]),
        ((3, 2), 0.5, [0, 1, 2, 3]),
        ((2, 3), 0.5, [0, 1, 3, 4]),
        ((3, 3), 0.3, [0, 2, 6, 8]),
    ],
)
def test_grid_bins(n_cells: Tuple[int, int], width: float, expected: List[int]):
    x = jnp.array([[0.2, 0.2], [0.2, 0.7], [0.7, 0.2], [0.7, 0.7]])
    _, b = utils.space.get_bins(x, n_cells, width)
    assert jnp.array_equal(b, jnp.array(expected))


@pytest.mark.parametrize(
    "topology, expected",
    [
        ("same-cell", [[0], [1], [2], [3]]),
        (
            "von-neumann",
            [
                [0, 1, 2, 1, 2],
                [1, 0, 3, 0, 3],
                [2, 3, 0, 3, 0],
                [3, 2, 1, 2, 1],
            ],
        ),
        (
            "moore",
            [
                [3, 2, 3, 1, 0, 1, 3, 2, 3],
                [2, 3, 2, 0, 1, 0, 2, 3, 2],
                [1, 0, 1, 3, 2, 3, 1, 0, 1],
                [0, 1, 0, 2, 3, 2, 0, 1, 0],
            ],
        ),
    ],
)
def test_topology(topology: str, expected: List[List[int]]):
    idxs = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    offsets = utils.space.get_neighbours_offsets(topology)
    x = jax.vmap(lambda a: utils.space.neighbour_indices(a, offsets, (2, 2)))(idxs)
    assert jnp.array_equal(x, jnp.array(expected))


@pytest.mark.parametrize(
    "n_bins, idx, expected",
    [
        ((3, 2), [0, 0], [5, 4, 5, 1, 0, 1, 3, 2, 3]),
        ((3, 2), [2, 1], [2, 3, 2, 4, 5, 4, 0, 1, 0]),
        ((2, 3), [0, 0], [5, 3, 4, 2, 0, 1, 5, 3, 4]),
        ((2, 3), [1, 2], [1, 2, 0, 4, 5, 3, 1, 2, 0]),
        ((4, 3), [1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ((4, 1), [1, 0], [0, 0, 0, 1, 1, 1, 2, 2, 2]),
    ],
)
def test_non_square_topology(
    n_bins: Tuple[int, int], idx: List[int], expected: List[int]
):
    idx = jnp.array(idx)
    offsets = utils.space.get_neighbours_offsets("moore")
    x = utils.space.neighbour_indices(idx, offsets, n_bins)
    assert jnp.array_equal(x, jnp.array(expected))


@pytest.mark.parametrize(
    "vec_a, vec_b, expected",
    [
        ([0.8, 0.5], [0.9, 0.5], [0.1, 0.0]),
        ([0.8, 0.5], [0.1, 0.5], [0.3, 0.0]),
        ([0.8, 0.5], [0.2, 0.5], [0.4, 0.0]),
        ([0.8, 0.5], [0.4, 0.5], [-0.4, 0.0]),
        ([0.8, 0.5], [0.6, 0.5], [-0.2, 0.0]),
        ([0.2, 0.5], [0.8, 0.5], [-0.4, 0.0]),
        ([0.5, 0.8], [0.5, 0.2], [0.0, 0.4]),
        ([0.5, 0.2], [0.5, 0.8], [0.0, -0.4]),
    ],
)
def test_shortest_vector(vec_a, vec_b, expected):
    shortest_vec = utils.shortest_vector(jnp.array(vec_a), jnp.array(vec_b), 1.0)

    assert jnp.allclose(jnp.array(expected), shortest_vec)


@pytest.mark.parametrize(
    "vec_a, vec_b, expected",
    [
        ([0.8, 0.5], [0.8, 0.5], 0),
        ([0.8, 0.5], [0.5, 0.5], 0.3),
        ([0.8, 0.5], [0.1, 0.5], 0.3),
        ([0.1, 0.5], [0.8, 0.5], 0.3),
        ([0.5, 0.8], [0.5, 0.5], 0.3),
        ([0.5, 0.8], [0.5, 0.1], 0.3),
        ([0.5, 0.1], [0.5, 0.8], 0.3),
    ],
)
def test_shortest_distance(vec_a, vec_b, expected):
    d = utils.shortest_distance(jnp.array(vec_a), jnp.array(vec_b), 1.0, norm=True)

    assert jnp.isclose(d, expected)


def test_keyword_arguments():
    def foo(_a, *, _b, _c):
        pass

    assert utils.functions.get_keyword_args(foo) == ["_b", "_c"]

    def bar(_a):
        pass

    assert utils.functions.get_keyword_args(bar) == []


def test_tree_key_split():
    k = jax.random.PRNGKey(451)

    a = {"a": 1, "b": (2, 3)}
    b = utils.tree.key_tree_split(k, a)

    assert jax.tree.structure(a) == jax.tree.structure(b)
    assert not jnp.array_equal(b["a"], b["b"][0])
    assert not jnp.array_equal(b["a"], b["b"][1])


def test_tuple_tree_transpose():
    a = {"a": 1, "b": 2}
    b = {"a": (1, (2, 3)), "b": (4, (5, 6))}
    c = utils.tree.transpose_tree_of_tuples(a, b, 2)

    assert c == ({"a": 1, "b": 4}, {"a": (2, 3), "b": (5, 6)})
