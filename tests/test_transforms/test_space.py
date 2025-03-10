from typing import List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import pytest

import esquilax.utils
from esquilax import reductions, transforms


@pytest.mark.parametrize(
    "expected, include_self, topology, i_range",
    [
        ([2, 10, 6, 14, 10], True, "same-cell", 10.0),
        ([0, 6, 0, 6, 0], False, "same-cell", 10.0),
        ([26, 30, 30, 42, 58], True, "von-neumann", 10.0),
        ([24, 26, 24, 34, 48], False, "von-neumann", 10.0),
        ([50, 50, 78, 70, 82], True, "moore", 10.0),
        ([48, 46, 72, 62, 72], False, "moore", 10.0),
        ([2, 4, 6, 8, 10], True, "moore", 0.00001),
        ([0, 0, 0, 0, 0], False, "moore", 0.00001),
    ],
)
def test_spatial_interaction(
    expected: chex.Array, include_self: bool, topology: str, i_range: float
):
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.1, 0.7], [0.75, 0.2], [0.75, 0.75]])

    def foo(params, a, b):
        return params + a + b

    vals = jnp.arange(5)
    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reductions.add(dtype=int),
        include_self=include_self,
        topology=topology,
        i_range=i_range,
    )(
        2,
        vals,
        vals,
        pos=x,
    )

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


@pytest.mark.parametrize(
    "expected, include_self, topology",
    [
        ([[1, 3], [7, 11], [9, 13]], True, "same-cell"),
        ([[0, 0], [4, 6], [4, 6]], False, "same-cell"),
        ([[11, 21], [11, 19], [15, 23]], True, "moore"),
        ([[10, 18], [8, 14], [10, 16]], False, "moore"),
    ],
)
def test_spatial_w_array(expected, include_self, topology):
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.75, 0.2]])

    def foo(params, a, b):
        return params + a + b

    agents = jnp.column_stack([jnp.arange(3), jnp.arange(3) + 1])

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reductions.add(shape=(2,), dtype=int),
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )(
        jnp.ones(2, dtype=int),
        agents,
        agents,
        pos=x,
    )

    assert jnp.array_equal(results, jnp.array(expected))


@pytest.mark.parametrize(
    "expected, include_self, topology",
    [
        ({"a": [2, 9, 11], "b": [4, 13, 15]}, True, "same-cell"),
        ({"a": [0, 5, 5], "b": [0, 7, 7]}, False, "same-cell"),
        ({"a": [16, 15, 19], "b": [26, 23, 27]}, True, "moore"),
        ({"a": [14, 11, 13], "b": [22, 17, 19]}, False, "moore"),
    ],
)
def test_spatial_w_dict(expected, include_self, topology):
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.75, 0.2]])

    def foo(params, a, b):
        return {"a": params + a["a"] + b["a"], "b": params + a["b"] + b["b"]}

    agents = {"a": jnp.arange(3), "b": jnp.arange(3) + 1}

    reduction = reductions.Reduction({"a": jnp.add, "b": jnp.add}, {"a": 0, "b": 0})

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reduction,
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )(
        2,
        agents,
        agents,
        pos=x,
    )

    expected = {"a": jnp.array(expected["a"]), "b": jnp.array(expected["b"])}

    assert jnp.array_equal(results["a"], expected["a"])
    assert jnp.array_equal(results["b"], expected["b"])


@pytest.mark.parametrize(
    "expected, include_self, topology",
    [
        (([2, 9, 11], [4, 13, 15]), True, "same-cell"),
        (([0, 5, 5], [0, 7, 7]), False, "same-cell"),
        (([16, 15, 19], [26, 23, 27]), True, "moore"),
        (([14, 11, 13], [22, 17, 19]), False, "moore"),
    ],
)
def test_spatial_w_tuple(expected, include_self, topology):
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.75, 0.2]])

    def foo(params, a, b):
        return params + a[0] + b[0], params + a[1] + b[1]

    agents = (jnp.arange(3), jnp.arange(3) + 1)

    reduction = reductions.Reduction((jnp.add, jnp.add), (0, 0))

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reduction,
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )(2, agents, agents, pos=x)

    expected = (jnp.array(expected[0]), jnp.array(expected[1]))

    assert jnp.array_equal(results[0], expected[0])
    assert jnp.array_equal(results[1], expected[1])


def test_spatial_interaction_w_static():
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.1, 0.7], [0.75, 0.2], [0.75, 0.75]])

    expected = [48, 46, 72, 62, 72]

    def foo(params, a, b, *, func):
        return func(params, a, b)

    def bar(a, b, c):
        return a + b + c

    vals = jnp.arange(5)
    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reductions.add(dtype=int),
        include_self=False,
        topology="moore",
        i_range=10.0,
    )(2, vals, vals, func=bar, pos=x)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


def test_spatial_w_rng(rng_key: chex.PRNGKey):
    x = jnp.array([[0.1, 0.1], [0.1, 0.2]])

    def foo(_p, _a, _b, *, key):
        return jax.random.choice(key, 10_000, ())

    results = transforms.spatial(
        foo,
        reduction=reductions.add(dtype=int),
        include_self=False,
        topology="moore",
        i_range=0.3,
    )(None, None, None, key=rng_key, pos=x)

    assert results.shape == (2,)
    assert results[0] != results[1]


def test_spatial_mixed_data():
    x = jnp.array([[0.1, 0.1], [0.1, 0.7]])

    def foo(params, a, b):
        return params + a + b

    vals_a = jnp.arange(1, 3)
    vals_b = jnp.arange(6, 8)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reductions.add(dtype=int),
        include_self=False,
        topology="von-neumann",
        i_range=10.0,
    )(2, vals_a, vals_b, pos=x)

    expected = [20, 20]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


def test_spatial_w_none():
    x = jnp.array([[0.1, 0.1], [0.1, 0.7]])

    def foo(params, _a, b):
        return params + b

    vals_b = jnp.arange(1, 3)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reductions.add(dtype=int),
        include_self=False,
        topology="von-neumann",
        i_range=10.0,
    )(2, None, vals_b, pos=x)

    expected = [8, 6]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )

    def bar(params, a, _b):
        return params + a

    vals_a = jnp.arange(1, 3)

    results = transforms.spatial(
        bar,
        n_bins=2,
        reduction=reductions.add(dtype=int),
        include_self=False,
        topology="von-neumann",
        i_range=10.0,
    )(2, vals_a, None, pos=x)

    expected = [6, 8]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


@pytest.mark.parametrize(
    "expected, include_self, topology, i_range",
    [
        ([0, 0, 5], True, "same-cell", 10.0),
        ([0, 0, 5], False, "same-cell", 10.0),
        ([6, 10, 17], True, "von-neumann", 10.0),
        ([6, 10, 17], False, "von-neumann", 10.0),
        ([22, 26, 17], True, "moore", 10.0),
        ([22, 26, 17], False, "moore", 10.0),
        ([0, 0, 0], True, "moore", 0.00001),
        ([0, 0, 0], False, "moore", 0.00001),
    ],
)
def test_mixed_type_spatial_interaction(
    expected: chex.Array, include_self: bool, topology: str, i_range: float
):
    xa = jnp.array([[0.1, 0.1], [0.1, 0.7], [0.75, 0.2]])
    xb = jnp.array([[0.7, 0.1], [0.75, 0.75]])

    def foo(params, a, b):
        return params + a + b

    vals_a = jnp.arange(3)
    vals_b = jnp.arange(1, 3)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=reductions.add(dtype=int),
        include_self=include_self,
        topology=topology,
        i_range=i_range,
    )(2, vals_a, vals_b, pos=xa, pos_b=xb)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


@pytest.mark.parametrize(
    "n_agents, i_range",
    [
        (2, 0.1),
        (10, 0.1),
        (20, 0.1),
        (40, 0.1),
        (2, 0.05),
        (10, 0.05),
        (20, 0.05),
        (40, 0.05),
    ],
)
def test_space_fuzzy_same_type(n_agents: int, i_range: float):
    k = jax.random.PRNGKey(101)
    x = jax.random.uniform(k, (n_agents, 2))

    def foo(_p, a, b):
        return a - b

    vals_a = jnp.arange(1, n_agents + 1)
    vals_b = jnp.arange(2, n_agents + 2)
    results = transforms.spatial(
        foo,
        reduction=reductions.add(dtype=int),
        include_self=True,
        topology="moore",
        i_range=i_range,
    )(None, vals_a, vals_b, pos=x)

    d = jax.vmap(
        lambda a: jax.vmap(lambda b: esquilax.utils.shortest_distance(a, b, norm=True))(
            x
        )
    )(x)

    expected = jax.vmap(lambda a, b: a - b, in_axes=(0, None))(vals_a, vals_b)
    expected = jnp.where(d < i_range, expected, 0)
    expected = jnp.sum(expected, axis=1)

    assert jnp.array_equal(results, expected)


@pytest.mark.parametrize(
    "n_agents, i_range",
    [
        (2, 0.1),
        (10, 0.1),
        (20, 0.1),
        (40, 0.1),
        (2, 0.05),
        (10, 0.05),
        (20, 0.05),
        (40, 0.05),
    ],
)
def test_space_fuzzy_diff_types(n_agents: int, i_range: float):
    k = jax.random.PRNGKey(101)
    ka, kb = jax.random.split(k)

    n_agents_a = n_agents
    n_agents_b = n_agents + 2

    xa = jax.random.uniform(ka, (n_agents_a, 2))
    xb = jax.random.uniform(kb, (n_agents_b, 2))

    def foo(_p, a, b):
        return a - b

    vals_a = jnp.arange(1, n_agents_a + 1)
    vals_b = jnp.arange(2, n_agents_b + 2)
    results = transforms.spatial(
        foo, reduction=reductions.add(dtype=int), topology="moore", i_range=i_range
    )(None, vals_a, vals_b, pos=xa, pos_b=xb)

    d = jax.vmap(
        lambda a: jax.vmap(lambda b: esquilax.utils.shortest_distance(a, b, norm=True))(
            xb
        )
    )(xa)

    expected = jax.vmap(lambda a, b: a - b, in_axes=(0, None))(vals_a, vals_b)
    expected = jnp.where(d < i_range, expected, 0)
    expected = jnp.sum(expected, axis=1)

    assert jnp.array_equal(results, expected)


@pytest.mark.parametrize(
    "x, i_range, dims, expected",
    [
        ([[1.0, 0.9], [1.0, 1.1]], 0.4, 2.0, [5, 5]),
        ([[1.0, 1.9], [1.0, 0.1]], 0.4, 2.0, [5, 5]),
        ([[1.9, 1.0], [0.1, 1.0]], 0.4, 2.0, [5, 5]),
        ([[0.25, 0.21], [0.25, 0.3]], 0.1, 0.5, [5, 5]),
        ([[0.25, 0.04], [0.25, 0.46]], 0.1, 0.5, [5, 5]),
        ([[0.5, 0.95], [0.5, 1.05]], 0.2, (1.0, 2.0), [5, 5]),
        ([[0.5, 0.05], [0.5, 1.95]], 0.2, (1.0, 2.0), [5, 5]),
        ([[0.95, 0.5], [1.05, 0.5]], 0.2, (2.0, 1.0), [5, 5]),
        ([[0.05, 0.5], [1.95, 0.5]], 0.2, (2.0, 1.0), [5, 5]),
    ],
)
def test_spatial_non_unit_region(
    x: List[List[float]],
    i_range: float,
    dims: Union[float, Tuple[float, float]],
    expected: List[int],
):
    x = jnp.array(x)

    def foo(params, a, b):
        return params + a + b

    vals = jnp.arange(1, 3)
    results = transforms.spatial(
        foo,
        reduction=reductions.add(dtype=int),
        include_self=False,
        topology="moore",
        i_range=i_range,
        dims=dims,
    )(
        2,
        vals,
        vals,
        pos=x,
    )

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


@pytest.mark.parametrize(
    "i_range, dims, n_bins, expected_n_bins, expected_width, expected_dims",
    [
        (None, [1.0, 2.0], [1, 2], (1, 2), 1.0, [1.0, 2.0]),
        (None, [0.2, 0.1], [2, 1], (2, 1), 0.1, [0.2, 0.1]),
        (1.0, [1.0, 2.0], None, (1, 2), 1.0, [1.0, 2.0]),
        (0.1, [0.2, 0.1], None, (2, 1), 0.1, [0.2, 0.1]),
        (None, 1.0, 2, (2, 2), 0.5, [1.0, 1.0]),
        (0.1, 1.0, None, (10, 10), 0.1, [1.0, 1.0]),
    ],
)
def test_parameter_processing(
    i_range: float,
    dims: float | List[float],
    n_bins: int | List[int],
    expected_n_bins: Tuple[int, int],
    expected_width: float,
    expected_dims: List[float],
):
    n_bins, width, dims = transforms._space._process_parameters(i_range, dims, n_bins)

    assert n_bins == expected_n_bins
    assert jnp.isclose(width, expected_width)
    assert jnp.array_equal(dims, jnp.array(expected_dims))


def test_parameter_checks():
    with pytest.raises(
        AssertionError, match="2 spatial dimensions should be provided got 3"
    ):
        transforms._space._process_parameters(0.1, [1.0, 1.0, 1.0], [2, 2, 2])

    with pytest.raises(
        AssertionError,
        match="n_bins should be a sequence if dims is a sequence, got <class 'int'>",
    ):
        transforms._space._process_parameters(0.1, [1.0, 1.0], 2)

    with pytest.raises(
        AssertionError,
        match="Number of bins should be provided for 2 dimensions, got 3",
    ):
        transforms._space._process_parameters(0.1, [1.0, 1.0], [2, 2, 2])

    with pytest.raises(
        AssertionError, match="n_bins should all be greater than 0, got \\[2, 0\\]"
    ):
        transforms._space._process_parameters(0.1, [1.0, 1.0], [2, 0])

    with pytest.raises(
        AssertionError,
        match="Dimensions of cells should be equal in both dimensions got 0.5 and 1.0",
    ):
        transforms._space._process_parameters(0.1, [1.0, 1.0], [2, 1])

    with pytest.raises(
        AssertionError,
        match="If n_bins is not provided, i_range should be provided",
    ):
        transforms._space._process_parameters(None, [1.0, 1.0], None)

    with pytest.raises(
        AssertionError,
        match="Dimensions should be a multiple of i_range",
    ):
        transforms._space._process_parameters(0.25, [0.8, 1.0], None)

    with pytest.raises(
        AssertionError,
        match="Dimensions should be a multiple of i_range",
    ):
        transforms._space._process_parameters(0.25, [1.0, 0.8], None)
