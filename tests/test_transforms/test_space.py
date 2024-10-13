import chex
import jax
import jax.numpy as jnp
import pytest

import esquilax.utils
from esquilax import transforms


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
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.1, 0.7], [0.75, 0.2], [0.75, 0.75]])

    def foo(_, params, a, b):
        return params + a + b

    vals = jnp.arange(5)
    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=jnp.add,
        default=0,
        include_self=include_self,
        topology=topology,
        i_range=i_range,
    )(
        k,
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
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.75, 0.2]])

    def foo(_, params, a, b):
        return params + a + b

    agents = jnp.column_stack([jnp.arange(3), jnp.arange(3) + 1])

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=jnp.add,
        default=jnp.zeros(2),
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )(
        k,
        jnp.ones(2),
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
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.75, 0.2]])

    def foo(_, params, a, b):
        return {"a": params + a["a"] + b["a"], "b": params + a["b"] + b["b"]}

    agents = {"a": jnp.arange(3), "b": jnp.arange(3) + 1}

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction={"a": jnp.add, "b": jnp.add},
        default={"a": 0, "b": 0},
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )(
        k,
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
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.75, 0.2]])

    def foo(_, params, a, b):
        return params + a[0] + b[0], params + a[1] + b[1]

    agents = (jnp.arange(3), jnp.arange(3) + 1)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=(jnp.add, jnp.add),
        default=(0, 0),
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )(k, 2, agents, agents, pos=x)

    expected = (jnp.array(expected[0]), jnp.array(expected[1]))

    assert jnp.array_equal(results[0], expected[0])
    assert jnp.array_equal(results[1], expected[1])


def test_spatial_interaction_w_static():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.1, 0.7], [0.75, 0.2], [0.75, 0.75]])

    expected = [48, 46, 72, 62, 72]

    def foo(_, params, a, b, *, func):
        return func(params, a, b)

    def bar(a, b, c):
        return a + b + c

    vals = jnp.arange(5)
    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=jnp.add,
        default=0,
        include_self=False,
        topology="moore",
        i_range=10.0,
    )(k, 2, vals, vals, func=bar, pos=x)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


def test_spatial_mixed_data():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.1, 0.7]])

    def foo(_, params, a, b):
        return params + a + b

    vals_a = jnp.arange(1, 3)
    vals_b = jnp.arange(6, 8)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=jnp.add,
        default=0,
        include_self=False,
        topology="von-neumann",
        i_range=10.0,
    )(k, 2, vals_a, vals_b, pos=x)

    expected = [20, 20]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


def test_spatial_w_none():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.1, 0.7]])

    def foo(_, params, _a, b):
        return params + b

    vals_b = jnp.arange(1, 3)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=jnp.add,
        default=0,
        include_self=False,
        topology="von-neumann",
        i_range=10.0,
    )(k, 2, None, vals_b, pos=x)

    expected = [8, 6]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )

    def bar(_, params, a, _b):
        return params + a

    vals_a = jnp.arange(1, 3)

    results = transforms.spatial(
        bar,
        n_bins=2,
        reduction=jnp.add,
        default=0,
        include_self=False,
        topology="von-neumann",
        i_range=10.0,
    )(k, 2, vals_a, None, pos=x)

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
    k = jax.random.PRNGKey(101)

    xa = jnp.array([[0.1, 0.1], [0.1, 0.7], [0.75, 0.2]])
    xb = jnp.array([[0.7, 0.1], [0.75, 0.75]])

    def foo(_, params, a, b):
        return params + a + b

    vals_a = jnp.arange(3)
    vals_b = jnp.arange(1, 3)

    results = transforms.spatial(
        foo,
        n_bins=2,
        reduction=jnp.add,
        default=0,
        include_self=include_self,
        topology=topology,
        i_range=i_range,
    )(k, 2, vals_a, vals_b, pos=xa, pos_b=xb)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


@pytest.mark.parametrize(
    "expected, topology, i_range",
    [
        ([3, 3, -1], "same-cell", 10.0),
        ([3, 3, 5], "von-neumann", 10.0),
        ([3, 3, 5], "moore", 10.0),
        ([-1, -1, -1], "moore", 0.00001),
    ],
)
def test_nearest_neighbour(expected: chex.Array, topology: str, i_range: float):
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.1, 0.2], [0.1, 0.6]])

    def foo(_, params, a, b):
        return params + a + b

    vals = jnp.arange(3)
    results = transforms.nearest_neighbour(
        foo, n_bins=2, default=-1, topology=topology, i_range=i_range
    )(k, 2, vals, vals, pos=x)

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

    def foo(_k, _p, a, b):
        return a - b

    vals_a = jnp.arange(1, n_agents + 1)
    vals_b = jnp.arange(2, n_agents + 2)
    results = transforms.spatial(
        foo,
        n_bins=10,
        reduction=jnp.add,
        default=0,
        include_self=True,
        topology="moore",
        i_range=i_range,
    )(k, None, vals_a, vals_b, pos=x)

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

    def foo(_k, _p, a, b):
        return a - b

    vals_a = jnp.arange(1, n_agents_a + 1)
    vals_b = jnp.arange(2, n_agents_b + 2)
    results = transforms.spatial(
        foo, n_bins=10, reduction=jnp.add, default=0, topology="moore", i_range=i_range
    )(k, None, vals_a, vals_b, pos=xa, pos_b=xb)

    d = jax.vmap(
        lambda a: jax.vmap(lambda b: esquilax.utils.shortest_distance(a, b, norm=True))(
            xb
        )
    )(xa)

    expected = jax.vmap(lambda a, b: a - b, in_axes=(0, None))(vals_a, vals_b)
    expected = jnp.where(d < i_range, expected, 0)
    expected = jnp.sum(expected, axis=1)

    assert jnp.array_equal(results, expected)
