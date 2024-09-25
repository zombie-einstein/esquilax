import chex
import jax
import jax.numpy as jnp
import pytest

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

    @transforms.spatial(
        2, jnp.add, 0, include_self=include_self, topology=topology, i_range=i_range
    )
    def foo(_, params, a, b):
        return params + a + b

    vals = jnp.arange(5)
    results = foo(k, 2, vals, vals, pos=x)

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

    @transforms.spatial(
        2,
        jnp.add,
        jnp.zeros(2),
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )
    def foo(_, params, a, b):
        return params + a + b

    agents = jnp.column_stack([jnp.arange(3), jnp.arange(3) + 1])

    results = foo(k, jnp.ones(2), agents, agents, pos=x)

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

    @transforms.spatial(
        2,
        {"a": jnp.add, "b": jnp.add},
        {"a": 0, "b": 0},
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )
    def foo(_, params, a, b):
        return {"a": params + a["a"] + b["a"], "b": params + a["b"] + b["b"]}

    agents = {"a": jnp.arange(3), "b": jnp.arange(3) + 1}

    results = foo(k, 2, agents, agents, pos=x)

    expected = {"a": jnp.array(expected["a"]), "b": expected["b"]}

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

    @transforms.spatial(
        2,
        (jnp.add, jnp.add),
        (0, 0),
        include_self=include_self,
        topology=topology,
        i_range=1.0,
    )
    def foo(_, params, a, b):
        return params + a[0] + b[0], params + a[1] + b[1]

    agents = (jnp.arange(3), jnp.arange(3) + 1)

    results = foo(k, 2, agents, agents, pos=x)

    expected = (jnp.array(expected[0]), jnp.array(expected[1]))

    assert jnp.array_equal(results[0], expected[0])
    assert jnp.array_equal(results[1], expected[1])


def test_spatial_interaction_w_static():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.7, 0.1], [0.1, 0.7], [0.75, 0.2], [0.75, 0.75]])

    expected = [48, 46, 72, 62, 72]

    @transforms.spatial(
        2, jnp.add, 0, include_self=False, topology="moore", i_range=10.0
    )
    def foo(_, params, a, b, *, func):
        return func(params, a, b)

    def bar(a, b, c):
        return a + b + c

    vals = jnp.arange(5)
    results = foo(k, 2, vals, vals, func=bar, pos=x)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


def test_spatial_mixed_data():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.1, 0.7]])

    @transforms.spatial(
        2, jnp.add, 0, include_self=False, topology="von-neumann", i_range=10.0
    )
    def foo(_, params, a, b):
        return params + a + b

    vals_a = jnp.arange(1, 3)
    vals_b = jnp.arange(6, 8)

    results = foo(k, 2, vals_a, vals_b, pos=x)

    expected = [20, 20]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )


def test_spatial_w_none():
    k = jax.random.PRNGKey(101)
    x = jnp.array([[0.1, 0.1], [0.1, 0.7]])

    @transforms.spatial(
        2, jnp.add, 0, include_self=False, topology="von-neumann", i_range=10.0
    )
    def foo(_, params, _a, b):
        return params + b

    vals_b = jnp.arange(1, 3)

    results = foo(k, 2, None, vals_b, pos=x)

    expected = [8, 6]

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )

    @transforms.spatial(
        2, jnp.add, 0, include_self=False, topology="von-neumann", i_range=10.0
    )
    def bar(_, params, a, _b):
        return params + a

    vals_a = jnp.arange(1, 3)

    results = bar(k, 2, vals_a, None, pos=x)

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

    @transforms.spatial(
        2, jnp.add, 0, include_self=include_self, topology=topology, i_range=i_range
    )
    def foo(_, params, a, b):
        return params + a + b

    vals_a = jnp.arange(3)
    vals_b = jnp.arange(1, 3)

    results = foo(k, 2, vals_a, vals_b, pos=xa, pos_b=xb)

    assert jnp.array_equal(
        results,
        jnp.array(expected),
    )
