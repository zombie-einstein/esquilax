import typing
from functools import partial
from typing import Any, Optional

import chex
import jax

from esquilax import utils


def amap(f: typing.Callable) -> typing.Callable:
    """
    Decorator that maps a function over agent state data

    Maps an update function over state data,
    broadcasting model parameters, and assigning unique
    random keys. The size of the output data is the same
    as the inputs, i.e. it produces a result for each agent.

    Inputs and outputs can be PyTrees of arrays
    where each array entry represent an agent. See
    below for examples.

    Examples
    --------

    .. testsetup:: amap

       import esquilax
       import jax

    .. testcode:: amap

       def foo(p, x):
           return p + x

       a = jax.numpy.arange(5)

       result = esquilax.transforms.amap(foo)(2, a)
       # [2, 3, 4, 5, 6]

    .. doctest:: amap
       :hide:

       >>> result.tolist()
       [2, 3, 4, 5, 6]

    It can also be used as a decortor. Arguments can
    also be PyTrees, and parameters ``None`` if unused

    .. testcode:: amap

       @esquilax.transforms.amap
       def foo(_, x):
           return x[0] + x[1]

       a = (jax.numpy.arange(5), jax.numpy.arange(5))

       result = foo(None, a)
       # result = [0, 2, 4, 6, 8]

    .. doctest:: amap
       :hide:

       >>> result.tolist()
       [0, 2, 4, 6, 8]

    Arguments can also multidimensional (as long as
    the first axis is the number of agents), and outputs
    can be PyTrees

    .. testcode:: amap

       @esquilax.transforms.amap
       def foo(_, x):
           # Returns a tuple
           return x[0], x[1]

       a = jax.numpy.arange(10).reshape(5, 2)

       foo(None, a)
       # ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

    .. doctest:: amap
       :hide:

       >> foo(None, a)
       ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

    JAX random keys can be passed to the wrapped function
    by including the ``key`` keyword argument

    .. testcode:: amap

       def foo(_, x, *, key):
           return x + jax.random.choice(key, 100, ())

       k = jax.random.PRNGKey(101)
       result = esquilax.transforms.amap(foo)(None, a, key=k)

    Parameters
    ----------
    f
        Update function that should have the signature

        .. code-block:: python

           def f(params, x, **static_kwargs):
               ...
               return y

        where the arguments are:

        - ``params``: Parameters (shared across the map)
        - ``x``: State data to map over.
        - ``**static_kwargs``: Any values required at compile
          time by JAX can be passed as keyword arguments.

        Random keys can be passed to the wrapped function
        by including the ``key`` keyword argument.
    """

    keyword_args = utils.functions.get_keyword_args(f)
    has_key, keyword_args = utils.functions.has_key_keyword(keyword_args)

    @partial(jax.jit, static_argnames=keyword_args)
    def _self(
        params: Any,
        x: chex.ArrayTree,
        key: Optional[chex.PRNGKey] = None,
        **static_kwargs,
    ) -> chex.ArrayTree:
        chex.assert_tree_has_only_ndarrays(x)
        n = utils.functions.get_size(x)

        if has_key:
            assert key is not None, "Expected keyword argument 'key'"
            keys = jax.random.split(key, n)
            results = jax.vmap(
                lambda k, p, a: f(p, a, key=k, **static_kwargs), in_axes=(0, None, 0)
            )(keys, params, x)
        else:
            assert key is None, "Received unexpected 'key' keyword argument"
            results = jax.vmap(partial(f, **static_kwargs), in_axes=(None, 0))(
                params, x
            )

        return results

    return _self
