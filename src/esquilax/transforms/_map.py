import typing
from functools import partial
from typing import Any

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

       @esquilax.transforms.amap
       def foo(k, p, x):
           return p + x

       k = jax.random.PRNGKey(101)
       a = jax.numpy.arange(5)

       foo(k, 2, a)
       # [2, 3, 4, 5, 6]

    .. doctest:: amap
       :hide:

       >>> foo(k, 2, a).tolist()
       [2, 3, 4, 5, 6]

    Arguments can also be PyTrees, and parameters
    ``None`` if unused

    .. testcode:: amap

       @esquilax.transforms.amap
       def foo(k, _, x):
           return x[0] + x[1]

       k = jax.random.PRNGKey(101)
       a = (jax.numpy.arange(5), jax.numpy.arange(5))

       foo(k, None, a)
       # [0, 2, 4, 6, 8]

    .. doctest:: amap
       :hide:

       >>> foo(k, None, a).tolist()
       [0, 2, 4, 6, 8]

    Arguments can also multidimensional (as long as
    the first axis is the number of agents), and outputs
    can be PyTrees

    .. testcode:: amap

       @esquilax.transforms.amap
       def foo(k, _, x):
           # Returns a tuple
           return x[0], x[1]

       k = jax.random.PRNGKey(101)
       a = jax.numpy.arange(10).reshape(5, 2)

       foo(k, None, a)
       # ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

    .. doctest:: amap
       :hide:

       >> foo(k, None, a)
       ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

    Parameters
    ----------
    f
        Update function that should have the signature

        .. code-block:: python

           def f(k, params, x, **static_kwargs):
               ...
               return y

        where the arguments are:

        - ``k``: A JAX PRNGKey
        - ``params``: Parameters (shared across the map)
        - ``x``: State data to map over.
        - ``**static_kwargs``: Any values required at compile
          time by JAX can be passed as keyword arguments.
    """

    keyword_args = utils.functions.get_keyword_args(f)

    @partial(jax.jit, static_argnames=keyword_args)
    def _self(k: chex.PRNGKey, params: Any, x: Any, **static_kwargs) -> Any:
        n = utils.functions.get_size(x)
        keys = jax.random.split(k, n)
        results = jax.vmap(partial(f, **static_kwargs), in_axes=(0, None, 0))(
            keys, params, x
        )
        return results

    return _self
