User Guide
==========

Concepts
--------

Parameters & State
^^^^^^^^^^^^^^^^^^

Data in an Esquilax model is generally broken down into two components:

- **Parameters**: These are generally model hyperparameters and are
  shared (i.e. broadcast) across agents and are static over time.
- **State**: These represent the *current* state of the model,
  are mapped across, and updated when applying updates of the model.

For example in a model of a swarm, parameters might be:

- The min/max speed of the agents
- The steering parameters of the agents

and the state might be

- The positions and velocities of each agent

PyTrees
^^^^^^^

JAX has the concept of a `PyTree <https://jax.readthedocs.io/en/latest/pytrees.html#pytrees>`_,
a tree structure generated from Python containers. In Esquilax the state of agents/entities
is generally represented by a PyTree of arrays, where the length of the arrays corresponds to
the number of entities/agents. For a simple model this could well be a single array

.. code-block:: python

   agent_state = jnp.zeros(10)

or a container/dataclass of multiple arrays of data

.. code-block:: python

   agent_state = {"x": jnp.zeros(10), "y": jnp.zeros(10)}

   agent_state = (jnp.zeros(10), jnp.zeros(10))

   @chex.dataclass
   class State:
       x: float
       y: float

   agent_state = State(
       x=jnp.zeros(10), y=jnp.zeros(10)
   )

or nested combinations of these

.. code-block::

   agent_state = (
       State(x=jnp.zeros(10), y=jnp.zeros(10)),
       jnp.zeros(10),
   )

As long as each array in the tree has the same length (i.e. in axis 0), Esquilax will
handle mapping over the tree structure.

Map-Reduce Interaction Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Esquilax is largely designed around the notion that in a multi-agent system, state
is updated by entities performing the following steps

- Observe the (local) state of the system
- Take some action or update their state

and then we might also

- Update the state of the system according to some model dynamics

For example in a swarm models we might have the following steps:

- Each agent observes the positions and velocities of agents within a given range
- Each agent updates it velocity vector based on this observation
- The position of each agent is updated using their updated velocity

This process may be familiar from the observation-action loop
(or Markov decision process) often seen in
reinforcement learning problems. Esquilax intends to extend this paradigm
to large-numbers of agents performing observations and updates in parallel.

As such Esquilax employs `map-reduce <https://en.wikipedia.org/wiki/MapReduce>`_
patterns to apply updates:

- **Observe**: Map an observation function over pairs of agents, then aggregate
  (i.e. reduce) the observations into a single observation for each agent.
- **Update**: Map an update function over a set of agents.

Esquilax handles the process of mapping observations/updates over sets of agents,
and allows these updates to be composed to build complex models. In each
case model parameters are broadcast across all agents.

For example, in a model where agents are nodes on a graph we might have
an update where each agent observes its neighbour on the graph. Using
Esquilax this could look like

.. code-block:: python

   @esquilax.transforms.graph_reduce(jnp.add, 0)
   def collect_opinions(_, params, a, b, edge_weight):
       return params.scale * edge_weight * (a - b)

This function then:

- Maps ``collect_opinions`` across graph edges (in parallel), calculating

  .. code-block:: python

     params.scale * edge_weight * (a - b)

  the difference of the node values, scaled by the weight assigned to the
  edge and a shared scaling parameter.
- Add up contributions from edges based on the start node, creating
  an array of results for each agent on the graph. In the case
  an agent has no edges the default value ``0`` is returned.

Some transformations use variations on this pattern. For example
a transformation might select a random neighbour, and the apply
a observation function to the agent and the randomly selected neighbour.

Parallelisation
^^^^^^^^^^^^^^^

Esquilax attempts to maintain performance as agent numbers scale by
ensuring observations/updates can be performed in parallel where possible.
The performance benefits come with some constraints.

In particular any reductions must be implemented as a monoid, i.e. a
function that takes two argument of a given type, and returns the same
type like ``(a, a) -> a``, and also has an identity/default value.

In the example above, the graph observation uses the function ``jnp.add``
along with identity ``0``, but other options could be:

- ``jnp.min`` with default value ``jnp.finfo(jnp.float32).max`` to get the minimum over neighbours
- ``jnp.logical_or`` with default value ``False`` to check if any neighbour satisfies a condition.

or you can define your own reduction function.

Neuro-Evolution and Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Esquilax provides utilities and functionality for training agent policies, where
an Esquilax simulation is used as a multi-agent training environment. They allow
for multiple strategies or RL policies to be trained inside the same training loop.
See :py:mod:`esquilax.ml` for more details.

Tips
----

Extending Functionality
^^^^^^^^^^^^^^^^^^^^^^^

Esquilax transformations can be used alongside other JAX code
and with other JAX libraries, allowing Esquilax transformations to be
combined with customised functionality. For custom behaviours to be used
inside the Esquilax simulation runner, the only requirement is that it
can be `JIT compiled <https://jax.readthedocs.io/en/latest/jit-compilation.html#just-in-time-compilation>`_.

Static Features
^^^^^^^^^^^^^^^

JAX requires certain values to be known at
`compile time <https://jax.readthedocs.io/en/latest/glossary.html#term-static>`_
such as certain data dimension, and functions passed as arguments to
JIT compiled function.

In some cases you may want to use a function inside a transformation
without writing the function ahead of time.

For example you may want to use a
`Flax <https://flax.readthedocs.io/en/latest/>`_ network inside
a transformation, without having to initialise the network when
writing the model (for instance if you want to vary network parameters).
The networks forward pass function (initialised when the network is initialised can
be passed as an argument to the inner function, but is required to be static,
i.e this

.. code-block:: python

   @esquilax.transforms.amap
   def foo(_k, f, x):
       return f(x)

will not work. Instead static arguments can be passed as additional keyword
arguments to the expected transformation signature, like

.. code-block:: python

   @esquilax.transforms.amap
   def foo(_k, _, x, *, f):
       return f(x)

   results = foo(k, None, y, f=network_func)

will mark the keyword arguments as static during compilation.
