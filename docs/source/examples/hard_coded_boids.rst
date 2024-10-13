.. _hard_coded_boids:

Hard-Coded Boids
================

This example implements the popular `boids <https://en.wikipedia.org/wiki/Boids>`_
swarming model first developed by Reynolds. The update algorithm implemented here
is adapted from
`this demo <https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html>`_.

State
-----

We first import JAX, `Chex <https://chex.readthedocs.io/en/latest/>`_, and Esquilax

.. testcode:: hard_coded_boids

   from functools import partial
   import chex
   import jax
   import jax.numpy as jnp
   import esquilax

State can be represented by any
`PyTree <https://jax.readthedocs.io/en/latest/pytrees.html#what-is-a-pytree>`_
(e.g. a dict or tuple), but in this case we will use a
`chex dataclass <https://chex.readthedocs.io/en/latest/api.html#chex.dataclass>`_
for readability

.. testcode:: hard_coded_boids

   @chex.dataclass
   class Boid:
       pos: chex.Array
       vel: chex.Array

   @chex.dataclass
   class Params:
       cohesion: float
       avoidance: float
       alignment: float
       max_speed: float
       min_speed: float
       close_range: float

The ``Boid`` class stores the state of the boids, their current positions
and velocities. The ``Params`` class stores parameters used for the steering
algorithm and when updating agent positions.

Updates
-------

We then use Esquilax to implement observation and update transformations.
Firstly agents observe the state of neighbours within a given range

.. testcode:: hard_coded_boids

   @partial(
       esquilax.transforms.spatial,
       n_bins=5,
       reduction=(jnp.add, jnp.add, jnp.add, jnp.add),
       default=(0, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)),
       include_self=False,
   )
   def observe(_key: chex.PRNGKey, params: Params, a: Boid, b: Boid):
       v = esquilax.utils.shortest_vector(b.pos, a.pos)
       d = jnp.sum(v**2)

       close_vec = jax.lax.cond(
           d < params.close_range**2,
           lambda: v,
           lambda: jnp.zeros(2),
       )

       return 1, b.pos, b.vel, close_vec

This function is mapped across all pairs of agents within range of each other.
The function calculates the distance between the two agents, and then returns a
tuple containing:

- ``1`` to count the neighbour
- The position of the neighbour
- The velocity of the neighbour
- The vector to the neighbour if it within collision range

The values are summed up, specified by the tuple of reduction functions
:code:`(jnp.add, jnp.add, jnp.add, jnp.add)` which form a monoid with
the default values :code:`(0, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2))`. The
space is subdivided into ``5`` cells along each dimension, and
agents do not include themselves in the observation by setting ``include_self=False``.
The result of this transformation is a tuple of arrays with combined observations
for each individual agent.

The next transformation combines the observations into a steering vector

.. testcode:: hard_coded_boids

   @esquilax.transforms.amap
   def steering(_key: chex.PRNGKey, params: Params, observations):
       x, v, n_nb, x_nb, v_nb, v_cl = observations

       def steer():
           x_nb_avg = x_nb / n_nb
           v_nb_avg = v_nb / n_nb
           _dv_x = params.cohesion * esquilax.utils.shortest_vector(x, x_nb_avg)
           _dv_v = params.alignment * esquilax.utils.shortest_vector(v, v_nb_avg)
           return _dv_x + _dv_v

       dv_nb = jax.lax.cond(n_nb > 0, steer, lambda: jnp.zeros(2))
       v = v + dv_nb + v_cl

       return v

``observations`` is a tuple of agent states, and the observations from ``observe``.
This function checks if the agent observed any neighbours, and if so combines
these values into a single steering vector. The function is mapped across the
argument data, and so produces a new velocity for each agent.

We then have two functions that rescales the agents velocity, and then updates their
position

.. testcode:: hard_coded_boids

   @esquilax.transforms.amap
   def limit_speed(_key: chex.PRNGKey, params: Params, v: chex.Array):
       s = jnp.sqrt(jnp.sum(v * v))

       v = jax.lax.cond(
           s < params.min_speed,
           lambda _v: params.min_speed * _v / s,
           lambda _v: _v,
           v,
       )

       v = jax.lax.cond(
           s > params.max_speed,
           lambda _v: params.max_speed * _v / s,
           lambda _v: _v,
           v
       )

       return v


   @esquilax.transforms.amap
   def move(_key: chex.PRNGKey, _params: Params, x):
       pos, vel = x
       return (pos + vel) % 1.0

These functions are also mapped across all the argument data, and so effectively
scale the velocity and update positions of all the agents.

Step Function
-------------

The step function defines how the state of the simulation is updated, it should
have the signature

.. code-block::

   step(i, k, params, state) -> (state, records)

where ``i`` is the current step number, ``k`` a JAX random key, ``params``
any parameters that are static over the simulation, and ``state`` the simulation
state. It should return a tuple containing the updated state, and any data to be recorded
over the course of the simulation.

For the boids model this looks like:

.. testcode:: hard_coded_boids

   def step(_i, k, params: Params, boids: Boid):
       n_nb, x_nb, v_nb, v_cl = observe(k, params, boids, boids, pos=boids.pos)

       vel = steering(
           k,
           params,
           (boids.pos, boids.vel, n_nb, x_nb, v_nb, v_cl)
       )
       vel = limit_speed(k, params, vel)
       pos = move(k, params, (boids.pos, vel))

       return Boid(pos=pos, vel=vel), pos

Each step the agents observe their neighbours, update and scale their velocities,
and update positions. It then returns the updates state, and the positions of the
agents are recorded at each step.

Initialise and Run
------------------

We can then initialise and run the simulation using JAX random sampling, and the
Esquilax ``sim_runner`` function

.. testcode:: hard_coded_boids

   def boids_sim(n: int, n_steps: int, show_progress: bool = True):
       k = jax.random.PRNGKey(101)
       k1, k2 = jax.random.split(k)

       pos = jax.random.uniform(k1, (n, 2))
       vel = 0.01 * jax.random.uniform(k2, (n, 2))
       boids = Boid(pos=pos, vel=vel)

       params = Params(
           cohesion=0.001,
           avoidance=0.05,
           alignment=0.05,
           max_speed=0.05,
           min_speed=0.01,
           close_range=0.02,
       )

       _, history, _ = esquilax.sim_runner(
           step, params, boids, n_steps, k, show_progress=show_progress
       )

       return history

   trajectories = boids_sim(
       5, 20, show_progress=False
   )
