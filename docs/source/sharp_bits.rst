Sharp-Bits
==========

JAX Constraints
---------------

Esquilax makes use of many features of JAX, and hence also
inherits many of its constraints, including:

- Functional programming patterns and pure functions
- Method of random number generation
- JIT compilation, and compile time requirements

If you are unfamiliar with these aspects of JAX, the
`JAX docs <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-the-sharp-bits>`_
have a very useful section on these aspects.

Unsupported Features
--------------------

Continuous Time
^^^^^^^^^^^^^^^

Esquilax is intended for step-based simulations where
the state of the simulation is updated at each step/fixed interval.
As such it is not suitable for discrete event simulation where
events are ordered and can occur at continuous time intervals.

Ordered Interactions
^^^^^^^^^^^^^^^^^^^^

Esquilax is designed around agents that update in parallel,
this especially allows models to be deployed on GPUs for large
performance gains. It is possible to implement behaviours
that depend on ordering (using JAX
`scans <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`_
or
`loops <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop>`_)
at the cost of performance at scale.
