********
Esquilax
********

**JAX multi-agent system simulation toolset**

Esquilax is a set of utilities and transformations
implementing common patterns seen in multi-agent systems,
allowing developers and researchers to quickly
build models without the need to re-implement or optimise
low level algorithms.

Esquilax is mainly intended for multi-agent RL,
neuro-evolution and alife use-cases, and can be
used alongside other JAX libraries like
`Flax <https://github.com/google/flax>`_,
`Evosax <https://github.com/RobertTLange/evosax>`_, and
`RLax <https://github.com/google-deepmind/rlax>`_.

**Features**

- Built on top of JAX, allowing for high performance
  and GPU support from JIT compiled Python.

- Interoperability with existing JAX ML, RL, and
  neuro-evolution libraries. Also works alongside
  the broader Python scientific/numerical ecosystem.

- Functional paradigm allows models to be easily
  combined and re-used.

- Performant implementations of common multi-agent
  patterns.

.. toctree::
   :caption: Contents:
   :maxdepth: 2
   :includehidden:

   getting_started
   user_guide
   sharp_bits
   examples
