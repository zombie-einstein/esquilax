Getting Started
===============

Installation
------------

Esquilax can be installed using pip

.. code-block:: bash

   pip install esquilax

The requirements for evolutionary and rl training are
not installed by default. They can be installed using the ``evo`` and ``rl``
extras respectively, e.g.:

.. code-block:: bash

   pip install esquilax[evo]

You may also need to manually install ``jaxlib``, for example
if you want GPU support. Instructions can be found
`here <https://github.com/google/jax?tab=readme-ov-file#installation>`_.
