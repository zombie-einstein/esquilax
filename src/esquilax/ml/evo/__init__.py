"""
Policy training patterns for use with `Evosax`_ neuro-evolution package

Classes definitions and functionality for training of evolutionary
policies (via `Evosax`_) using esquilax as a multi-agent
(and potentially multi-policy) training environment.

.. _Evosax: https://github.com/RobertTLange/evosax

.. note::

   This module requires additional dependencies that can be installed
   using the installation extra ``evo``, for example with pip using

   .. code::

      pip install esquilax[evo]
"""
try:
    import evosax  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Additional requirements are required for this functionality. "
        "They can be installed with the extra `pip install esquilax[evo]`"
    ) from e
from . import tree_utils
from .strategy import BasicStrategy, Strategy
from .training import test, train
from .types import TrainingData
