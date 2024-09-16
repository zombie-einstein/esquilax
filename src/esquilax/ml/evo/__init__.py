"""
Policy training patterns for use with `Evosax`_ neuro-evolution package

Base-classes and functionality for training of evolutionary
policies (via `Evosax`_) using esquilax as a multi-agent
(and potentially multi-policy) training environment.

.. _Evosax: https://github.com/RobertTLange/evosax
"""
from .strategy import BasicStrategy, Strategy
from .training import TrainingData, test, train
