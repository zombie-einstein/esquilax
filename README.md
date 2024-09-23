<div align="center">
  <img src="https://github.com/zombie-einstein/esquilax/raw/main/.github/images/text_logo.png" />
  <br>
  <em>JAX Multi-Agent RL, A-Life, and Simulation Framework</em>
</div>
<br>

Esquilax is set of transformations and utilities
intended to allow developers and researchers to
quickly implement models of multi-agent systems
for rl-training, evolutionary methods, and a-life.

It is intended for systems involving large number of
agents, and to work alongside other JAX packages
like [Flax](https://github.com/google/flax) and
[Evosax](https://github.com/RobertTLange/evosax).

**Full documentation can be found
[here](https://zombie-einstein.github.io/esquilax/)**

## Features

- ***Built on top of JAX***

  This has the benefits of JAX; high-performance, built in
  GPU support etc., but also means Esquilax can interoperate
  with existing JAX ML and RL libraries.

- ***Interaction Algorithm Implementations***

  Implements common agent interaction patterns. This
  allows users to concentrate on model design instead of low-level
  algorithm implementation details.

- ***Scale and Performance***

  JIT compilation and GPU support enables simulations and multi-agent
  systems containing large numbers of agents whilst maintaining
  performance and training throughput.

- ***Functional Patterns***

  Esquilax is designed around functional patterns, ensuring models
  can be readily parallelised, but also aiding composition
  and readability

- ***Built-in RL and Evolutionary Training***

  Esquilax provides functionality for running multi-agent RL
  and multi-strategy neuro-evolution training, within Esquilax
  simulations.

## Should I Use Esquilax?

Esquilax is intended for time-stepped models of large scale systems
with fixed numbers of entities, where state is updated in parallel.
As such you should probably *not* use Esquilax if:

- You want to use something other than stepped updates, e.g.
  continuous time, event driven models, or where agents are intended to
  update in sequence.
- You need variable numbers of entities or temporary entities, e.g.
  message passing.
- You need a high-fidelity physics/robotics simulation.

## Getting Started

Esquilax can be installed from pip using

``` bash
pip install esquilax
```

You may need to manually install JAXlib, especially for GPU support.
Installation instructions for JAX can be found
[here](https://github.com/google/jax?tab=readme-ov-file#installation).

## Examples

Example models and multi-agent policy training implemented using Esquilax
can be found [here](https://github.com/zombie-einstein/esquilax/tree/main/examples).

For a larger project using Esquilax see this
[Boid flock RL environment](https://github.com/zombie-einstein/flock_env).

## Contributing

### Issues

Please report any issues or feature suggestions
[here](https://github.com/zombie-einstein/esquilax/issues).

### Developers

Developer notes can be found
[here](https://github.com/zombie-einstein/esquilax/blob/main/.github/docs/developers.md),
Esquilax is under active development and contributions are very welcome!
