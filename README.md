# Diotima [WIP]

This is a testbed for experiments on a differentiable pipeline for measuring agency. An interactive report that documents this is in the works, and will eventually make for a new installment of [a previous series focused on operationalizing philosophy](https://compphil.github.io/).

This codebase follows a functional paradigm so as to facilitate parallelization, and is organized as follows:

- **`diotima.world`**, a differentiable particle-based system that extends [Particle Lenia](https://google-research.github.io/self-organising-systems/particle-lenia/) with multiple elements and optimized rendering. Particle Lenia itself extends [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) with continuous space, time, and energy, as well as with a differentiable update function (in contrast to its predecessor).
- **`diotima.perceiver`**, a differentiable pipeline for quantifying agency built around the [Perceiver IO](https://deepmind.google/discover/blog/building-architectures-that-can-handle-the-worlds-data/) architecture. It is used to model the evolution of particles that make up the microverse described above. The "bounded Laplace's demon" from the report.

The goal of this work is to optimize the "fundamental physical constants" governing the Diotima microverse so as to yield increasing readings of agency. The fact that both components are differentiable is meant to help make traditional gradient descent workable in this setting.

Agency is operationalized as _the sophisticated influence on the world in the face of adversarial optimization_. The sophistication of a signal, in turn, is operationalized as _the minimum effective parameter count required to model a signal with infinitesimal epistemic uncertainty_. The simplest form of adversarial optimization is simply heat: increasing entropy is synonymous with avoiding lurking around particular states.

Putting everything together, the proposed operationalization of agency roughly works by first meddling with the world in random ways, before measuring the amount of structure present in the world's response to that.

## Acknowledgements

This work has been made possibled by access to the [TPU Research Cloud](https://sites.research.google/trc/about/).
