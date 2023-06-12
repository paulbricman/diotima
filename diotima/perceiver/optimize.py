import diotima.world.universe as default_universe_config, seed, run, trim, spawn_counterfactuals

import jax
from jax._src.prng import PRNGKeyArray
from jax import Array

from einops import repeat


class Data(NamedTuple):
    atom_elems: Array
    locs_history: Array
    locs_future: Array


class UniverseDataConfig(NamedTuple):
    steps: int
    num_cfs: int
    start: int
    key: PRNGKeyArray


def synth_universe_data(config: DataConfig, key: PRNGKeyArray):
    keys = jax.random.split(key, num=2)

    universe_config = universe.default_universe_config()
    universe = seed(universe, keys[0])
    universe = run(universe, config.steps)
    return spawn_counterfactuals(
        universe,
        config.start,
        config.num_cfs,
        keys[1]
    )


def synth_data(config: UniverseDataConfig, num_universes: int, key: PRNGKeyArray):
    keys = jax.random.split(key, num=num_universes)

    pure_synth_universe_data = lambda key: synth_universe_data(config, key)
    return jax.vmap(pure_synth_universe_data)(keys)


def init_perceiver():
    # Load weights object from params.

    # Return them.


def forward():
    # Given inputs, apply perceiver to get outputs.

    # Return them.


def backward():
    # Given true and predicted outputs, update weights.

    # Return them.


def optimize():
    # Repeatedly run forward and backward.

    # Return final weights.


def orchestrate():
    # Synthesize data.

    # Load weights.

    # Optimize them.

    # Return them, along with optimization history.

