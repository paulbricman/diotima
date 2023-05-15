from diotima.world import UniverseConfig, Universe, seed, run
import diotima.physics as physics
import pytest
import jax.numpy as np
from copy import deepcopy
from jax import random


@pytest.fixture
def universe_config():
    return UniverseConfig()


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


def test_init_universe_config(universe_config: UniverseConfig):
    pass


def test_init_universe(universe: Universe):
    pass


def test_one_step(universe: Universe):
    motions = physics.motion(
        universe.atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    init_locs = universe.atom_locs
    universe = run(universe)
    final_locs = universe.atom_locs

    # Given the one step run default, this should hold.
    assert np.allclose(
        init_locs +
        universe.universe_config.dt *
        motions,
        final_locs)


def test_run(universe: Universe):
    parallel_universe = deepcopy(universe)
    universe1 = run(universe)
    universe2 = run(universe1)
    parallel_universe2 = run(parallel_universe, 2)

    assert not np.allclose(universe1.atom_locs, universe2.atom_locs)
    assert np.allclose(universe2.atom_locs, parallel_universe2.atom_locs)

    assert np.allclose(universe2.history, parallel_universe2.history)
    assert universe2.history.shape == (
        2,
        universe2.universe_config.n_atoms,
        universe2.universe_config.n_dims
    )
