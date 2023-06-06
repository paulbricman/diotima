from diotima.world.universe import *
from diotima.world.physics import *

import jax.numpy as jnp

from copy import deepcopy
import pytest


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
    motions = motion(
        universe.atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    init_locs = universe.atom_locs
    universe = run(universe)
    final_locs = universe.atom_locs

    # Given the one step run default, this should hold.
    assert jnp.allclose(
        init_locs +
        universe.universe_config.dt *
        motions.sum(axis=1),
        final_locs
    )


def test_run(universe: Universe):
    parallel_universe = deepcopy(universe)
    parallel_universe2 = run(parallel_universe, 2)
    universe1 = run(universe)
    universe2 = run(universe1)

    assert not jnp.allclose(universe1.atom_locs, universe2.atom_locs)
    assert jnp.allclose(universe2.atom_locs, parallel_universe2.atom_locs)

    assert jnp.allclose(
        universe2.locs_history,
        parallel_universe2.locs_history)
    assert universe2.locs_history.shape == (
        2,
        universe2.universe_config.n_atoms,
        universe2.universe_config.n_dims
    )
    assert universe2.motions_history.shape == (
        2,
        universe2.universe_config.n_atoms,
        universe2.universe_config.n_atoms,
        universe2.universe_config.n_dims
    )
