from diotima.world.universe import *
from diotima.world.physics import *

import pytest


@pytest.fixture
def universe_config():
    return default_universe_config()


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


def test_validate_physics(universe_config: UniverseConfig):
    assert valid_physics_config(universe_config.physics_config, universe_config.n_elems)


def test_fields(universe: Universe, universe_config: UniverseConfig):
    fields = compute_fields(
        universe.atom_locs[0],
        universe.atom_locs,
        universe.atom_elems,
        universe_config,
    )

    # For each element, how present it is, as informed by individual atoms.
    assert fields.matters.shape == (
        universe_config.n_elems,
        universe_config.n_atoms,
    )

    # For each elem, how much is it attracted by each elem, as informed by
    # individual atoms.
    assert fields.attractions.shape == (
        universe_config.n_elems,
        universe_config.n_elems,
        universe_config.n_atoms,
    )

    # For each elem, how much is it repelled by each elem, as informed by
    # individual atoms.
    assert fields.repulsions.shape == (
        universe_config.n_elems,
        universe_config.n_elems,
        universe_config.n_atoms,
    )

    # For each elem, its energy informed by each elem, as informed by
    # individual atoms.
    assert fields.energies.shape == (
        universe_config.n_elems,
        universe_config.n_elems,
        universe_config.n_atoms,
    )


def test_element_weighted_fields(universe: Universe, universe_config: UniverseConfig):
    fields = compute_element_weighted_fields(
        universe.atom_locs[0],
        universe.atom_elems[0],
        universe.atom_locs,
        universe.atom_elems,
        universe_config,
    )

    # Matter as informed by individual atoms.
    assert fields.matters.shape == (universe_config.n_atoms,)

    # Attraction as informed by individual atoms.
    assert fields.attractions.shape == (universe_config.n_atoms,)

    # Repulsion as informed by individual atoms.
    assert fields.repulsions.shape == (universe_config.n_atoms,)

    # Energy as informed by individual atoms.
    assert fields.energies.shape == (universe_config.n_atoms,)


def test_motion(universe: Universe, universe_config: UniverseConfig):
    motions = motion(universe.atom_locs, universe.atom_elems, universe_config)

    # Velocity vector for each atom.
    assert motions.shape == (
        universe_config.n_atoms,
        universe_config.n_dims,
    )
