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


def test_validate_physics(universe_config: UniverseConfig):
    assert physics.valid_physics_config(
        universe_config.physics_config,
        universe_config.n_elems
    )


def test_fields(universe: Universe):
    fields = physics.compute_fields(
        universe.atom_locs[0],
        universe.atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    # For each element, how present it is, as informed by individual atoms.
    assert fields.matters.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_atoms,
    )

    # For each elem, how much is it attracted by each elem, as informed by individual atoms.
    assert fields.attractions.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_elems,
        universe.universe_config.n_atoms,
    )

    # For each elem, how much is it repelled by each elem, as informed by individual atoms.
    assert fields.repulsions.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_elems,
        universe.universe_config.n_atoms,
    )

    # For each elem, its energy informed by each elem, as informed by individual atoms.
    assert fields.energies.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_elems,
        universe.universe_config.n_atoms,
    )


def test_element_weighted_fields(universe: Universe):
    fields = physics.compute_element_weighted_fields(
        universe.atom_locs[0],
        universe.atom_elems[0],
        universe.atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    # Matter as informed by individual atoms.
    assert fields.matters.shape == (
        universe.universe_config.n_atoms,
    )

    # Attraction as informed by individual atoms.
    assert fields.attractions.shape == (
        universe.universe_config.n_atoms,
    )

    # Repulsion as informed by individual atoms.
    assert fields.repulsions.shape == (
        universe.universe_config.n_atoms,
    )

    # Energy as informed by individual atoms.
    assert fields.energies.shape == (
        universe.universe_config.n_atoms,
    )


def test_motion(universe: Universe):
    motions = physics.motion(
        universe.atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    # Velocity vector for each atom.
    assert motions.shape == (
        universe.universe_config.n_atoms,
        universe.universe_config.n_atoms,
        universe.universe_config.n_dims,
    )



