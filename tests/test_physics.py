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
    fields = physics.fields(
        universe.atom_locs[0],
        universe.atom_locs,
        universe.universe_config
    )

    # For each element, how present it is.
    assert fields.matters.shape == (
        universe.universe_config.n_elems,
    )

    # For each element, how much is it attracted by each element.
    assert fields.attractions.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_elems,
    )

    # For each element, how much is it repelled by each element.
    assert fields.repulsions.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_elems,
    )

    # For each element, its energy informed by each element.
    assert fields.energies.shape == (
        universe.universe_config.n_elems,
        universe.universe_config.n_elems,
    )


def test_element_weighted_fields(universe: Universe):
    fields = physics.element_weighted_fields(
        universe.atom_locs[0],
        universe.atom_elems[0],
        universe.atom_locs,
        universe.universe_config
    )

    # How present the element mixture is.
    assert fields.matters.size == 1

    # How attracted the atom is, element mixture considered.
    assert fields.attractions.size == 1

    # How repelled the atom is, element mixture considered.
    assert fields.repulsions.size == 1

    # The energy of the atom, element mixture considered.
    assert fields.energies.size == 1


def test_motion(universe: Universe):
    motions = physics.motion(
        universe.atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    # Velocity vector for each atom.
    assert motions.shape == (
        universe.universe_config.n_atoms,
        universe.universe_config.n_dims,
    )
