from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np
from jax import Array, jit, vmap, grad, random
from jax.lax import scan


class PhysicsConfig(NamedTuple):
    """
    Object containing fundamental physical constants which parametrize physics.
    """
    mu_ks: Array
    sigma_ks: Array
    w_ks: Array

    mu_gs: Array
    sigma_gs: Array
    c_reps: Array


def default_physics_config(n_elems: int):
    return PhysicsConfig(
        np.tile(4.0, (n_elems)),
        np.tile(1.0, (n_elems)),
        np.tile(0.022, (n_elems)),
        np.tile(0.6, (n_elems, n_elems)),
        np.tile(0.15, (n_elems, n_elems)),
        np.tile(1.0, (n_elems, n_elems))
    )


def default_elem_distrib(n_elems: int):
    return np.ones((n_elems,)) / n_elems


def valid_physics_config(physics_config: PhysicsConfig, n_elems: int):
    elem_constants = [
        physics_config.mu_ks,
        physics_config.sigma_ks,
        physics_config.w_ks
    ]

    elem_elem_constants = [
        physics_config.mu_gs,
        physics_config.sigma_gs,
        physics_config.c_reps
    ]

    return all([e.shape == (n_elems,) for e in elem_constants]) and all([e.shape == (n_elems, n_elems,) for e in elem_elem_constants])


class Fields(NamedTuple):
    """
    Utility object containing linked field information.
    """
    matters: Array
    attractions: Array
    repulsions: Array
    energies: Array


def peak(x, mu, sigma):
    return np.exp(-((x - mu) / sigma) ** 2)


def compute_matter_fields(distances, universe_config):
    def compute_matter_field(elem_idx):
        return peak(
            distances,
            universe_config.physics_config.mu_ks[elem_idx],
            universe_config.physics_config.sigma_ks[elem_idx]
        ).sum() * universe_config.physics_config.w_ks[elem_idx]

    return vmap(compute_matter_field)(
        np.arange(universe_config.n_elems))


def compute_attraction_fields(matters, universe_config):
    def send_attraction_fields(from_idx):
        def send_attraction_field(to_idx):
            return peak(
                matters[from_idx],
                universe_config.physics_config.mu_gs[from_idx][to_idx],
                universe_config.physics_config.sigma_gs[from_idx][to_idx])

        # All attraction fields sent from element from_idx
        return vmap(send_attraction_field)(
            np.arange(universe_config.n_elems))

    # All attraction fields sent from all elements
    return vmap(send_attraction_fields)(
        np.arange(universe_config.n_elems))


def compute_repulsion_fields(distances, universe_config):
    def send_repulsion_fields(from_idx):
        def send_repulsion_field(to_idx):
            return universe_config.physics_config.c_reps[from_idx][to_idx] / 2 * (
                (1.0 - distances[from_idx]).clip(0.0) ** 2).sum()

        # All repulsion fields sent from element from_idx
        return vmap(send_repulsion_field)(
            np.arange(universe_config.n_elems))

    # All repulsion fields sent from all elements
    return vmap(send_repulsion_fields)(
        np.arange(universe_config.n_elems))


def compute_fields(loc, atom_locs, universe_config):
    distances = np.sqrt(np.square(loc - atom_locs).sum(-1).clip(1e-10))
    matters = compute_matter_fields(distances, universe_config)
    attractions = compute_attraction_fields(matters, universe_config)
    repulsions = compute_repulsion_fields(distances, universe_config)
    energies = repulsions - attractions
    return Fields(matters, attractions, repulsions, energies)


def compute_element_weighted_fields(loc, elem, atom_locs, universe_config):
    unweighted_fields = compute_fields(loc, atom_locs, universe_config)
    matters = np.dot(unweighted_fields.matters, elem)
    attractions = np.dot(unweighted_fields.attractions.sum(axis=1), elem)
    repulsions = np.dot(unweighted_fields.repulsions.sum(axis=1), elem)
    energies = repulsions - attractions
    return Fields(matters, attractions, repulsions, energies)


def motion(atom_locs, atom_elems, universe_config):
    grad_energies = grad(
        lambda loc,
        elem: compute_element_weighted_fields(
            loc,
            elem,
            atom_locs,
            universe_config).energies)
    return -vmap(grad_energies)(atom_locs, atom_elems)


def step(atom_locs, atom_elems, universe_config):
    updated_locs = atom_locs + universe_config.dt * motion(
        atom_locs,
        atom_elems,
        universe_config
    )
    return updated_locs, updated_locs
