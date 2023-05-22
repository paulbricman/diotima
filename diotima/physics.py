from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np
from jax import Array, jit, vmap, grad, random, jacfwd, jacrev
from jax.lax import scan
from diotima.utils import norm, normalize
from einops import rearrange


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
        np.tile(1.15, (n_elems)),
        np.tile(0.6, (n_elems, n_elems)),
        np.tile(-1.5, (n_elems, n_elems)),
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
        ) * universe_config.physics_config.w_ks[elem_idx]

    return vmap(compute_matter_field)(
        np.arange(universe_config.n_elems))


def compute_attraction_fields(matters, universe_config):
    def send_attraction_fields(to_idx):
        def send_attraction_field(from_idx):
            return peak(
                matters[from_idx],
                universe_config.physics_config.mu_gs[from_idx][to_idx],
                universe_config.physics_config.sigma_gs[from_idx][to_idx]
            )

        # All attraction fields sent from element from_idx
        return vmap(send_attraction_field)(
            np.arange(universe_config.n_elems))

    # All attraction fields sent from all elements
    return vmap(send_attraction_fields)(
        np.arange(universe_config.n_elems))


def compute_repulsion_fields(distances, universe_config):
    def send_repulsion_fields(to_idx):
        def send_repulsion_field(from_idx):
            return universe_config.physics_config.c_reps[from_idx][to_idx] / 2 * (
                (1.0 - distances).clip(0.0) ** 2)

        # All repulsion fields sent from element from_idx
        return vmap(send_repulsion_field)(
            np.arange(universe_config.n_elems))

    # All repulsion fields sent from all elements
    return vmap(send_repulsion_fields)(
        np.arange(universe_config.n_elems))


def compute_fields(loc, atom_locs, universe_config):
    distances = norm(loc - atom_locs)
    matters = compute_matter_fields(distances, universe_config)
    attractions = compute_attraction_fields(matters, universe_config)
    repulsions = compute_repulsion_fields(distances, universe_config)
    energies = repulsions - attractions
    return Fields(matters, attractions, repulsions, energies)


def compute_element_weighted_fields(loc, elem, atom_locs, universe_config):
    unweighted_fields = compute_fields(loc, atom_locs, universe_config)
    matters = (unweighted_fields.matters.T * elem).T.sum(axis=0)
    attractions = (unweighted_fields.attractions.T * elem).T.sum(axis=(0, 1))
    repulsions = (unweighted_fields.repulsions.T * elem).T.sum(axis=(0, 1))
    energies = repulsions - attractions
    return Fields(matters, attractions, repulsions, energies)


def motion(atom_locs, atom_elems, universe_config):
    def compute_energy_gradients(from_idx):
        def compute_energy_gradient(loc, elem):
            energies = compute_element_weighted_fields(
                loc,
                elem,
                atom_locs,
                universe_config).energies

            return energies[from_idx]

        loc_to_grad = grad(compute_energy_gradient)
        return -vmap(loc_to_grad)(atom_locs, atom_elems)

    motions = vmap(compute_energy_gradients)(np.arange(universe_config.n_atoms))
    motions = rearrange(motions, "f t dims -> t f dims")
    return motions


class Snapshot(NamedTuple):
    locs: Array
    motions: Array
    jac: Array


def first_snapshot(atom_locs, universe_config):
    return Snapshot(
        atom_locs,
        np.zeros((
            universe_config.n_atoms,
            universe_config.n_atoms,
            universe_config.n_dims,
        )),
        np.zeros((
            universe_config.n_atoms,
            universe_config.n_atoms,
        ))
    )


def step(atom_locs, atom_elems, universe_config):
    motions = motion(
        atom_locs,
        atom_elems,
        universe_config
    )

    updated_locs = atom_locs + universe_config.dt * motions.sum(axis=1)

    # How much each atom's location influences each atom's location next,
    # aggregated across dimensions both for cause and effect.
    pure_al_grad = lambda als: (als + universe_config.dt * motion(
        als,
        atom_elems,
        universe_config
    ).sum(axis=(1))).sum(axis=1)
    jac = jacfwd(pure_al_grad)(atom_locs).sum(axis=2)

    state = Snapshot(updated_locs, motions, jac)
    return state, state
