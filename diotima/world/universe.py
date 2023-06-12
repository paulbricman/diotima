import diotima.world.physics as physics

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
from jax import Array

from typing import NamedTuple
from collections import namedtuple


class UniverseConfig(NamedTuple):
    """
    Object containing universe configuration details.
    """
    n_elems: int
    n_atoms: int
    n_dims: int
    dt: float
    physics_config: Array
    elem_distrib: Array


def default_universe_config():
    n_elems = 3

    return UniverseConfig(
        n_elems,
        n_atoms=2,
        n_dims=2,
        dt=0.1,
        physics_config=physics.default_physics_config(n_elems),
        elem_distrib=physics.default_elem_distrib(n_elems)
    )


class Universe(NamedTuple):
    """
    Object holding universe state.
    """
    universe_config: UniverseConfig
    atom_locs: Array
    atom_elems: Array
    locs_history: Array = None
    motions_history: Array = None
    jac_history: Array = None
    step: int = 0


def seed(universe_config: UniverseConfig,
         key: PRNGKeyArray = jax.random.PRNGKey(0)) -> Universe:
    """
    Seed universe (i.e. assign pseudorandom atom locations and elements).

    Args:
        universe_config: Universe configuration to use in seeding universe.
        key: PRNG key to use in determining atom locations and elements.

    Returns:
        Seeded universe object.
    """
    key_locs, key_elems = jax.random.split(key, num=2)
    atom_locs = jax.random.normal(key_locs, shape=(
        universe_config.n_atoms,
        universe_config.n_dims
    ))
    atom_elems = physics.elem_distrib_to_elems(
        universe_config.n_atoms,
        universe_config.n_elems,
        universe_config.elem_distrib,
        key_elems
    )

    return Universe(
        universe_config,
        atom_locs,
        atom_elems
    )


def run(universe: Universe, n_steps: int = 1,
        get_jac: bool = False, init_adv_opt = None) -> Universe:
    """
    Run universe `n_steps` forward.

    Args:
        universe: Starting universe to run forward.
        n_steps: Number of steps to run universe forward.
        get_jac: Whether to also compute the grad-based causal graph.

    Returns:
        Update universe object.
    """
    def pure_step(state, _):
        snapshot, adv_opt = state
        new_snapshot = physics.step(
            snapshot.locs,
            universe.atom_elems,
            universe.universe_config,
            get_jac
        )

        if adv_opt:
            key, subkey = jax.random.split(adv_opt.key, num=2)
            delta = jax.random.normal(subkey, shape=(
                universe.universe_config.n_atoms,
                universe.universe_config.n_dims
            ))
            new_snapshot = physics.Snapshot(
                new_snapshot.locs + delta,
                new_snapshot.motions,
                new_snapshot.jac
            )
            adv_opt = BrownianOptimizer(key)

        state = new_snapshot, adv_opt
        return state, state

    last_state, state_history = jax.lax.scan(
        pure_step,
        (physics.first_snapshot(
            universe.atom_locs,
            universe.universe_config
        ), init_adv_opt),
        None,
        n_steps
    )
    last_state = last_state[0]
    state_history = state_history[0]

    if universe.locs_history is not None:
        updated_locs_history = jnp.concatenate(
            (universe.locs_history,
             state_history.locs)
        )
        updated_motions_history = jnp.concatenate(
            (universe.motions_history,
             state_history.motions)
        )
        updated_jac_history = jnp.concatenate(
            (universe.jac_history,
             state_history.jac))
    else:
        updated_locs_history = state_history.locs
        updated_motions_history = state_history.motions
        updated_jac_history = state_history.jac

    return Universe(
        universe.universe_config,
        last_state.locs,
        universe.atom_elems,
        updated_locs_history,
        updated_motions_history,
        updated_jac_history,
        universe.step + n_steps
    )


class BrownianOptimizer(NamedTuple):
    key: PRNGKeyArray


def spawn_counterfactuals(
        universe: Universe,
        start: int,
        n_cfs: int,
        key: PRNGKeyArray = jax.random.PRNGKey(0),
):
    """
    Instantiate new universes based on specified one, by adversarially optimizing from `start` into `n_cfs` counterfactuals.
    """
    assert start >= 0 and start < universe.step

    # Isolate common thread.
    common_thread = trim(universe, start)

    # Split into n_cfs keys.
    keys = jax.random.split(key, num=n_cfs)

    # Run universes forward using adversarial optimizers
    spawn_counterfactual = lambda key: run(
        common_thread,
        universe.step - start,
        False,
        BrownianOptimizer(key)
    )
    counterfactuals = jax.vmap(spawn_counterfactual)(keys)

    return counterfactuals


def trim(
        universe: Universe,
        until: int,
):
    """
    Given universe, return universe as if only until `until` timestep.
    """
    assert until < universe.step

    return Universe(
        universe.universe_config,
        universe.locs_history[until - 1],
        universe.atom_elems,
        universe.locs_history[:until],
        universe.motions_history[:until],
        universe.jac_history[:until],
        until
    )