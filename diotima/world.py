from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np
from jax import Array, jit, vmap, grad, random
from jax.lax import scan
import diotima.physics as physics


class UniverseConfig(NamedTuple):
    """
    Object containing universe configuration details.
    """
    n_elems: int = 2
    n_atoms: int = 3
    n_dims: int = 2
    dt: float = 0.1
    physics_config: physics.PhysicsConfig = physics.default_physics_config(n_elems)
    elem_distrib: Array = physics.default_elem_distrib(n_elems)


class Universe(NamedTuple):
    """
    Object holding universe state.
    """
    universe_config: UniverseConfig
    atom_locs: Array
    atom_elems: Array
    locs_history: Array = None
    motions_history: Array = None
    step: int = 0


def seed(universe_config: UniverseConfig, key: prng.PRNGKeyArray = random.PRNGKey(0)) -> Universe:
    """
    Seed universe (i.e. assign pseudorandom atom locations and elements).

    Args:
        universe_config: Universe configuration to use in seeding universe.
        key: PRNG key to use in determining atom locations and elements.

    Returns:
        Seeded universe object.
    """
    key_locs, key_elems = random.split(key, num=2)
    atom_locs = random.normal(key_locs, shape=(
        universe_config.n_atoms,
        universe_config.n_dims
    ))
    # TODO: Implement Gumbel-Softmax sampling based on elem_distrib in
    # universe_config
    atom_elems = random.uniform(key_elems, shape=(
        universe_config.n_atoms,
        universe_config.n_elems
    ))
    return Universe(
        universe_config,
        atom_locs,
        atom_elems
    )


def run(universe: Universe, n_steps: int = 1) -> Universe:
    """
    Run universe `n_steps` forward.

    Args:
        universe: Starting universe to run forward.
        n_steps: Number of steps to run universe forward.

    Returns:
        Update universe object.
    """
    def pure_step(state, _):
        return physics.step(
            state.locs,
            universe.atom_elems,
            universe.universe_config
        )

    last_state, state_history = scan(
        pure_step,
        physics.first_snapshot(
            universe.atom_locs,
            universe.universe_config
        ),
        None,
        n_steps
    )

    if universe.locs_history is not None:
        updated_locs_history = np.concatenate(
            (universe.locs_history,
             state_history.locs)
        )
    else:
        updated_locs_history = state_history.locs

    if universe.motions_history is not None:
        updated_motions_history = np.concatenate(
            (universe.motions_history,
             state_history.motions)
        )
    else:
        updated_motions_history = state_history.motions

    return Universe(
        universe.universe_config,
        last_state.locs,
        universe.atom_elems,
        updated_locs_history,
        updated_motions_history,
        universe.step + n_steps
    )
