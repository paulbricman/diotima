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
    n_atoms: int = 10
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
    history: Array = None
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
    def pure_step(atom_locs, _): return physics.step(
        atom_locs,
        universe.atom_elems,
        universe.universe_config
    )

    updated_locs, recent_history = scan(pure_step, universe.atom_locs, None, n_steps)
    if universe.history is not None:
        updated_history = np.concatenate(
            (universe.history,
            recent_history)
        )
    else:
        updated_history = recent_history

    return Universe(
        universe.universe_config,
        updated_locs,
        universe.atom_elems,
        updated_history,
        universe.step + n_steps
    )
