from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np
from jax import Array, jit, vmap, grad, random
from jax.lax import scan
import diotima.physics as physics


class UniverseConfig():
    """
    Object containing universe configuration details.
    """
    def __init__(
            self,
            n_elems: int = 3,
            n_atoms: int = 2,
            n_dims: int = 2,
            dt: float = 0.1,
            physics_config: Array = None,
            elem_distrib: Array = None,
    ):
        self.n_elems = n_elems
        self.n_atoms = n_atoms
        self.n_dims = n_dims
        self.dt = dt

        if physics_config:
            self.physics_config = physics_config
        else:
            self.physics_config = physics.default_physics_config(n_elems)

        if elem_distrib:
            self.elem_distrib = elem_distrib
        else:
            self.elem_distrib = physics.default_elem_distrib(n_elems)


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


def run(universe: Universe, n_steps: int = 1, get_jac: bool = False) -> Universe:
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
        return physics.step(
            state.locs,
            universe.atom_elems,
            universe.universe_config,
            get_jac
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
        updated_motions_history = np.concatenate(
            (universe.motions_history,
             state_history.motions)
        )
        updated_jac_history = np.concatenate(
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
