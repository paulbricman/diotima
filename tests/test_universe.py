from diotima.world.universe import *
from diotima.world.physics import *

import jax.numpy as jnp

from copy import deepcopy
import pytest


@pytest.fixture
def universe_config():
    return default_universe_config()


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


def test_init_universe_config(universe_config: UniverseConfig):
    pass


def test_init_universe(universe: Universe):
    pass


def test_one_step(universe: Universe):
    motions = motion(universe.atom_locs, universe.atom_elems, universe.universe_config)

    init_locs = universe.atom_locs
    universe = run(universe)
    final_locs = universe.atom_locs

    # Given the one step run default, this should hold.
    assert jnp.allclose(init_locs + universe.universe_config.dt * motions, final_locs)


def test_run(universe: Universe):
    parallel_universe = deepcopy(universe)
    parallel_universe2 = run(parallel_universe, 2)
    universe1 = run(universe)
    universe2 = run(universe1)

    assert not jnp.allclose(universe1.atom_locs, universe2.atom_locs)
    assert jnp.allclose(universe2.atom_locs, parallel_universe2.atom_locs)

    assert jnp.allclose(universe2.locs_history, parallel_universe2.locs_history)
    assert universe2.locs_history.shape == (
        2,
        universe2.universe_config.n_atoms,
        universe2.universe_config.n_dims,
    )


def test_simple_run_with_adv_opt(universe: Universe):
    vanilla_universe = run(universe, 5)
    adv_opt_universe = run(universe, 5, False, BrownianOptimizer(jax.random.PRNGKey(0)))
    assert not jnp.allclose(vanilla_universe.atom_locs, adv_opt_universe.atom_locs)


def test_trim_rerun(universe: Universe):
    target = run(universe, 4)
    trimmed = trim(target, 2)
    retarget = run(trimmed, 2)

    assert jnp.allclose(target.atom_locs, retarget.atom_locs)
    assert jnp.allclose(target.locs_history, retarget.locs_history)
    assert jnp.allclose(target.jac_history, retarget.jac_history)
    assert target.step == retarget.step


def test_spawn_counterfactuals(universe: Universe):
    n_steps = 4
    n_cfs = 3

    no_adv_opt = run(universe, n_steps)
    cfs = spawn_counterfactuals(no_adv_opt, 2, n_cfs)

    assert jnp.allclose(cfs.step, jnp.array([n_steps] * n_cfs))
    assert not jnp.allclose(cfs.locs_history[0], cfs.locs_history[1])

    # In the beginning, there was only physics (no adv opt).
    assert jnp.allclose(cfs.locs_history[0][0], cfs.locs_history[1][0])
