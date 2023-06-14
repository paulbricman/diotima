from diotima.perceiver.optimize import *

import jax
import jax.numpy as jnp
import haiku as hk

import pytest


@pytest.fixture
def universe_config():
    return default_universe_config()


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


@pytest.fixture
def config():
    return UniverseDataConfig(
        steps=4,
        n_cfs=2,
        start=2,
        key=jax.random.PRNGKey(0)
    )


def test_synth_universe_data(config: UniverseDataConfig):
    universe_data = synth_universe_data(config)

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size


def test_data(config: UniverseDataConfig):
    n_univs = 2
    data = synth_data(config, n_univs=n_univs, key=jax.random.PRNGKey(0))

    assert data.atom_elems.shape[0] == n_univs * config.n_cfs


def test_forward(config: UniverseDataConfig):
    n_univs = 2
    data = synth_data(config, n_univs=n_univs, key=jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(42)

    transf_forward = hk.transform_with_state(raw_forward)
    params, state = transf_forward.init(rng, data, config, True)
    out, state = transf_forward.apply(params, state, rng, data, config, True)

    assert out.pred_locs_future.shape == (
        n_univs * config.n_cfs,
        config.steps - config.start,
        int(out.universe_config.n_atoms[0][0]),
        int(out.universe_config.n_dims[0][0])
    )
