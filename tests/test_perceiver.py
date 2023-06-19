from diotima.perceiver.optimize import *

import jax
import jax.numpy as jnp
import haiku as hk
import optax

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
        rng=hk.PRNGSequence(jax.random.PRNGKey(0))
    )


def test_hk_next_rng(config):
    first = next(config.rng)
    second = next(config.rng)

    assert not jnp.allclose(first, second)


def test_synth_universe_data(config):
    universe_data = synth_universe_data(config)

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size


def test_synth_data(config):
    n_univs = 2
    data = synth_data(config, n_univs)

    assert data.atom_elems.shape[0] == n_univs * config.n_cfs


def test_forward(config):
    n_univs = 2
    data = synth_data(config, n_univs=n_univs)
    params, state, forward = init_opt(config)

    out, state = forward.apply(params, state, next(config.rng), data, config, True)

    assert out.pred_locs_future.shape == (
        n_univs * config.n_cfs,
        (config.steps - config.start) * int(out.universe_config.n_atoms[0][0]),
        int(out.universe_config.n_dims[0][0])
    )


def test_loss(config: UniverseDataConfig):
    n_univs = 2
    data = synth_data(config, n_univs=n_univs)
    params, state, forward = init_opt(config)

    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)
    error = loss(params, state, opt_state, forward, next(config.rng), data, config)

    assert error.size == 1


def test_backward(config: UniverseDataConfig):
    n_univs = 2
    data = synth_data(config, n_univs=n_univs)

    params, state, forward = init_opt(config)
    optim = optax.adam(1e-4)
    opt_state = optim.init(params)
    new_params, new_state, new_opt_state = backward(params, state, opt_state, forward, next(config.rng), optim, data, config)

    assert params is params
    assert new_params is not params


def test_optimize(config: UniverseDataConfig):
    optimize(config)
