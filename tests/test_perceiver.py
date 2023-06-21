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
def config(universe_config):
    return default_config(universe_config)


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


def test_hk_next_rng(config):
    first = next(config.rng)
    second = next(config.rng)

    assert not jnp.allclose(first, second)


def test_synth_universe_data(config):
    universe_data = synth_universe_data(config)

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size
    assert universe_data.locs_history.shape == (
        config.data.start,
        universe_data.universe_config.n_atoms,
        universe_data.universe_config.n_dims
    )


def test_synth_data(config):
    data = synth_data(config)

    assert data.atom_elems.shape[0] == config.data.n_univs


def test_raw_forward(config):
    data = synth_data(config)
    params, state, forward = init_opt(config)

    out, state = forward.apply(params, state, next(config.rng), data, config, True)

    print(out.universe_config)
    assert out.pred_locs_future.shape == (
        config.data.n_univs,
        (config.data.steps - config.data.start),
        int(out.universe_config.n_atoms[0]),
        config.optimization.branches,
        int(out.universe_config.n_dims[0])
    )


def test_loss(config):
    data = synth_data(config)
    params, state, forward = init_opt(config)

    optimizer = optax.adam(config.optimizer.lr)
    opt_state = optimizer.init(params)
    error = loss(params, state, opt_state, forward, data, config)

    assert error.size == 1


def test_backward(config):
    data = synth_data(config)

    params, state, forward = init_opt(config)
    optim = optax.adam(config.optimizer.lr)
    opt_state = optim.init(params)
    new_params, new_state, new_opt_state = backward(params, state, opt_state, forward, optim, data, config)

    assert params is params
    assert new_params is not params


def test_optimize(config):
    optimize(config)
