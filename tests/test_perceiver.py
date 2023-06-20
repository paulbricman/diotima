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
def perceiver_config(universe_config):
    return default_perceiver_config(universe_config)


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


@pytest.fixture
def universe_data_config():
    return UniverseDataConfig(
        steps=4,
        n_cfs=2,
        start=2,
        rng=hk.PRNGSequence(jax.random.PRNGKey(0))
    )


def test_hk_next_rng(universe_data_config):
    first = next(universe_data_config.rng)
    second = next(universe_data_config.rng)

    assert not jnp.allclose(first, second)


def test_synth_universe_data(universe_data_config):
    universe_data = synth_universe_data(universe_data_config)

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size


def test_synth_data(universe_data_config, perceiver_config):
    data = synth_data(universe_data_config, perceiver_config)

    assert data.atom_elems.shape[0] == perceiver_config.data.n_univs * universe_data_config.n_cfs


def test_forward(universe_data_config, perceiver_config):
    data = synth_data(universe_data_config, perceiver_config)
    params, state, forward = init_opt(universe_data_config, perceiver_config)

    out, state = forward.apply(params, state, next(universe_data_config.rng), data, universe_data_config, perceiver_config, True)

    assert out.pred_locs_future.shape == (
        perceiver_config.data.n_univs * universe_data_config.n_cfs,
        (universe_data_config.steps - universe_data_config.start) * int(out.universe_config.n_atoms[0][0]),
        int(out.universe_config.n_dims[0][0])
    )


def test_loss(universe_data_config, perceiver_config):
    data = synth_data(universe_data_config, perceiver_config)
    params, state, forward = init_opt(universe_data_config, perceiver_config)

    optimizer = optax.adam(perceiver_config.optimizer.lr)
    opt_state = optimizer.init(params)
    error = loss(params, state, opt_state, forward, data, universe_data_config, perceiver_config)

    assert error.size == 1


def test_backward(universe_data_config, perceiver_config):
    data = synth_data(universe_data_config, perceiver_config)

    params, state, forward = init_opt(universe_data_config, perceiver_config)
    optim = optax.adam(perceiver_config.optimizer.lr)
    opt_state = optim.init(params)
    new_params, new_state, new_opt_state = backward(params, state, opt_state, forward, optim, data, universe_data_config, perceiver_config)

    assert params is params
    assert new_params is not params


def test_optimize(universe_data_config, perceiver_config):
    optimize(universe_data_config, perceiver_config)
