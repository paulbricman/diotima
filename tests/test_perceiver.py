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
        key=jax.random.PRNGKey(0)
    )


@pytest.fixture
def experiment(config):
    return Experiment(
        None,
        hk.PRNGSequence(jax.random.PRNGKey(42)),
        config
    )


def test_experiment(experiment):
    pass


def test_synth_universe_data(experiment):
    universe_data = experiment.synth_universe_data()

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size


def test_data(experiment):
    n_univs = 2
    data = experiment.synth_data(n_univs)

    assert data.atom_elems.shape[0] == n_univs * experiment._config.n_cfs


def test_forward(experiment):
    n_univs = 2
    out, state = experiment.forward.apply(
        experiment._params,
        experiment._state,
        next(experiment._rng),
        experiment.synth_data(n_univs),
        experiment._config,
        True
    )

    assert out.pred_locs_future.shape == (
        n_univs * experiment._config.n_cfs,
        (experiment._config.steps - experiment._config.start) * int(out.universe_config.n_atoms[0][0]),
        int(out.universe_config.n_dims[0][0])
    )


def test_loss(experiment):
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(experiment._params)
    error = experiment.loss(
        experiment._params,
        experiment._state,
        opt_state,
        experiment.synth_data(2),
        experiment._config
    )

    assert error.size == 1


def test_backward(experiment):
    optim = optax.adam(1e-4)
    opt_state = optim.init(experiment._params)
    new_params, new_state, new_opt_state = experiment.backward(
        experiment._params,
        experiment._state,
        opt_state,
        optim,
        experiment.synth_data(2),
        experiment._config
    )

    assert experiment._params is experiment._params
    assert new_params is not experiment._params
