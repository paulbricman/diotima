from diotima.perceiver.optimize import *
from diotima.world.universe import *
from diotima.world.physics import *

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jax.flatten_util import ravel_pytree

import pytest
from safetensors.flax import load_file
import os


flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = flags + " --xla_force_host_platform_device_count=2"


@pytest.fixture
def config():
    config = default_config()
    config = default_config(
        default_physics_config(config["data"]["universe_config"]["n_elems"]),
        default_elem_distrib(config["data"]["universe_config"]["n_elems"]),
    )
    return config


def test_synth_universe_data(config):
    universe_data = synth_universe_data(config, jax.random.PRNGKey(0))

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size
    assert universe_data.locs_history.shape == (
        config["data"]["start"],
        config["data"]["universe_config"]["n_atoms"],
        config["data"]["universe_config"]["n_dims"],
    )
    assert universe_data.locs_future.shape == (
        config["data"]["n_cfs"],
        config["data"]["steps"] - config["data"]["start"],
        config["data"]["universe_config"]["n_atoms"],
        config["data"]["universe_config"]["n_dims"],
    )


def test_synth_data(config):
    data = synth_data(config, jax.random.PRNGKey(0))

    assert data.atom_elems.shape[0] == config["data"]["n_univs"]
    assert data.locs_future.shape == (
        config["data"]["n_univs"],
        config["data"]["n_cfs"],
        config["data"]["steps"] - config["data"]["start"],
        config["data"]["universe_config"]["n_atoms"],
        config["data"]["universe_config"]["n_dims"],
    )


def test_raw_forward(config):
    data = synth_data(config, jax.random.PRNGKey(0))
    params, opt_state, optim, forward = init_opt(config, jax.random.PRNGKey(0))

    out = forward.apply(params, jax.random.PRNGKey(0), data, config, True)
    data, agents = out

    assert data.pred_locs_future.shape == (
        config["data"]["n_univs"],
        config["optimize_perceiver"]["branches"],
        config["data"]["steps"] - config["data"]["start"],
        config["data"]["universe_config"]["n_atoms"],
        config["data"]["universe_config"]["n_dims"],
    )

    assert agents.substrates.shape == (
        config["data"]["n_univs"],
        config["data"]["start"] * config["data"]["universe_config"]["n_atoms"],
        config["encoder"]["z_index_dim"],
    )

    assert agents.flows.shape == (
        config["data"]["n_univs"],
        config["optimize_perceiver"]["branches"],
        config["encoder"]["z_index_dim"],
        config["data"]["universe_config"]["n_atoms"],
        config["data"]["universe_config"]["n_dims"],
    )


def test_distance():
    cfs0 = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 2, 1, 1)) * 1e-6
    bs0 = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 2, 1, 1)) * 1e-6

    assert jnp.isclose(distance(cfs0, bs0), 0, atol=1e-4, rtol=1e-3)

    bs1 = jnp.ones((1, 2, 2, 1, 1))

    assert distance(cfs0, bs0) < distance(cfs0, bs1)
    assert distance(cfs0, bs1) == distance(bs1, cfs0)


def test_loss(config):
    data = synth_data(config, jax.random.PRNGKey(0))
    params, opt_state, optim, forward = init_opt(config, jax.random.PRNGKey(0))
    error = loss(params, forward, data, config, jax.random.PRNGKey(0))

    assert error.size == 1


def test_optimize_perceiver(config):
    params, opt_state, optim, forward = init_opt(config, jax.random.PRNGKey(0))

    carry, history = optimize_perceiver(
        config, params, opt_state, optim, forward, jax.random.PRNGKey(0)
    )
    params, opt_state, epoch, perceiver_loss = carry
    assert epoch == config["optimize_perceiver"]["epochs"]


def test_optimize_universe_config(config):
    new_config = optimize_universe_config(config, jax.random.PRNGKey(0))
    assert not pytrees_equal(
        config["data"]["universe_config"]["physics_config"],
        new_config["data"]["universe_config"]["physics_config"],
    )


def pytrees_equal(tree1, tree2):
    tree1, unravel = jax.flatten_util.ravel_pytree(tree1)
    tree2, unravel = jax.flatten_util.ravel_pytree(tree2)
    return jnp.allclose(tree1, tree2, atol=1e-10, rtol=1e-10)


def test_minimal_nested():
    def inner_loop(x):
        return jax.lax.scan(lambda y, _: [y + 1] * 2, x, None, 2)[0]

    def outer_loop(x):
        return jax.lax.scan(lambda y, _: [inner_loop(y)] * 2, x, None, 3)[0]

    assert outer_loop(0) == 6
