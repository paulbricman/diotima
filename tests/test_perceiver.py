from diotima.perceiver.optimize import *
from diotima.world.universe import *
from diotima.world.physics import *

import jax
import jax.numpy as jnp
import haiku as hk
import optax

import pytest
from safetensors.flax import load_file


@pytest.fixture
def config():
    return default_config(default_physics_config(2), default_elem_distrib(2))


def test_hk_next_rng(config):
    first = next(config["rng"])
    second = next(config["rng"])

    assert not jnp.allclose(first, second)


def test_synth_universe_data(config):
    universe_data = synth_universe_data(config)

    assert universe_data.atom_elems[0].size == universe_data.atom_elems[1].size
    assert universe_data.locs_history.shape == (
        config["data"]["start"],
        universe_data.universe_config.n_atoms,
        universe_data.universe_config.n_dims
    )


def test_synth_data(config):
    data = synth_data(config)

    assert data.atom_elems.shape[0] == config["data"]["n_univs"]


def test_raw_forward(config):
    data = synth_data(config)
    params, state, opt_state, optim, forward = init_opt(config)

    out, state = forward.apply(
        params, state, next(config["rng"]), data, config, True)
    data, agents = out

    assert data.pred_locs_future.shape == (
        config["data"]["n_univs"],
        (config["data"]["steps"] - config["data"]["start"]),
        config["data"]["universe_config"]["n_atoms"],
        config["optimize_perceiver"]["branches"],
        config["data"]["universe_config"]["n_dims"]
    )

    assert agents.substrates.shape == (
        config["data"]["n_univs"],
        config["data"]["start"] * config["data"]["universe_config"]["n_atoms"],
        config["encoder"]["z_index_dim"]
    )

    assert agents.flows.shape == (
        config["data"]["n_univs"],
        config["optimize_perceiver"]["branches"],
        config["encoder"]["z_index_dim"],
        config["data"]["universe_config"]["n_atoms"],
        config["data"]["universe_config"]["n_dims"]
    )


def test_distance():
    cfs0 = jnp.array([[[0], [0], [0]], [[10], [10], [10]]])
    bs0 = jnp.array([[[0], [0]], [[10], [10]]])

    assert jnp.isclose(distance(cfs0, bs0), 0)

    bs1 = jnp.array([[[0], [1]], [[10], [20]]])
    bs2 = jnp.array([[[1], [0]], [[20], [10]]])

    assert distance(cfs0, bs0) < distance(cfs0, bs1)
    assert distance(cfs0, bs1) == distance(cfs0, bs2)
    assert distance(cfs0, bs1) == distance(bs1, cfs0)


def test_loss(config):
    data = synth_data(config)
    params, state, opt_state, optim, forward = init_opt(config)
    error, new_state = loss(params, state, opt_state, forward, data, config)

    assert error.size == 1


def test_optimize_perceiver(config):
    params, state, opt_state, optim, forward = init_opt(config)

    state = optimize_perceiver(
        config, params, state, opt_state, optim, forward)
    params, state, opt_state, epoch = state
    assert epoch == config["optimize_perceiver"]["epochs"]


def test_optimize_universe_config(config):
    new_config = optimize_universe_config(config)
    assert not pytrees_equal(
        config["data"]["universe_config"]["physics_config"],
        new_config["data"]["universe_config"]["physics_config"])
    assert not pytrees_equal(
        config["data"]["universe_config"]["elem_distrib"],
        new_config["data"]["universe_config"]["elem_distrib"])


def pytrees_equal(tree1, tree2):
    tree1, unravel = jax.flatten_util.ravel_pytree(tree1)
    tree2, unravel = jax.flatten_util.ravel_pytree(tree2)
    return jnp.allclose(tree1, tree2, atol=1e-10, rtol=1e-10)


def test_distributed(config):
    jax.distributed.initialize(config["infra"]["coordinator_address"],
                               int(config["infra"]["num_hosts"]),
                               int(config["infra"]["process_id"]))


def test_minimal_nested():
    def inner_loop(x):
        return jax.lax.scan(lambda y, _: [y + 1] * 2, x, None, 2)[0]

    def outer_loop(x):
        return jax.lax.scan(lambda y, _: [inner_loop(y)] * 2, x, None, 3)[0]

    assert outer_loop(0) == 6


# TODO: Implement to_json_best_effort()
def test_persist_config(config):
    pass
