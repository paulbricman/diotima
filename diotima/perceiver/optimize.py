from diotima.world.universe import UniverseConfig, seed, run, spawn_counterfactuals
import diotima.world.physics as physics
from diotima.perceiver.io_processors import DynamicPointCloudPreprocessor
from diotima.perceiver.perceiver import PerceiverEncoder, BasicDecoder

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
from jax import Array
import haiku as hk
import optax
from jaxline import experiment

from einops import repeat, rearrange, reduce
from typing import Tuple, NamedTuple
from ml_collections.config_dict import ConfigDict
from safejax.haiku import serialize
from pathlib import Path
import json


OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]


class Data(NamedTuple):
    universe_config: UniverseConfig
    atom_elems: Array
    idx_history: Array
    locs_history: Array
    locs_future: Array
    pred_locs_future: Array


def synth_universe_data(config: ConfigDict):
    universe_config = UniverseConfig(**config.data.universe_config)
    universe = seed(universe_config, next(config.rng))
    universe = run(universe, config.data.steps)

    cfs = spawn_counterfactuals(
        universe,
        config.data.start,
        config.data.n_cfs,
        next(config.rng)
    )

    cf_to_datum = lambda cf: Data(
        universe_config=None,
        atom_elems=None,
        idx_history=None,
        locs_history=None,
        locs_future=cf.locs_history[config.data.start:],
        pred_locs_future=None,
    )

    data = jax.vmap(cf_to_datum)(cfs)
    data = data._replace(universe_config=universe.universe_config)
    data = data._replace(atom_elems=universe.atom_elems)
    data = data._replace(idx_history=jnp.arange(config.data.start))
    data = data._replace(locs_history=universe.locs_history[:config.data.start])

    return data


def synth_data(config: ConfigDict):
    pure_synth_universe_data = lambda _: synth_universe_data(config)
    return jax.vmap(pure_synth_universe_data)(jnp.arange(config.data.n_univs))


def raw_forward(data: Data, config: ConfigDict, is_training: bool):
    future_steps = config.data.steps - config.data.start
    n_atoms = data.locs_history.shape[2]
    atom_locs = rearrange(data.locs_history, "u t a l -> t u a l")[-1]

    input = repeat(data.atom_elems, "u a e -> u (t a) e", t=config.data.start)
    space_pos = repeat(data.locs_history, "u t a l -> u (t a) l")
    time_pos = repeat(data.idx_history, "u t -> u (t a) 1", a=n_atoms)
    pos = jnp.concatenate((space_pos, time_pos), axis=2)

    preprocessor = DynamicPointCloudPreprocessor(**config.preprocessor)
    encoder = PerceiverEncoder(**config.encoder)
    decoder = BasicDecoder(**config.decoder)

    input, _, input_without_pos = preprocessor(input, pos)
    encoder_query = encoder.latents(input)

    decoder_query_input = repeat(data.atom_elems, "u a e -> u (b a) e", b=config.optimize_perceiver.branches)
    decoder_query_space_pos = repeat(atom_locs, "u a l -> u (b a) l", b=config.optimize_perceiver.branches)
    decoder_query_branch_pos = repeat(jnp.arange(config.optimize_perceiver.branches), "b -> u (b a) 1", u=config.data.n_univs, a=n_atoms)

    decoder_query_pos = jnp.concatenate((decoder_query_space_pos, decoder_query_branch_pos), axis=2)

    decoder_query, _, decoder_query_without_pos = preprocessor(decoder_query_input, decoder_query_pos)

    latents = encoder(input, encoder_query, is_training=is_training)

    def decode_slot(latent):
        latent = rearrange(latent, "u z -> u 1 z")
        return decoder(decoder_query, latent, is_training=is_training)

    one_step_preds = jax.vmap(decode_slot, in_axes=1)(latents)
    agg_one_step_preds = reduce(one_step_preds, "z u a l -> u a l", "sum")

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = jax.lax.scan(inject_one_step, decoder_query_space_pos, None, future_steps)
        forecast = rearrange(forecast[1], "t u (b a) l -> u t a b l", b=config.optimize_perceiver.branches)
        return forecast

    forecast = preds_to_forecast()

    return Data(
        data.universe_config,
        data.atom_elems,
        data.idx_history,
        data.locs_history,
        data.locs_future,
        forecast
    )


def init_opt(config: ConfigDict):
    data = synth_data(config)

    forward = hk.transform_with_state(raw_forward)
    params, state = forward.init(next(config.rng), data, config, True)

    optim = optax.adam(config.optimize_perceiver.lr)
    opt_state = optim.init(params)

    return params, state, opt_state, optim, forward


def loss(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        data: Data,
        config: ConfigDict
):
    data, state = forward.apply(params, state, next(config.rng), data, config, is_training=True)
    error = distance(data.pred_locs_future, data.locs_future)
    return error


def distance(
        cfs: Array,
        bs: Array,
):
    # Differentiable cousin of Hausdorff distance
    def compute_batch_distances(cfs, bs):
        def compute_distances(cf, bs):
            distances = jnp.linalg.norm(cf - bs, axis=1)
            weights = 1 - jax.nn.softmax(distances)
            softmin = jnp.dot(distances, weights)
            return softmin

        # Iterate over cfs to find shortest way to bs
        dist_from_cfs = jax.vmap(lambda cf: compute_distances(cf, bs))(cfs)
        print(dist_from_cfs)
        dist_from_cfs = jnp.mean(dist_from_cfs)

        # Iterate over bs to estimate shortest way to cfs
        dist_from_bs = jax.vmap(lambda b: compute_distances(b, cfs))(bs)
        print(dist_from_bs)
        dist_from_bs = jnp.mean(dist_from_bs)

        dist = dist_from_cfs + dist_from_bs
        return dist

    # Iterate over batches
    return jnp.mean(jax.vmap(lambda cfs, bs: compute_batch_distances(cfs, bs))(cfs, bs))


def l2_penalty(params):
    # TODO: Specialize L2 regularization to specific branches of param pytree.
    theta, unravel = jax.flatten_util.ravel_pytree(params)
    return jnp.sum(theta ** 2)


def backward(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        optimizer,
        data: Data,
        config: ConfigDict,
        epoch: Array
):
    grads = jax.grad(loss)(params, state, opt_state, forward, data, config)
    agg_grads = lambda grad: jax.lax.pmean(grads, axis_name="devices")
    no_agg_grads = lambda grad: grad
    grads = jax.lax.cond(
        epoch % config.optimize_perceiver.agg_every == 0,
        agg_grads,
        no_agg_grads,
        grads
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, state, opt_state


def optimize_perceiver(
        config: ConfigDict,
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        optim,
        forward
):
    def scanned_backward(state, _):
        data = synth_data(config)
        params, state, opt_state, epoch = state
        new_params, new_state, new_opt_state = backward(params, state, opt_state, forward, optim, data, config, epoch)

        state = new_params, new_state, new_opt_state, epoch + 1
        return state, state

    scan_backward = lambda _: jax.lax.scan(scanned_backward, (params, state, opt_state, 0), None, config.optimize_perceiver.epochs)

    return jax.pmap(scan_backward, axis_name="devices")(jnp.arange(jax.local_device_count()))


def optimize_universe_config(config: ConfigDict):
    optim = optax.adam(config.optimize_universe_config.lr)
    opt_state = optim.init(params)

    def universe_config_state_to_config(universe_config_state):
        physics_config, elem_distrib = universe_config_state
        config.data.universe_config.physics_config = physics_config
        config.data.universe_config.elem_distrib = elem_distrib
        return config

    def param_count(universe_config_state):
        config = universe_config_state_to_config(universe_config_state)

        params, state, opt_state, optim, forward = init_opt(config)
        state, history = optimize_perceiver(config, params, state, opt_state, optim, forward)
        params, state, opt_state, epoch = state
        return l2_penalty(params)

    def scanned_optimize_universe_config(state, _):
        universe_config_state, opt_state = state
        param_count, grads = jax.value_and_grad(config_to_param_count)(universe_config_state)
        updates, opt_state = optimizer.update(grads, opt_state, universe_config_state)
        universe_config_state = optax.apply_updates(state, universe_config_state)
        state = universe_config_state, opt_state

        return state, state

    universe_config_state = (config.data.universe_config.physics_config,
                             config.data.universe_config.elem_distrib)

    state, history = jax.lax.scan(scanned_optimize_universe_config,
                                  (universe_config_state, opt_state),
                                  0,
                                  config.optimize_universe_config.epochs)

    universe_config_state, opt_state = state
    config = universe_config_state_to_config(universe_config_state)
    return config


def checkpoint(
        params: hk.Params,
        state: hk.State,
        filepath: str
):
    root = Path(filepath)

    serialize(params, filename=root / "params.safetensors")
    serialize(state, filename=root / "state.safetensors")


def default_config(universe_config):
    config = {
        "optimize_perceiver": {
            "epochs": 3,
            "branches": 2,
            "agg_every": 2,
            "lr": 1e-4,
        },
        "optimize_universe_config": {
            "epochs": 3,
            "lr": 1e-4,
        },
        "data": {
            "n_univs": 3,
            "steps": 6,
            "n_cfs": 5,
            "start": 4,
            "universe_config": {
                "n_elems": 2,
                "n_atoms": 2,
                "n_dims": 2,
                "dt": 0.1,
                "physics_config": physics.default_physics_config(2),
                "elem_distrib": physics.default_elem_distrib(2)
            }
        },
        "rng": hk.PRNGSequence(jax.random.PRNGKey(0)),
        "preprocessor": {
            "fourier_position_encoding_kwargs": {
                "num_bands": 1,
                "max_resolution": [1],
                "sine_only": True,
                "concat_pos": False
            }
        },
        "encoder": {
            "z_index_dim": 7,
            "num_z_channels": 8,
            "num_blocks": 1,
            "num_self_attends_per_block": 1,
            "num_self_attend_heads": 1
        },
        "decoder": {
            "output_num_channels": 2, # Has to match n_dims
            "num_z_channels": 8,
            "position_encoding_type": "fourier",
            "fourier_position_encoding_kwargs": {
                "num_bands": 1,
                "max_resolution": [1],
                "sine_only": True,
                "concat_pos": False
            }
        }
    }

    return ConfigDict(config)
