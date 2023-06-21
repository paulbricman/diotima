from diotima.world.universe import default_universe_config, UniverseConfig, seed, run, trim, spawn_counterfactuals
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
from ml_collections.config_dict import FrozenConfigDict


OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]


class Data(NamedTuple):
    universe_config: UniverseConfig
    atom_elems: Array
    idx_history: Array
    locs_history: Array
    locs_future: Array
    pred_locs_future: Array


def synth_universe_data(config: FrozenConfigDict):
    # TODO: Extend from dummy.
    universe_config = default_universe_config()
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


def synth_data(config: FrozenConfigDict):
    pure_synth_universe_data = lambda _: synth_universe_data(config)
    return jax.vmap(pure_synth_universe_data)(jnp.arange(config.data.n_univs))


def raw_forward(data: Data, config: FrozenConfigDict, is_training: bool):
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

    decoder_query_input = repeat(data.atom_elems, "u a e -> u (b a) e", b=config.optimization.branches)
    decoder_query_space_pos = repeat(atom_locs, "u a l -> u (b a) l", b=config.optimization.branches)
    decoder_query_branch_pos = repeat(jnp.arange(config.optimization.branches), "b -> u (b a) 1", u=config.data.n_univs, a=n_atoms)

    decoder_query_pos = jnp.concatenate((decoder_query_space_pos, decoder_query_branch_pos), axis=2)

    decoder_query, _, decoder_query_without_pos = preprocessor(decoder_query_input, decoder_query_pos)

    latents = encoder(input, encoder_query, is_training=is_training)

    def decode_slot(latent):
        latent = rearrange(latent, "u z -> u 1 z")
        return decoder(decoder_query, latent, is_training=is_training)

    one_step_preds = jax.vmap(decode_slot, in_axes=1)(latents)
    # What aggregate into?
    agg_one_step_preds = reduce(one_step_preds, "z u a l -> u a l", "sum")

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = jax.lax.scan(inject_one_step, decoder_query_space_pos, None, future_steps)
        forecast = rearrange(forecast[1], "t u (b a) l -> u t a b l", b=config.optimization.branches)
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


def init_opt(config: FrozenConfigDict):
    data = synth_data(config)

    forward = hk.transform_with_state(raw_forward)
    params, state = forward.init(next(config.rng), data, config, True)
    return params, state, forward


def loss(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        data: Data,
        config: FrozenConfigDict
):
    data, state = forward.apply(params, state, next(config.rng), data, config, is_training=True)
    # TODO: Implement loss for affinity of cfs and bs
    # return jnp.square(data.pred_locs_future - data.locs_future).mean()
    return jnp.array(42.)


def backward(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        optimizer,
        data: Data,
        config: FrozenConfigDict
):
    grads = jax.grad(loss)(params, state, opt_state, forward, data, config)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, state, opt_state


def optimize(
        config: FrozenConfigDict
):
    data = synth_data(config)
    params, state, forward = init_opt(config)
    optim = optax.adam(config.optimizer.lr)
    opt_state = optim.init(params)

    def scanned_backward(state, _):
        params, state, opt_state = state
        new_params, new_state, new_opt_state = backward(params, state, opt_state, forward, optim, data, config)
        state = new_params, new_state, new_opt_state
        return state, state

    return jax.lax.scan(scanned_backward, (params, state, opt_state), None, config.optimization.epochs)


def default_config(universe_config):
    config = {
        "optimization": {
            "epochs": 2,
            "branches": 2
        },
        "optimizer": {
            "lr": 1e-4
        },
        "data": {
            "n_univs": 3,
            "steps": 6,
            "n_cfs": 5,
            "start": 4
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
            "output_num_channels": universe_config.n_dims,
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

    cfg = FrozenConfigDict(config)
    return cfg
