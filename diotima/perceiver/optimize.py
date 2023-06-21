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
    atom_elems: Array
    locs_history: Array
    idx_history: Array
    locs_future: Array
    universe_config: UniverseConfig
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
    cfs_to_data = lambda cf: Data(
        cf.atom_elems,
        cf.locs_history[:config.data.start],
        jnp.arange(config.data.start),
        cf.locs_history[config.data.start:],
        cf.universe_config,
        None
    )
    return jax.vmap(cfs_to_data)(cfs)


def synth_data(config: FrozenConfigDict):
    pure_synth_universe_data = lambda _: synth_universe_data(config)
    data = jax.vmap(pure_synth_universe_data)(jnp.arange(config.data.n_univs))

    data = Data(
        atom_elems=rearrange(data.atom_elems, "univs cfs a e -> (univs cfs) a e"),
        locs_history=rearrange(data.locs_history, "univs cfs t a l -> (univs cfs) (t a) l"),
        idx_history=rearrange(data.idx_history, "univs cfs t -> (univs cfs) t"),
        locs_future=rearrange(data.locs_future, "univs cfs t a l -> (univs cfs) t a l"),
        universe_config=data.universe_config,
        pred_locs_future=None
    )
    return data


def raw_forward(data: Data, config: FrozenConfigDict, is_training: bool):
    n_dims = data.locs_history.shape[-1]
    n_atoms = data.atom_elems.shape[-2]
    future_steps = config.data.steps - config.data.start
    atom_locs = rearrange(data.locs_history, "univs (t a) l -> t univs a l", a=n_atoms)[-1]

    input = repeat(data.atom_elems, "univs a e -> univs (t a) e", t=config.data.start)
    space_pos = data.locs_history
    time_pos = repeat(data.idx_history, "univs t -> univs (t a) 1", a=n_atoms)
    pos = jnp.concatenate((space_pos, time_pos), axis=2)

    preprocessor = DynamicPointCloudPreprocessor(**config.preprocessor)
    encoder = PerceiverEncoder(**config.encoder)
    decoder = BasicDecoder(**config.decoder)

    input, _, input_without_pos = preprocessor(input, pos)
    encoder_query = encoder.latents(input)

    decoder_query_input = repeat(data.atom_elems, "univs a e -> univs (br a) e", br=config.optimization.branches)
    decoder_query_space_pos = repeat(atom_locs, "univs a l -> univs (br a) l", br=config.optimization.branches)
    decoder_query_branch_pos = repeat(jnp.arange(config.optimization.branches), "br -> univs (br a) 1", univs=config.data.n_univs * config.data.n_cfs, a=n_atoms)
    decoder_query_pos = jnp.concatenate((decoder_query_space_pos, decoder_query_branch_pos), axis=2)

    decoder_query, _, decoder_query_without_pos = preprocessor(decoder_query_input, decoder_query_pos)

    latents = encoder(input, encoder_query, is_training=is_training)

    def decode_slot(latent):
        latent = rearrange(latent, "b z -> b 1 z")
        return decoder(decoder_query, latent, is_training=is_training)

    one_step_preds = jax.vmap(decode_slot, in_axes=1)(latents)
    # What aggregate into?
    agg_one_step_preds = reduce(one_step_preds, "z b a l -> b a l", "sum")

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = jax.lax.scan(inject_one_step, decoder_query_space_pos, None, future_steps)
        forecast = rearrange(forecast[1], "t b (br a) l -> b t a br l", br=config.optimization.branches)
        return forecast

    forecast = preds_to_forecast()

    return Data(
        data.atom_elems,
        data.locs_history,
        data.idx_history,
        data.locs_future,
        data.universe_config,
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
    return jnp.square(data.pred_locs_future - repeat(data.locs_future, "b t a l -> b t a br l", br=config.optimization.branches)).mean()


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
            "n_univs": 2,
            "steps": 4,
            "n_cfs": 2,
            "start": 2
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
            "z_index_dim": 5,
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
