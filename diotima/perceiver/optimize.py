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


OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]


class Data(NamedTuple):
    atom_elems: Array
    locs_history: Array
    idx_history: Array
    locs_future: Array
    universe_config: UniverseConfig
    pred_locs_future: Array


class UniverseDataConfig(NamedTuple):
    steps: int
    n_cfs: int
    start: int
    rng: hk.PRNGSequence


def synth_universe_data(config: UniverseDataConfig):
    # TODO: Extend from dummy.
    universe_config = default_universe_config()
    universe = seed(universe_config, next(config.rng))
    universe = run(universe, config.steps)

    cfs = spawn_counterfactuals(
        universe,
        config.start,
        config.n_cfs,
        next(config.rng)
    )
    cfs_to_data = lambda cf: Data(
        cf.atom_elems,
        cf.locs_history[:config.start],
        jnp.arange(config.start),
        cf.locs_history[config.start:],
        cf.universe_config,
        None
    )
    return jax.vmap(cfs_to_data)(cfs)


def synth_data(config: UniverseDataConfig, n_univs: int):
    pure_synth_universe_data = lambda _: synth_universe_data(config)
    data = jax.vmap(pure_synth_universe_data)(jnp.arange(n_univs))

    data = Data(
        atom_elems=rearrange(data.atom_elems, "univs cfs a e -> (univs cfs) a e"),
        locs_history=rearrange(data.locs_history, "univs cfs t a l -> (univs cfs) (t a) l"),
        idx_history=rearrange(data.idx_history, "univs cfs t -> (univs cfs) t"),
        locs_future=rearrange(data.locs_future, "univs cfs t a l -> (univs cfs) (t a) l"),
        universe_config=data.universe_config,
        pred_locs_future=None
    )
    return data


def raw_forward(data: Data, config: UniverseDataConfig, is_training: bool):
    n_dims = data.locs_history.shape[-1]
    n_atoms = data.atom_elems.shape[-2]
    future_steps = config.steps - config.start
    atom_locs = rearrange(data.locs_history, "univs (t a) l -> t univs a l", a=n_atoms)[-1]

    input = repeat(data.atom_elems, "univs a e -> univs (t a) e", t=config.start)
    space_pos = data.locs_history
    time_pos = repeat(data.idx_history, "univs t -> univs (t a) 1", a=n_atoms)
    pos = jnp.concatenate((space_pos, time_pos), axis=2)

    # TODO: Move config to separate get_config
    preprocessor = DynamicPointCloudPreprocessor(
        fourier_position_encoding_kwargs=dict(
            num_bands=1,
            max_resolution=[1],
            sine_only=True,
            concat_pos=False,
        )
    )
    encoder = PerceiverEncoder(
        z_index_dim=5,
        num_z_channels=8,
        num_blocks=1,
        num_self_attends_per_block=1,
        num_self_attend_heads=1,
    )

    # TODO: Make things smol.
    decoder = BasicDecoder(
        output_num_channels=n_dims,
        num_z_channels=8,
        position_encoding_type="fourier",
        fourier_position_encoding_kwargs=dict(
            num_bands=1,
            max_resolution=[1],
            sine_only=True,
            concat_pos=False,
        )
    )

    input, _, input_without_pos = preprocessor(input, pos)
    encoder_query = encoder.latents(input)

    decoder_query, _, decoder_query_without_pos = preprocessor(
        data.atom_elems,
        atom_locs
    )

    latents = encoder(input, encoder_query, is_training=is_training)

    def decode_slot(latent):
        latent = rearrange(latent, "b z_dim -> b 1 z_dim")
        return decoder(decoder_query, latent, is_training=is_training)

    # TODO: Implement some kind of sampling everything else equal.
    one_step_preds = jax.vmap(decode_slot, in_axes=1)(latents)
    agg_one_step_preds = reduce(one_step_preds, "z b a l -> b a l", "sum")

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = jax.lax.scan(inject_one_step, atom_locs, None, future_steps)
        forecast = rearrange(forecast[1], "t b a l -> b (t a) l")
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


def init_opt(config: UniverseDataConfig):
    data = synth_data(config, n_univs=2)

    forward = hk.transform_with_state(raw_forward)
    params, state = forward.init(next(config.rng), data, config, True)
    return params, state, forward


def loss(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        rng: Array,
        data: Data,
        config: UniverseDataConfig
):
    data, state = forward.apply(params, state, rng, data, config, is_training=True)
    return jnp.square(data.locs_future - data.pred_locs_future).mean()


def backward(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        rng: Array,
        optimizer,
        data: Data,
        config: UniverseDataConfig
):
    grads = jax.grad(loss)(params, state, opt_state, forward, rng, data, config)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, state, opt_state


def optimize(
        config: UniverseDataConfig
):
    # TODO: Move to broader config.
    epochs = 2
    n_univs = 2

    data = synth_data(config, n_univs=n_univs)
    params, state, forward = init_opt(config)
    optim = optax.adam(1e-4)
    opt_state = optim.init(params)

    def scanned_backward(state, _):
        params, state, opt_state = state
        new_params, new_state, new_opt_state = backward(params, state, opt_state, forward, next(config.rng), optim, data, config)
        state = new_params, new_state, new_opt_state
        return state, state

    return jax.lax.scan(scanned_backward, (params, state, opt_state), None, epochs)
