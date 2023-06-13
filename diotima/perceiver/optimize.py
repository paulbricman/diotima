from diotima.world.universe import default_universe_config, UniverseConfig, seed, run, trim, spawn_counterfactuals
from diotima.perceiver.io_processors import DynamicPointCloudPreprocessor
from diotima.perceiver.perceiver import PerceiverEncoder, BasicDecoder

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
from jax import Array
import haiku as hk
import optax

from einops import repeat, rearrange
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
    key: PRNGKeyArray


def synth_universe_data(config: UniverseDataConfig):
    # TODO: Extend from dummy.
    keys = jax.random.split(config.key, num=2)

    universe_config = default_universe_config()
    universe = seed(universe_config, keys[0])
    universe = run(universe, config.steps)
    cfs = spawn_counterfactuals(
        universe,
        config.start,
        config.n_cfs,
        keys[1]
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


def synth_data(config: UniverseDataConfig, n_univs: int, key: PRNGKeyArray):
    keys = jax.random.split(key, num=n_univs)

    pure_synth_universe_data = lambda key: synth_universe_data(config)
    data = jax.vmap(pure_synth_universe_data)(keys)

    data = Data(
        atom_elems=rearrange(data.atom_elems, "univs cfs a e -> (univs cfs) a e"),
        locs_history=rearrange(data.locs_history, "univs cfs t a l -> (univs cfs) t a l"),
        idx_history=rearrange(data.idx_history, "univs cfs t -> (univs cfs) t"),
        locs_future=rearrange(data.locs_future, "univs cfs t a l -> (univs cfs) t a l"),
        universe_config=data.universe_config,
        pred_locs_future=None
    )
    return data


def raw_forward(data: Data, config: UniverseDataConfig, is_training: bool):
    input = repeat(data.atom_elems, "univs a e -> univs t a e", t=config.start)
    space_pos = data.locs_history
    time_pos = repeat(data.idx_history, "univs t -> univs t a 1", a=int(data.universe_config.n_atoms[0][0]))
    pos = jnp.concatenate((space_pos, time_pos), axis=3)

    preprocessor = DynamicPointCloudPreprocessor(
        fourier_position_encoding_kwargs=dict(
          num_bands=2,
          max_resolution=[1] * int(data.universe_config.n_dims[0][0]),
          sine_only=False,
          concat_pos=True,
      )
    )
    encoder = PerceiverEncoder()
    decoder = BasicDecoder(
        output_num_channels=int(data.universe_config.n_dims[0][0]),
        position_encoding_type="fourier",
        fourier_position_encoding_kwargs=dict(
          num_bands=2,
          max_resolution=[1] * int(data.universe_config.n_dims[0][0]),
          sine_only=False,
          concat_pos=True,
      )
    )

    input, _, input_without_pos = preprocessor(input, pos)
    encoder_query = encoder.latents(input)
    decoder_query = decoder.decoder_query(input, None, input_without_pos)
    # TODO: Only use last timestep to build decoder queries

    latents = encoder(input, encoder_query, is_training=is_training, input_mask=input_mask)
    outputs = decoder(decoder_query, latents, is_training=is_training, query_mask=query_mask)
    # TODO: Decode one latent at a time for slot attention
    # TODO: Scan the application of outputs for forecast.

    return Data(
        data.atom_elems,
        data.locs_history,
        data.idx_history,
        data.locs_future,
        data.universe_config,
        forecast
    )


forward = hk.transform_with_state(raw_forward)


def loss(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        rng: Array,
        data: Data):
    data = forward.apply(params, state, rng, data, is_training=True)
    return jnp.square(data.locs_future - data.pred_locs_future).mean()


def backward(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        data: Data
):
    # Given true and predicted outputs, update weights.
    grads = jax.grad(loss)(params, state, data, rng)
    # TODO: Streamline params

    updates, opt_state = optimizer.update(grads, opt_state, params)
    # TODO: Initialize optimizer
    params = optax.apply_updates(params, updates)

    return params, state, opt_state


def optimize():
    pass
    # Repeatedly run forward and backward.

    # Return final weights.


def orchestrate():
    pass
    # Synthesize data.

    # Load weights.

    # Optimize them.

    # Return them, along with optimization history.

