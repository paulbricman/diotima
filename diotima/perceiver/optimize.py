from functools import partial
from diotima.world.universe import UniverseConfig, seed, run, spawn_counterfactuals
import diotima.world.physics as physics
from diotima.perceiver.io_processors import DynamicPointCloudPreprocessor
from diotima.perceiver.perceiver import PerceiverEncoder, BasicDecoder

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
from jax.experimental.maps import xmap
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax import Array
import haiku as hk
import optax

from einops import repeat, rearrange, reduce
from typing import Tuple, NamedTuple, Dict
from safetensors.flax import save_file
from pathlib import Path
import os
import wandb


jax.config.update("jax_spmd_mode", "allow_all")
OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
hk.vmap.require_split_rng = False
devices = mesh_utils.create_device_mesh((jax.process_count(), jax.local_device_count()))
mesh = Mesh(devices, axis_names=("hosts", "devices"))


class Data(NamedTuple):
    atom_elems: Array
    idx_history: Array
    locs_history: Array
    locs_future: Array
    pred_locs_future: Array


class Agents(NamedTuple):
    flows: Array
    substrates: Array


def synth_universe_data(config: Dict, key: PRNGKeyArray):
    universe_config = UniverseConfig(**config["data"]["universe_config"])
    universe = seed(universe_config, key)
    universe = run(universe, universe_config, config["data"]["steps"])

    cfs = spawn_counterfactuals(
        universe,
        universe_config,
        config["data"]["start"],
        config["data"]["n_cfs"],
        key,
    )

    def cf_to_datum(cf):
        return Data(
            atom_elems=None,
            idx_history=None,
            locs_history=None,
            locs_future=cf.locs_history[config["data"]["start"] :],
            pred_locs_future=None,
        )

    data = jax.vmap(cf_to_datum)(cfs)
    data = data._replace(atom_elems=universe.atom_elems)
    data = data._replace(idx_history=jnp.arange(config["data"]["start"]))
    data = data._replace(locs_history=universe.locs_history[: config["data"]["start"]])

    return data


def synth_data(config: Dict, key: PRNGKeyArray):
    synth_data_rngs = jax.random.split(key, config["data"]["n_univs"])
    # Local config copy to avoid side effects:
    config = config.copy()

    def pure_synth_universe_data(univ):
        subkey = synth_data_rngs[univ]
        return synth_universe_data(config, subkey)

    return jax.vmap(pure_synth_universe_data)(jnp.arange(config["data"]["n_univs"]))


def raw_forward(data: Data, config: Dict, is_training: bool):
    future_steps = config["data"]["steps"] - config["data"]["start"]
    n_atoms = config["data"]["universe_config"]["n_atoms"]
    atom_locs = rearrange(data.locs_history, "u t a l -> t u a l")[-1]

    input = repeat(data.atom_elems, "u a e -> u (t a) e", t=config["data"]["start"])
    space_pos = repeat(data.locs_history, "u t a l -> u (t a) l")
    time_pos = repeat(data.idx_history, "u t -> u (t a) 1", a=n_atoms)
    pos = jnp.concatenate((space_pos, time_pos), axis=2)

    preprocessor = DynamicPointCloudPreprocessor(**config["preprocessor"])
    encoder = PerceiverEncoder(**config["encoder"])
    decoder = BasicDecoder(**config["decoder"])

    input, _, input_without_pos = preprocessor(input, pos)
    encoder_query = encoder.latents(input)

    decoder_query_input = repeat(
        data.atom_elems,
        "u a e -> u (b a) e",
        b=config["optimize_perceiver"]["branches"],
    )
    decoder_query_space_pos = repeat(
        atom_locs, "u a l -> u (b a) l", b=config["optimize_perceiver"]["branches"]
    )
    decoder_query_branch_pos = repeat(
        jnp.arange(config["optimize_perceiver"]["branches"]),
        "b -> u (b a) 1",
        u=config["data"]["n_univs"],
        a=n_atoms,
    )

    decoder_query_pos = jnp.concatenate(
        (decoder_query_space_pos, decoder_query_branch_pos), axis=2
    )

    decoder_query, _, decoder_query_without_pos = preprocessor(
        decoder_query_input, decoder_query_pos
    )

    latents, scores = encoder(input, encoder_query, is_training=is_training)

    def decode_slot(latent):
        latent = rearrange(latent, "u z -> u 1 z")
        return decoder(decoder_query, latent, is_training=is_training).astype(
            "bfloat16"
        )

    one_step_preds = hk.vmap(decode_slot, in_axes=1)(latents)
    agg_one_step_preds = reduce(one_step_preds, "z u a l -> u a l", "sum")
    flows = rearrange(
        one_step_preds,
        "z u (b a) l -> u b z a l",
        b=config["optimize_perceiver"]["branches"],
    )

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = hk.scan(inject_one_step, decoder_query_space_pos, None, future_steps)
        forecast = rearrange(
            forecast[1],
            "t u (b a) l -> u b t a l",
            b=config["optimize_perceiver"]["branches"],
        )
        return forecast

    forecast = preds_to_forecast()

    enriched_data = Data(
        data.atom_elems,
        data.idx_history,
        data.locs_history,
        data.locs_future,
        forecast,
    )

    agents = Agents(flows, compute_attn_rollouts(scores))

    return enriched_data, agents


def compute_attn_rollouts(scores, repr_self_residual=True):
    cross, self = scores
    cross = reduce(cross, "u cah z a -> u a z", "sum")
    self = reduce(self, "sa u sah z1 z2 -> u sa z1 z2", "sum")

    def compute_attn_rollout(self, cross):
        if repr_self_residual:
            # Relevant: https://arxiv.org/pdf/2005.00928.pdf#page=3
            z = self.shape[-1]
            self += jnp.eye(z)
            norm_factor = repeat(
                jnp.sum(self, axis=(1, 2)), "sa -> sa z1 z2", z1=z, z2=z
            )
            self /= norm_factor

        # Relevant: https://github.com/deepmind/deepmind-research/issues/253
        first = [cross @ self[0]]
        rest = list(self[1:])
        return jnp.linalg.multi_dot(first + rest)

    return jax.vmap(compute_attn_rollout)(self, cross)


def init_opt(config: Dict, key: PRNGKeyArray):
    data_key, init_key = jax.random.split(key, num=2)
    data = synth_data(config, data_key)

    forward = hk.transform(raw_forward)
    params = forward.init(init_key, data, config, True)

    optim = optax.adamw(
        config["optimize_perceiver"]["lr"],
        eps_root=1e-10,
        weight_decay=config["optimize_perceiver"]["weight_decay"],
    )
    opt_state = optim.init(params)

    return params, opt_state, optim, forward


def loss(params: hk.Params, forward, data: Data, config: Dict, key: PRNGKeyArray):
    out = forward.apply(params, key, data, config, is_training=True)
    data, agents = out
    error = distance(data.pred_locs_future, data.locs_future)

    return error


def distance(
    cfs: Array,
    bs: Array,
):
    # Differentiable cousin of Hausdorff distance
    def compute_batch_distances(cfs, bs):
        def compute_distances(x, ys):
            distances = jnp.linalg.norm(x - ys, axis=-1)
            weights = 1 - jax.nn.softmax(distances) + 1e-8
            softmin = jnp.average(distances, weights=weights, axis=(1, 2))
            return softmin

        # Iterate over cfs to find shortest way to bs
        dist_from_cfs = jax.vmap(lambda cf: compute_distances(cf, bs))(cfs)
        dist_from_cfs = jnp.mean(dist_from_cfs)

        # Iterate over bs to estimate shortest way to cfs
        dist_from_bs = jax.vmap(lambda b: compute_distances(b, cfs))(bs)
        dist_from_bs = jnp.mean(dist_from_bs)

        dist = dist_from_cfs + dist_from_bs
        return dist

    # Iterate over batches
    return jnp.mean(jax.vmap(lambda cfs, bs: compute_batch_distances(cfs, bs))(cfs, bs))


def compute_l2_penalty(params):
    theta, unravel = jax.flatten_util.ravel_pytree(params)
    return jnp.sum(theta**2)


def backward(
    params: hk.Params,
    opt_state: OptState,
    forward,
    optimizer,
    data: Data,
    config: Dict,
    epoch: Array,
    key: PRNGKeyArray,
):
    perceiver_loss, grads = jax.value_and_grad(loss)(params, forward, data, config, key)
    grads = jax.lax.pmean(grads, axis_name="devices")
    grads = jax.lax.pmean(grads, axis_name="hosts")

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, perceiver_loss


def optimize_perceiver(
    config: Dict,
    perceiver_params: hk.Params,
    perceiver_opt_state: OptState,
    perceiver_optim,
    forward,
    key: PRNGKeyArray,
):
    optimize_perceiver_rngs = jax.random.split(
        key, config["optimize_perceiver"]["epochs"]
    )

    def scanned_backward(opt_perceiver_state, _):
        params, opt_state, opt_perceiver_epoch, _ = opt_perceiver_state
        data_key, apply_key = jax.random.split(
            optimize_perceiver_rngs[opt_perceiver_epoch]
        )

        data = synth_data(config, data_key)
        params, opt_state, perceiver_loss = backward(
            params,
            opt_state,
            forward,
            perceiver_optim,
            data,
            config,
            opt_perceiver_epoch,
            apply_key,
        )

        opt_perceiver_state = params, opt_state, opt_perceiver_epoch + 1, perceiver_loss

        return opt_perceiver_state, opt_perceiver_state

    init_opt_perceiver_state = (perceiver_params, perceiver_opt_state, 0, 0)

    def scan_backward(_):
        carry, history = jax.lax.scan(
            scanned_backward,
            init_opt_perceiver_state,
            None,
            config["optimize_perceiver"]["epochs"],
        )
        return carry, history

    return scan_backward(None)


def optimize_universe_config(config: Dict, key: PRNGKeyArray, return_params=False):
    log = config["infra"]["log"]

    def universe_config_state_to_config(universe_config_state):
        physics_config, elem_distrib = universe_config_state
        return default_config(physics_config, elem_distrib, log)

    def compute_sophistication(universe_config_state, opt_universe_epoch, key):
        config = universe_config_state_to_config(universe_config_state)
        init_key, apply_key = jax.random.split(key)

        params, perceiver_opt_state, perceiver_optim, forward = init_opt(
            config, init_key
        )

        opt_perceiver_carry, opt_perceiver_history = optimize_perceiver(
            config, params, perceiver_opt_state, perceiver_optim, forward, apply_key
        )
        params, perceiver_opt_state, epoch, _ = opt_perceiver_carry

        return -compute_l2_penalty(params), (params, opt_perceiver_history[3])

    def init():
        universe_config_state = (
            config["data"]["universe_config"]["physics_config"],
            config["data"]["universe_config"]["elem_distrib"],
        )
        universe_config_optim = optax.adam(
            config["optimize_universe_config"]["lr"], eps_root=1e-10
        )
        universe_config_opt_state = universe_config_optim.init(universe_config_state)
        return (
            universe_config_state,
            universe_config_opt_state,
            universe_config_optim,
        )

    universe_config_state, universe_config_opt_state, universe_config_optim = init()

    def scan_optimize_universe_config(seed):
        def scanned_optimize_universe_config(opt_universe_config_state, _):
            (
                universe_config_state,
                universe_config_opt_state,
                opt_universe_epoch,
                _,
                _,
            ) = opt_universe_config_state
            value, grads = jax.value_and_grad(compute_sophistication, has_aux=True)(
                universe_config_state, opt_universe_epoch, jax.random.PRNGKey(seed)
            )
            param_count, aux = value
            params, perceiver_loss_history = aux

            grads = jax.lax.pmean(grads, axis_name="devices")
            grads = jax.lax.pmean(grads, axis_name="hosts")

            updates, universe_config_opt_state = universe_config_optim.update(
                grads, universe_config_opt_state, universe_config_state
            )
            universe_config_state = optax.apply_updates(universe_config_state, updates)

            opt_universe_config_state = (
                universe_config_state,
                universe_config_opt_state,
                opt_universe_epoch + 1,
                param_count,
                perceiver_loss_history,
            )

            return opt_universe_config_state, opt_universe_config_state

        return jax.lax.scan(
            scanned_optimize_universe_config,
            (
                universe_config_state,
                universe_config_opt_state,
                0,
                0,
                jnp.zeros((config["optimize_perceiver"]["epochs"],), dtype="bfloat16"),
            ),
            None,
            config["optimize_universe_config"]["epochs"],
        )

    with mesh:
        (
            opt_universe_config_carry,
            opt_universe_config_history,
        ) = xmap(
            scan_optimize_universe_config,
            in_axes=["hosts", "devices"],
            out_axes=[...],
            axis_resources={"hosts": "hosts", "devices": "devices"},
        )(
            jnp.arange(jax.device_count()).reshape(
                (jax.process_count(), jax.local_device_count())
            )
        )

    # TODO: Refactor the end of this function into a separate one for pushing history to wandb.
    # This one should only return a history.
    universe_config_state, opt_state, _, _, _ = opt_universe_config_carry
    config = universe_config_state_to_config(universe_config_state)

    _, _, _, param_count, perceiver_loss_history = opt_universe_config_history

    # Good old for loops for non-JIT, non-JVP logging.
    if config["infra"]["log"]:
        for idx, epoch in enumerate(param_count):
            wandb.log({"optimize_universe_config_param_count": epoch})

        perceiver_loss_history = reduce(
            perceiver_loss_history, "ue pe -> (ue pe)", "mean"
        )
        for idx, epoch in enumerate(perceiver_loss_history):
            wandb.log({"optimize_perceiver_loss": epoch})

    if return_params:
        with mesh:

            def last_perceiver(_):
                params, opt_state, optim, forward = init_opt(
                    config, jax.random.PRNGKey(0)
                )
                opt_perceiver_carry, opt_perceiver_history = optimize_perceiver(
                    config, params, opt_state, optim, forward, jax.random.PRNGKey(0)
                )
                params, perceiver_opt_state, epoch, _ = opt_perceiver_carry
                return params

            last_params = xmap(
                last_perceiver,
                in_axes=["hosts", "devices"],
                out_axes=[...],
                axis_resources={"hosts": "hosts", "devices": "devices"},
            )(
                jnp.arange(jax.device_count()).reshape(
                    (jax.process_count(), jax.local_device_count())
                )
            )

        return config, last_params
    return config


def checkpoint(params: hk.Params, filepath: str = "."):
    root = Path(filepath)

    save_file(params, filename=root / "params.safetensors")


def sanitize_config(config: Dict):
    config["data"]["universe_config"]["elem_distrib"] = None
    config["data"]["universe_config"]["physics_config"] = None
    return config


def default_config(physics_config=None, elem_distrib=None, log=False):
    config = {
        "infra": {
            "log": log,
        },
        "optimize_perceiver": {
            "epochs": 12,
            "branches": 4,
            "lr": 1e-2,
            "weight_decay": 1e-6,
        },
        "optimize_universe_config": {"epochs": 64, "lr": 1e-4},
        "data": {
            "n_univs": 2,
            "steps": 128,
            "n_cfs": 4,
            "start": 96,
            "universe_config": {
                "n_elems": 2,
                "n_atoms": 128,
                "n_dims": 2,
                "dt": 0.1,
                "physics_config": physics_config,
                "elem_distrib": elem_distrib,
                "batch_size": 16,
            },
        },
        "preprocessor": {
            "fourier_position_encoding_kwargs": {
                "num_bands": 9,
                "max_resolution": [1],
                "sine_only": True,
                "concat_pos": True,
            }
        },
        "encoder": {
            "z_index_dim": 64,
            "num_z_channels": 16,
            "num_cross_attend_heads": 4,
            "num_blocks": 4,
            "num_self_attends_per_block": 2,
            "num_self_attend_heads": 2,
            "dropout_prob": 0.1,
        },
        "decoder": {
            "output_num_channels": 2,  # Has to match n_dims
            "num_z_channels": 16,
            "use_query_residual": False,
            "fourier_position_encoding_kwargs": {
                "num_bands": 8,
                "max_resolution": [1],
                "sine_only": False,
                "concat_pos": True,
            },
        },
    }

    return config
