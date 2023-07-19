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
from jax.experimental.host_callback import id_tap, id_print, call

from einops import repeat, rearrange, reduce
from typing import Tuple, NamedTuple, Dict
from safetensors.flax import save_file
from pathlib import Path
import os
import wandb


OptState = Tuple[optax.TraceState,
                 optax.ScaleByScheduleState,
                 optax.ScaleState]


class Data(NamedTuple):
    universe_config: UniverseConfig
    atom_elems: Array
    idx_history: Array
    locs_history: Array
    locs_future: Array
    pred_locs_future: Array


class Agents(NamedTuple):
    flows: Array
    substrates: Array


def synth_universe_data(config: Dict):
    universe_config = UniverseConfig(**config["data"]["universe_config"])
    universe = seed(universe_config, next(config["rng"]))
    universe = run(universe, config["data"]["steps"])

    cfs = spawn_counterfactuals(
        universe,
        config["data"]["start"],
        config["data"]["n_cfs"],
        next(config["rng"])
    )

    def cf_to_datum(cf): return Data(
        universe_config=None,
        atom_elems=None,
        idx_history=None,
        locs_history=None,
        locs_future=cf.locs_history[config["data"]["start"]:],
        pred_locs_future=None,
    )

    data = jax.vmap(cf_to_datum)(cfs)
    data = data._replace(universe_config=universe.universe_config)
    data = data._replace(atom_elems=universe.atom_elems)
    data = data._replace(idx_history=jnp.arange(config["data"]["start"]))
    data = data._replace(
        locs_history=universe.locs_history[:config["data"]["start"]])

    return data


def synth_data(config: Dict):
    def pure_synth_universe_data(_): return synth_universe_data(config)
    return jax.vmap(pure_synth_universe_data)(jnp.arange(config["data"]["n_univs"]))


def raw_forward(data: Data, config: Dict, is_training: bool):
    future_steps = config["data"]["steps"] - config["data"]["start"]
    n_atoms = data.locs_history.shape[2]
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
        b=config["optimize_perceiver"]["branches"])
    decoder_query_space_pos = repeat(
        atom_locs,
        "u a l -> u (b a) l",
        b=config["optimize_perceiver"]["branches"])
    decoder_query_branch_pos = repeat(
        jnp.arange(
            config["optimize_perceiver"]["branches"]),
        "b -> u (b a) 1",
        u=config["data"]["n_univs"],
        a=n_atoms)

    decoder_query_pos = jnp.concatenate(
        (decoder_query_space_pos, decoder_query_branch_pos), axis=2)

    decoder_query, _, decoder_query_without_pos = preprocessor(
        decoder_query_input, decoder_query_pos)

    latents, scores = encoder(input, encoder_query, is_training=is_training)

    def decode_slot(latent):
        latent = rearrange(latent, "u z -> u 1 z")
        return decoder(decoder_query, latent, is_training=is_training)

    one_step_preds = jax.vmap(decode_slot, in_axes=1)(latents)
    agg_one_step_preds = reduce(one_step_preds, "z u a l -> u a l", "sum")
    flows = rearrange(one_step_preds, "z u (b a) l -> u b z a l", b=config["optimize_perceiver"]["branches"])

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = jax.lax.scan(
            inject_one_step,
            decoder_query_space_pos,
            None,
            future_steps)
        forecast = rearrange(
            forecast[1],
            "t u (b a) l -> u t a b l",
            b=config["optimize_perceiver"]["branches"])
        return forecast

    forecast = preds_to_forecast()

    enriched_data = Data(
        data.universe_config,
        data.atom_elems,
        data.idx_history,
        data.locs_history,
        data.locs_future,
        forecast
    )

    agents = Agents(
        flows,
        compute_attn_rollouts(scores)
    )

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
            norm_factor = repeat(jnp.sum(self, axis=(1, 2)), "sa -> sa z1 z2", z1=z, z2=z)
            self /= norm_factor

        # Relevant: https://github.com/deepmind/deepmind-research/issues/253
        first = [cross @ self[0]]
        rest = list(self[1:])
        return jnp.linalg.multi_dot(first + rest)

    return jax.vmap(compute_attn_rollout)(self, cross)


def init_opt(config: Dict):
    data = synth_data(config)

    forward = hk.transform_with_state(raw_forward)
    params, state = forward.init(next(config["rng"]), data, config, True)

    optim = optax.adamw(config["optimize_perceiver"]["lr"], eps_root=1e-10)
    opt_state = optim.init(params)

    return params, state, opt_state, optim, forward


def loss(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        data: Data,
        config: Dict
):
    out, new_state = forward.apply(params, state, next(config["rng"]), data, config, is_training=True)
    data, agents = out
    error = distance(data.pred_locs_future, data.locs_future)
    cond_log(config, {"optimize_perceiver_loss": error})
    return -error, new_state


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
        dist_from_cfs = jnp.mean(dist_from_cfs)

        # Iterate over bs to estimate shortest way to cfs
        dist_from_bs = jax.vmap(lambda b: compute_distances(b, cfs))(bs)
        dist_from_bs = jnp.mean(dist_from_bs)

        dist = dist_from_cfs + dist_from_bs
        return dist

    # Iterate over batches
    return jnp.mean(
        jax.vmap(lambda cfs, bs: compute_batch_distances(cfs, bs))(cfs, bs))


def compute_l2_penalty(params):
    theta, unravel = jax.flatten_util.ravel_pytree(params)
    return jnp.sum(theta ** 2)


def backward(
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        forward,
        optimizer,
        data: Data,
        config: Dict,
        epoch: Array
):
    grads, new_state = jax.grad(loss, has_aux=True)(params, state, opt_state, forward, data, config)

    def agg_grads(grad):
        return jax.lax.pmean(grad, axis_name="devices")

    def no_agg_grads(grad):
        return grad

    grads = jax.lax.cond(
        epoch % config["optimize_perceiver"]["agg_every"] == 0,
        agg_grads,
        no_agg_grads,
        grads
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, new_state, opt_state


def optimize_perceiver(
        config: Dict,
        perceiver_params: hk.Params,
        perceiver_state: hk.State,
        perceiver_opt_state: OptState,
        perceiver_optim,
        forward
):
    def scanned_backward(opt_perceiver_state, _):
        data = synth_data(config)
        params, state, opt_state, epoch = opt_perceiver_state
        new_params, new_state, new_opt_state = backward(
            params, state, opt_state, forward, perceiver_optim, data, config, epoch)

        cond_checkpoint(config, epoch, new_params, new_state)

        opt_perceiver_state = new_params, new_state, new_opt_state, epoch + 1
        return opt_perceiver_state, opt_perceiver_state

    def scan_backward(_):
        init_opt_perceiver_state = (perceiver_params, perceiver_state, perceiver_opt_state, 0)
        # Weirdest bug ever:
        # Jitting without scanning works here, so does scanning with jit disabled.
        # Yet scanning with jit enabled fails.
        # Solution? Jit in advance, scan with locally disabled jit.
        jitted_scanned_backwards = jax.jit(scanned_backward)
        with jax.disable_jit():
            carry = jax.lax.scan(
                jitted_scanned_backwards,
                init_opt_perceiver_state,
                None,
                config["optimize_perceiver"]["epochs"]
            )[0]
        return carry
    return jax.pmap(scan_backward, axis_name="devices")(jnp.arange(jax.local_device_count()))


def optimize_universe_config(config: Dict):
    # wandb.init(
    #     project="diotima",
    #     config=sanitize_config(default_config()),
    #     group="tpu-cluster"
    # )
    log = config["infra"]["log"]

    def universe_config_state_to_config(universe_config_state):
        physics_config, elem_distrib = universe_config_state
        return default_config(physics_config, elem_distrib, log)

    def compute_param_count(universe_config_state):
        config = universe_config_state_to_config(universe_config_state)

        params, perceiver_state, perceiver_opt_state, perceiver_optim, forward = init_opt(config)

        opt_perceiver_carry = optimize_perceiver(
            config, params, perceiver_state, perceiver_opt_state, perceiver_optim, forward)
        params, perceiver_state, perceiver_opt_state, epoch = opt_perceiver_carry
        return -compute_l2_penalty(params)

    def scan_optimize_universe_config(_):
        def init():
            universe_config_state = (config["data"]["universe_config"]["physics_config"],
                                 config["data"]["universe_config"]["elem_distrib"])
            universe_config_optim = optax.adamw(config["optimize_universe_config"]["lr"], eps_root=1e-10)
            universe_config_opt_state = universe_config_optim.init(universe_config_state)
            return universe_config_state, universe_config_opt_state, universe_config_optim

        universe_config_state, universe_config_opt_state, universe_config_optim = init()

        def scanned_optimize_universe_config(opt_universe_config_state, _):
            universe_config_state, universe_config_opt_state = opt_universe_config_state
            param_count, grads = jax.value_and_grad(compute_param_count)(universe_config_state)
            cond_log(config, {"optimize_universe_config_loss": param_count})

            grads = jax.lax.pmean(grads, axis_name="hosts")
            updates, universe_config_opt_state = universe_config_optim.update(
                grads, universe_config_opt_state, universe_config_state)
            universe_config_state = optax.apply_updates(
                universe_config_state, updates)
            opt_universe_config_state = universe_config_state, universe_config_opt_state

            return opt_universe_config_state, opt_universe_config_state

        return jax.lax.scan(scanned_optimize_universe_config,
                            (universe_config_state,
                             universe_config_opt_state),
                            None,
                            config["optimize_universe_config"]["epochs"])

    opt_universe_config_carry, history = jax.experimental.maps.xmap(scan_optimize_universe_config,
                                                                    in_axes=["hosts", ...],
                                                                    out_axes=["hosts", ...])(jnp.arange(config["infra"]["num_hosts"]))
    # opt_universe_config_carry, history = jax.pmap(scan_optimize_universe_config, axis_name="hosts")(jnp.arange(config["infra"]["num_hosts"]))

    universe_config_state, opt_state = opt_universe_config_carry
    config = universe_config_state_to_config(universe_config_state)
    return config


def checkpoint(
        params: hk.Params,
        state: hk.State,
        filepath: str = "."
):
    root = Path(filepath)

    save_file(params, filename=root / "params.safetensors")
    save_file(state, filename=root / "state.safetensors")


def cond_log(config: Dict, payload):
    def log_wrapper(payload, transforms):
        wandb.log(payload)

    def tap_payload(payload):
        id_tap(log_wrapper, payload)
        return None

    jax.lax.cond(config["infra"]["log"], lambda: tap_payload(payload), lambda: None)


def cond_checkpoint(config: Dict, epoch, params, state):
    def tap_checkpoint(params, state):
        id_tap(lambda args, transforms: checkpoint(**args), {
            "params": params,
            "state": state
        })
        return None

    jax.lax.cond((epoch % config["optimize_perceiver"]["ckpt_every"] == 0 and epoch > 0), lambda: tap_checkpoint(params, state), lambda: None)


def jit_print(payload):
    call(lambda _: print("(*) ", end=""), None)
    id_print(payload)


def sanitize_config(config: Dict):
    config["data"]["universe_config"]["elem_distrib"] = None
    config["data"]["universe_config"]["physics_config"] = None
    config["rng"] = None
    return config


def default_config(physics_config=None, elem_distrib=None, log=False):
    config = {
        "infra": {
            "coordinator_address": os.environ.get("JAX_COORD_ADDR", "127.0.0.1:8888"),
            "num_hosts": os.environ.get("JAX_NUM_HOSTS", 1),
            "process_id": os.environ.get("JAX_PROCESS_ID", 0),
            "log": log
        },
        "optimize_perceiver": {
            "epochs": 2,
            "branches": 2,
            "agg_every": 2,
            "ckpt_every": 2,
            "lr": 1e-4,
            "weight_decay": 1e-8
        },
        "optimize_universe_config": {
            "epochs": 2,
            "lr": 1e-4,
            "weight_decay": 1e-8
        },
        "data": {
            "n_univs": 2,
            "steps": 6,
            "n_cfs": 4,
            "start": 4,
            "universe_config": {
                "n_elems": 2,
                "n_atoms": 2,
                "n_dims": 2,
                "dt": 0.1,
                "physics_config": physics_config,
                "elem_distrib": elem_distrib
            }
        },
        "rng": hk.PRNGSequence(jax.random.PRNGKey(0)),
        "preprocessor": {
            "fourier_position_encoding_kwargs": {
                "num_bands": 1,
                "max_resolution": [1],
                "sine_only": False,
                "concat_pos": True
            }
        },
        "encoder": {
            "z_index_dim": 4,
            "num_z_channels": 4,
            "num_cross_attend_heads": 1,
            "num_blocks": 1,
            "num_self_attends_per_block": 2,
            "num_self_attend_heads": 2
        },
        "decoder": {
            "output_num_channels": 2,  # Has to match n_dims
            "num_z_channels": 4,
            "use_query_residual": False,
            "position_encoding_type": "fourier",
            "fourier_position_encoding_kwargs": {
                "num_bands": 1,
                "max_resolution": [1],
                "sine_only": True,
                "concat_pos": False
            }
        }
    }

    return config
