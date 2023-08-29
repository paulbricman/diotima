from diotima.world.universe import UniverseConfig, seed, run, spawn_counterfactuals
import diotima.world.physics as physics
from diotima.perceiver.io_processors import DynamicPointCloudPreprocessor
from diotima.perceiver.perceiver import PerceiverEncoder, BasicDecoder

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap
from jax._src.prng import PRNGKeyArray
from jax import Array
import haiku as hk
import optax

from einops import repeat, rearrange, reduce
from typing import Tuple, NamedTuple, Dict
from safetensors.flax import save_file
from pathlib import Path
import os
import wandb


OptState = Tuple[optax.TraceState,
                 optax.ScaleByScheduleState,
                 optax.ScaleState]


hk.vmap.require_split_rng = False


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
    universe = seed(universe_config, config["rng"])
    universe = run(universe, config["data"]["steps"])

    cfs = spawn_counterfactuals(
        universe,
        config["data"]["start"],
        config["data"]["n_cfs"],
        config["rng"]
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
    synth_data_rngs = jax.random.split(config["rng"], config["data"]["n_univs"])
    # Local config copy to avoid side effects:
    config = config.copy()

    def pure_synth_universe_data(univ):
        config["rng"] = synth_data_rngs[univ]
        return synth_universe_data(config)
    
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

    one_step_preds = hk.vmap(decode_slot, in_axes=1)(latents)
    agg_one_step_preds = reduce(one_step_preds, "z u a l -> u a l", "sum")
    flows = rearrange(one_step_preds, "z u (b a) l -> u b z a l", b=config["optimize_perceiver"]["branches"])

    def preds_to_forecast():
        def inject_one_step(state, _):
            state = state + agg_one_step_preds
            return state, state

        forecast = hk.scan(
            inject_one_step,
            decoder_query_space_pos,
            None,
            future_steps)
        forecast = rearrange(
            forecast[1],
            "t u (b a) l -> u b t a l",
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

    forward = hk.transform(raw_forward)
    params = forward.init(config["rng"], data, config, True)

    optim = optax.adamw(config["optimize_perceiver"]["lr"], eps_root=1e-10)
    opt_state = optim.init(params)

    return params, opt_state, optim, forward


def loss(
        params: hk.Params,
        forward,
        data: Data,
        config: Dict
):
    out = forward.apply(params, config["rng"], data, config, is_training=True)
    data, agents = out
    error = distance(data.pred_locs_future, data.locs_future)

    return -error


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
    return jnp.mean(
        jax.vmap(lambda cfs, bs: compute_batch_distances(cfs, bs))(cfs, bs))


def compute_l2_penalty(params):
    theta, unravel = jax.flatten_util.ravel_pytree(params)
    return jnp.sum(theta ** 2)


def backward(
        params: hk.Params,
        opt_state: OptState,
        forward,
        optimizer,
        data: Data,
        config: Dict,
        epoch: Array
):
    perceiver_loss, grads = jax.value_and_grad(loss)(params, forward, data, config)

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

    return params, opt_state, perceiver_loss


def optimize_perceiver(
        config: Dict,
        perceiver_params: hk.Params,
        perceiver_opt_state: OptState,
        perceiver_optim,
        forward
):
    optimize_perceiver_rngs = jax.random.split(config["rng"], config["optimize_perceiver"]["epochs"])

    def scanned_backward(opt_perceiver_state, _):
        params, opt_state, opt_perceiver_epoch, _ = opt_perceiver_state
        config["rng"] = optimize_perceiver_rngs[opt_perceiver_epoch]
        data = synth_data(config)
        params, opt_state, perceiver_loss = backward(
            params, opt_state, forward, perceiver_optim, data, config, opt_perceiver_epoch)

        opt_perceiver_state = params, opt_state, opt_perceiver_epoch + 1, perceiver_loss
        return opt_perceiver_state, opt_perceiver_state

    def scan_backward(_):
        init_opt_perceiver_state = (perceiver_params, perceiver_opt_state, 0, 0)

        carry, history = jax.lax.scan(
            scanned_backward,
            init_opt_perceiver_state,
            None,
            config["optimize_perceiver"]["epochs"])
        return carry, history
    return jax.pmap(scan_backward, axis_name="devices")(jnp.arange(jax.local_device_count()))


def optimize_universe_config(config: Dict):
    log = config["infra"]["log"]
    optimize_universe_config_rngs = jax.random.split(config["rng"], config["optimize_universe_config"]["epochs"])

    def universe_config_state_to_config(universe_config_state):
        physics_config, elem_distrib = universe_config_state
        return default_config(physics_config, elem_distrib, log)

    def compute_sophistication(universe_config_state, opt_universe_epoch):
        config = universe_config_state_to_config(universe_config_state)
        config["rng"] = optimize_universe_config_rngs[opt_universe_epoch]

        params, perceiver_opt_state, perceiver_optim, forward = init_opt(config)

        opt_perceiver_carry, opt_perceiver_history = optimize_perceiver(
            config, params, perceiver_opt_state, perceiver_optim, forward)
        params, perceiver_opt_state, epoch, _ = opt_perceiver_carry

        return -compute_l2_penalty(params), (params, opt_perceiver_history[3])

    def scan_optimize_universe_config(_):
        def init():
            universe_config_state = (config["data"]["universe_config"]["physics_config"],
                                    config["data"]["universe_config"]["elem_distrib"])
            universe_config_optim = optax.adamw(config["optimize_universe_config"]["lr"], eps_root=1e-10)
            universe_config_opt_state = universe_config_optim.init(universe_config_state)
            return universe_config_state, universe_config_opt_state, universe_config_optim

        universe_config_state, universe_config_opt_state, universe_config_optim = init()

        @jax.checkpoint
        def scanned_optimize_universe_config(opt_universe_config_state, _):
            universe_config_state, universe_config_opt_state, opt_universe_epoch, _, _ = opt_universe_config_state
            value, grads = jax.value_and_grad(compute_sophistication, has_aux=True)(universe_config_state, opt_universe_epoch)
            param_count, aux = value
            params, perceiver_loss_history = aux

            updates, universe_config_opt_state = universe_config_optim.update(
                grads, universe_config_opt_state, universe_config_state)
            universe_config_state = optax.apply_updates(
                universe_config_state, updates)
            opt_universe_config_state = universe_config_state, universe_config_opt_state, opt_universe_epoch + 1, param_count, perceiver_loss_history

            return opt_universe_config_state, opt_universe_config_state

        return jax.lax.scan(scanned_optimize_universe_config,
                            (universe_config_state,
                            universe_config_opt_state,
                            0,
                            0,
                            jnp.zeros((config["infra"]["num_devices"], config["optimize_perceiver"]["epochs"]))
                            ),
                            None,
                            config["optimize_universe_config"]["epochs"])

    opt_universe_config_carry, opt_universe_config_history = scan_optimize_universe_config(None)

    universe_config_state, opt_state, _, _, _ = opt_universe_config_carry
    config = universe_config_state_to_config(universe_config_state)

    _, _, _, param_count, perceiver_loss_history = opt_universe_config_history

    # Good old for loops for non-JIT, non-JVP logging.
    # Aggregate across hosts and devices.
    if config["infra"]["log"] and jax.process_index() == 0:
        for idx, epoch in enumerate(param_count):
            wandb.log({"optimize_universe_config_param_count": epoch}, step=idx)
        
        perceiver_loss_history = reduce(perceiver_loss_history, "ue d pe -> (ue pe)", "mean")
        for idx, epoch in enumerate(perceiver_loss_history):
            wandb.log({"optimize_perceiver_loss": epoch}, step=idx)

    return config


def checkpoint(
        params: hk.Params,
        filepath: str = "."
):
    root = Path(filepath)

    save_file(params, filename=root / "params.safetensors")


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
            "num_devices": jax.local_device_count(),
            "log": log
        },
        "optimize_perceiver": {
            "epochs": 8,
            "branches": 4,
            "agg_every": 4,
            "lr": 1e-3,
            "weight_decay": 1e-8
        },
        "optimize_universe_config": {
            "epochs": 8,
            "lr": 1e-3,
            "weight_decay": 1e-8
        },
        "data": {
            "n_univs": 8,
            "steps": 16,
            "n_cfs": 4,
            "start": 8,
            "universe_config": {
                "n_elems": 2,
                "n_atoms": 16,
                "n_dims": 2,
                "dt": 0.1,
                "physics_config": physics_config,
                "elem_distrib": elem_distrib
            }
        },
        "rng": jax.random.PRNGKey(0),
        "preprocessor": {
            "fourier_position_encoding_kwargs": {
                "num_bands": 4,
                "max_resolution": [1],
                "sine_only": False,
                "concat_pos": True
            }
        },
        "encoder": {
            "z_index_dim": 8,
            "num_z_channels": 8,
            "num_cross_attend_heads": 1,
            "num_blocks": 2,
            "num_self_attends_per_block": 2,
            "num_self_attend_heads": 2,
            "dropout_prob": 0.2
        },
        "decoder": {
            "output_num_channels": 2,  # Has to match n_dims
            "num_z_channels": 8,
            "use_query_residual": False,
            "position_encoding_type": "fourier",
            "fourier_position_encoding_kwargs": {
                "num_bands": 4,
                "max_resolution": [1],
                "sine_only": True,
                "concat_pos": False
            }
        }
    }

    return config
