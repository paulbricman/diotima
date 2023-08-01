from diotima.perceiver.optimize import optimize_universe_config, default_config, sanitize_config
from diotima.world.physics import default_elem_distrib, default_physics_config

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import pickle
import wandb


print("[*] Initializing config...")
config = default_config(log=True)
config["data"]["universe_config"]["elem_distrib"] = default_elem_distrib(config["data"]["universe_config"]["n_elems"])
config["data"]["universe_config"]["physics_config"] = default_physics_config(config["data"]["universe_config"]["n_elems"])

print("[*] Initializing logging...")
wandb.init(
    project="diotima",
    config=sanitize_config(default_config()),
    group="tpu-cluster"
)

print("[*] Initializing distributed job...")
jax.distributed.initialize(config["infra"]["coordinator_address"],
                           int(config["infra"]["num_hosts"]),
                           int(config["infra"]["process_id"]))

print("[*] Initializing optimization...")
# mesh_devices = jnp.array(jax.devices()).reshape((2, 4))
# with Mesh(*(mesh_devices, ('hosts', 'devices'))):
config = optimize_universe_config(config)

print("[*] Saving final config...")
pickle.dump(config, open("./config.pickle", "wb+"))
