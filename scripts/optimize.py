from diotima.perceiver.optimize import optimize_universe_config, default_config, sanitize_config
from diotima.world.physics import default_elem_distrib, default_physics_config

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import numpy as np
import pickle
import wandb


print("[*] Initializing config...")
config = default_config(log=True)
config["data"]["universe_config"]["elem_distrib"] = default_elem_distrib(config["data"]["universe_config"]["n_elems"])
config["data"]["universe_config"]["physics_config"] = default_physics_config(config["data"]["universe_config"]["n_elems"])

if jax.process_index() == 0:
    print("[*] Initializing logging...")
    wandb.init(
        project="diotima",
        config=sanitize_config(default_config())
    )

print("[*] Initializing optimization...")
config = optimize_universe_config(config)

print("[*] Saving final config...")
pickle.dump(config, open("./config.pickle", "wb+"))
