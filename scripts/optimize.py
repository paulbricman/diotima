from diotima.perceiver.optimize import optimize_universe_config, default_config, checkpoint
from diotima.world.physics import default_elem_distrib, default_physics_config

import jax

import pickle


print("[*] Initializing config...")
config = default_config(log=True)
config["data"]["universe_config"]["elem_distrib"] = default_elem_distrib(config["data"]["universe_config"]["n_elems"])
config["data"]["universe_config"]["physics_config"] = default_physics_config(config["data"]["universe_config"]["n_elems"])

print("[*] Initializing distributed job...")
jax.distributed.initialize(config["infra"]["coordinator_address"],
                           int(config["infra"]["num_hosts"]),
                           int(config["infra"]["process_id"]))

print("[*] Initializing optimization...")
config = optimize_universe_config(config)

print("[*] Saving final config...")
pickle.dump(config, open("./config.pickle", "wb+"))
