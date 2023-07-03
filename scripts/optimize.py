from diotima.perceiver.optimize import optimize_universe_config, default_config, checkpoint

import jax

import pickle


config = default_config()
jax.distributed.initialize(config.infra.coordinator_address,
                           config.infra.num_hosts,
                           config.infra.process_id)

config = optimize_universe_config(config)
pickle.dump(config, open("./config.pickle", "wb+"))

