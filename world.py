from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np


class PhysicsConfig(NamedTuple):
    # Physical constants to parametrize Lenia fields (1x per elem).
    mu_ks: np.ndarray
    sigma_ks: np.ndarray
    w_ks: np.ndarray

    # Physical constants to parametrize growth fields (1x per elem-elem).
    mu_gs: np.ndarray
    sigma_gs: np.ndarray

    # Physical constants to parametrize repulsion fields (1x per elem-elem).
    c_reps: np.ndarray


class UniverseConfig:
    def __init__(
            key: prng.PRNGKeyArray,
            n_elems: int = 4,
            n_atoms: int = 100,
            elem_distrib: np.ndarray = None,
            physics_config: PhysicsConfig = None
    ):
        assert n_elems > 0
        self.n_elems = n_elems

        assert n_atoms > 0
        self.n_atoms = n_atoms

        self.key = key
        if elem_distrib is None:
            self.key, subkey = jax.random.split(self.key)
            self.elem_distrib = jax.random.uniform(subkey, (self.n_elems,))
        else:
            assert elem_distrib.shape == (n_elems,)
            self.elem_distrib = elem_distrib

        if physics is None:
            self.load_default_physics_config()
        else:
            self.validate_physics_config(physics_config)
            self.physics_config = physics_config


    def validate_physics_config(physics_config: PhysicsConfig):
        elem_constants = [
            physics_config.mu_ks,
            physics_config.sigma_ks,
            physics_config.w_ks
        ]

        elem_elem_constants = [
            physics_config.mu_gs,
            physics_config.sigma_gs,
            physics_config.c_reps
        ]

        assert all([e.shape == (self.n_elems,) for e in elem_constants]
        assert all([e.shape == (self.n_elems, self.n_elems) for e in elem_constants]


    def load_default_physics_config():
        physics_config = PhysicsConfig()

        physics_config.mu_ks = np.tile(4.0, (self.n_elems))
        physics_config.sigma_ks = np.tile(1.0, (self.n_elems))
        physics_config.w_ks = np.tile(0.022, (self.n_elems))

        physics_config.mu_gs = np.tile(0.6, (self.n_elems, self.n_elems))
        physics_config.sigma_gs = np.tile(0.15, (self.n_elems, self.n_elems))
        physics_config.c_reps = np.tile(1.0, (self.n_elems, self.n_elems))

        self.physics_config = physics_config
