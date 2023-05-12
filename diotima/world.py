from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np
from jax import Array, jit, vmap, grad, random
from jax.lax import scan
import diotima.physics as physics


class UniverseConfig:
    def __init__(
            self,
            n_elems: int = 2,
            n_atoms: int = 2,
            n_dims: int = 2,
            elem_distrib: np.ndarray = None,
            physics_config: physics.PhysicsConfig = None,
            key: prng.PRNGKeyArray = None,
            dt: float = 0.1
    ):
        assert n_elems > 0
        self.n_elems = n_elems

        assert n_atoms > 0
        self.n_atoms = n_atoms

        assert n_dims > 0
        self.n_dims = n_dims

        assert dt > 0
        self.dt = dt

        if key is None:
            self.key = random.PRNGKey(0)
        else:
            self.key = key

        if elem_distrib is None:
            self.key, subkey = random.split(self.key)
            self.elem_distrib = random.uniform(subkey, (self.n_elems,))
        else:
            assert elem_distrib.shape == (n_elems,)
            self.elem_distrib = elem_distrib

        if physics_config is None:
            self.physics_config = physics.default_physics_config(self.n_elems)
        else:
            physics.validate_physics_config(physics_config, self.n_elems)
            self.physics_config = physics_config


class Universe:
    def __init__(
        self,
        universe_config: UniverseConfig = None,
        key=None,
    ):
        if universe_config is None:
            self.universe_config = UniverseConfig()
        else:
            self.universe_config = universe_config

        if key is None:
            self.key = random.PRNGKey(0)
        else:
            self.key = key

        self.step = 0
        self.seed()

    def seed(self):
        self.key, key_locs, key_elems = random.split(self.key, num=3)
        self.atom_locs = random.normal(key_locs, shape=(
            self.universe_config.n_atoms,
            self.universe_config.n_dims
        ))
        # TODO: Implement Gumbel-Softmax sampling based on elem_distrib in
        # universe_config
        self.atom_elems = random.uniform(key_elems, shape=(
            self.universe_config.n_atoms,
            self.universe_config.n_elems
        ))

    def locs_after_steps(self, n_steps: int = 1):
        def pure_step(atom_locs, _): return physics.step(
            atom_locs,
            self.atom_elems,
            self.universe_config
        )

        updated_locs, history = scan(pure_step, self.atom_locs, None, n_steps)
        return updated_locs, history

    def run(self, n_steps: int = 1):
        self.atom_locs = self.locs_after_steps(n_steps)[0]


class MultiverseConfig:
    def __init__(
            n_branches: int,
            universe_configs: Array = None
    ):
        assert n_branches > 0
        self.n_branches = n_branches

        if universe_configs is None:
            self.universe_configs = [UniverseConfig()]
        else:
            self.universe_configs = universe_configs
