from typing import NamedTuple, Array
from collections import namedtuple
from jax._src import prng
import jax.numpy as np


class PhysicsConfig(NamedTuple):
    mu_ks: np.ndarray
    sigma_ks: np.ndarray
    w_ks: np.ndarray

    mu_gs: np.ndarray
    sigma_gs: np.ndarray
    c_reps: np.ndarray


class UniverseConfig:
    def __init__(
            n_elems: int = 4,
            n_atoms: int = 100,
            n_dims: int = 2,
            elem_distrib: np.ndarray = None,
            physics_config: PhysicsConfig = None
    ):
        assert n_elems > 0
        self.n_elems = n_elems

        assert n_atoms > 0
        self.n_atoms = n_atoms

        assert n_dims > 0
        self.n_dims = n_dims

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

        assert all([e.shape == (self.n_elems,) for e in elem_constants])
        assert all([e.shape == (self.n_elems, self.n_elems)
                   for e in elem_constants])

    def load_default_physics_config():
        physics_config = PhysicsConfig()

        physics_config.mu_ks = np.tile(4.0, (self.n_elems))
        physics_config.sigma_ks = np.tile(1.0, (self.n_elems))
        physics_config.w_ks = np.tile(0.022, (self.n_elems))

        physics_config.mu_gs = np.tile(0.6, (self.n_elems, self.n_elems))
        physics_config.sigma_gs = np.tile(0.15, (self.n_elems, self.n_elems))
        physics_config.c_reps = np.tile(1.0, (self.n_elems, self.n_elems))

        self.physics_config = physics_config
        # TODO: Test load defaults and validate


class Universe:
    def __init__(
        universe_config: UniverseConfig = None,
        key=None,
    ):
        if universe_config is None:
            self.universe_config = UniverseConfig()
        else:
            self.universe_config = universe_config

        if key is None:
            self.key = jax.random.PRNGKey(0)
        else:
            self.key = key

        self.step = 0
        self.seed()

    @jit
    def seed():
        self.key, key_locs, key_elems = jax.random.split(self.key, num=3)
        self.atom_locs = jax.random.normal(size=(
            self.universe_config.n_atoms,
            self.universe_config.n_dims
        ))
        self.atom_elems = jax.random.uniform(size=(
            self.universe_config.n_atoms,
            self.universe_config.n_elems
        ))

    @jit
    def run(n_steps: int = 1):
        def peak(x, mu, sigma):
            return jp.exp(-((x - mu) / sigma) ** 2)

        def compute_lenia_fields(distances):
            def compute_lenia_field(elem_idx):
                return peak(
                    distances,
                    self.universe_config.physics_config.mu_ks[elem_idx],
                    self.universe_config.physics_config.sigma_ks[elem_idx]
                ).sum() * self.universe_config.physics_config.w_ks[elem_idx]

            return jax.vmap(compute_lenia_field)(
                np.arange(self.universe_config.n_elems))

        def compute_growth_fields(potentials):
            def send_growth_fields(from_idx):
                def send_growth_field(to_idx):
                    return peak(
                        potentials,
                        self.universe_config.physics_config.mu_gs[from_idx][to_idx],
                        self.universe_config.physics_config.sigma_gs[from_idx][to_idx])

                # all growth fields sent from from_idx
                return jax.vmap(send_growth_field)(
                    np.arange(self.universe_config.n_elems))

            # all growth fields sent from all from idx
            return jax.vmap(send_growth_fields)(
                np.arange(self.universe_config.n_elems))

        def compute_repulsion_fields(distances):
            def send_repulsion_fields(from_idx):
                def send_repulsion_field(to_idx):
                    return self.universe_config.physics_config.c_reps[from_idx][to_idx] / 2 * (
                        (1.0 - distances).clip(0.0) ** 2).sum()

                # all repulsion fields sent from from_idx
                return jax.vmap(send_growth_field)(
                    np.arange(self.universe_config.n_elems))

            # all repulsion fields sent from all idx
            return jax.vmap(send_growth_fields)(
                np.arange(self.universe_config.n_elems))

        def fields(loc):
            # TODO: Write out sizes
            # TODO: Test shapes are correct with various n_elems
            distances = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
            potentials = compute_lenia_fields(distances)
            growths = compute_growth_fields(potentials)
            repulsions = compute_repulsion_fields(distances)
            energies = repulsions - growths

        def motion():
            # TODO: Take into account atom elements to weigh energy gradients
            # TODO: Test motions are invariant to n_elems if all elem constants identical
            # See motion_f

        # See step_f, odeint_euler


class MultiverseConfig:
    def __init__(
            n_branches: int,
            universe_configs: Array[UniverseConfig] = None
    ):
        assert n_branches > 0
        self.n_branches = n_branches

        if universe_configs is None:
            self.universe_configs = [UniverseConfig()]
        else:
            self.universe_configs = universe_configs
