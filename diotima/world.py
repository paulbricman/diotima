from typing import NamedTuple
from collections import namedtuple
from jax._src import prng
import jax.numpy as np
from jax import Array, jit, vmap, grad, random
from jax.lax import fori_loop


class PhysicsConfig(NamedTuple):
    """
    Contains fundamental physical constants which parametrize the physics of a universe.
    Each field from the first group requires one value per element type.
    Each field from the second group requires one value per element-element combination.
    """
    mu_ks: Array
    sigma_ks: Array
    w_ks: Array

    mu_gs: Array
    sigma_gs: Array
    c_reps: Array


class UniverseConfig:
    def __init__(
            self,
            n_elems: int = 4,
            n_atoms: int = 10,
            n_dims: int = 2,
            elem_distrib: np.ndarray = None,
            physics_config: PhysicsConfig = None,
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
            self.physics_config = self.default_physics_config()
        else:
            self.validate_physics_config(physics_config)
            self.physics_config = physics_config

    def validate_physics_config(self, physics_config: PhysicsConfig):
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
        assert all([e.shape == (self.n_elems, self.n_elems,)
                   for e in elem_elem_constants])

    def default_physics_config(self):
        return PhysicsConfig(
            np.tile(4.0, (self.n_elems)),
            np.tile(1.0, (self.n_elems)),
            np.tile(0.022, (self.n_elems)),
            np.tile(0.6, (self.n_elems, self.n_elems)),
            np.tile(0.15, (self.n_elems, self.n_elems)),
            np.tile(1.0, (self.n_elems, self.n_elems))
        )


class Fields(NamedTuple):
    """
    Utility abstraction to package up fields.
    """
    matters: Array
    attractions: Array
    repulsions: Array
    energies: Array


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
        # TODO: Implement Gumbel-Softmax sampling based on elem_distrib in universe_config
        self.atom_elems = random.uniform(key_elems, shape=(
            self.universe_config.n_atoms,
            self.universe_config.n_elems
        ))

    def peak(self, x, mu, sigma):
        return np.exp(-((x - mu) / sigma) ** 2)

    def compute_matter_fields(self, distances):
        def compute_matter_field(elem_idx):
            return self.peak(
                distances,
                self.universe_config.physics_config.mu_ks[elem_idx],
                self.universe_config.physics_config.sigma_ks[elem_idx]
            ).sum() * self.universe_config.physics_config.w_ks[elem_idx]

        return vmap(compute_matter_field)(
            np.arange(self.universe_config.n_elems))

    def compute_attraction_fields(self, matters):
        def send_attraction_fields(from_idx):
            def send_attraction_field(to_idx):
                return self.peak(
                    matters[from_idx],
                    self.universe_config.physics_config.mu_gs[from_idx][to_idx],
                    self.universe_config.physics_config.sigma_gs[from_idx][to_idx])

            # All attraction fields sent from element from_idx
            return vmap(send_attraction_field)(
                np.arange(self.universe_config.n_elems))

        # All attraction fields sent from all elements
        return vmap(send_attraction_fields)(
            np.arange(self.universe_config.n_elems))

    def compute_repulsion_fields(self, distances):
        def send_repulsion_fields(from_idx):
            def send_repulsion_field(to_idx):
                return self.universe_config.physics_config.c_reps[from_idx][to_idx] / 2 * (
                    (1.0 - distances[from_idx]).clip(0.0) ** 2).sum()

            # All repulsion fields sent from element from_idx
            return vmap(send_repulsion_field)(
                np.arange(self.universe_config.n_elems))

        # All repulsion fields sent from all elements
        return vmap(send_repulsion_fields)(
            np.arange(self.universe_config.n_elems))

    def fields(self, loc):
        distances = np.sqrt(np.square(loc - self.atom_locs).sum(-1).clip(1e-10))
        matters = self.compute_matter_fields(distances)
        attractions = self.compute_attraction_fields(matters)
        repulsions = self.compute_repulsion_fields(distances)
        energies = repulsions - attractions
        return Fields(matters, attractions, repulsions, energies)

    def element_weighted_fields(self, loc, elem):
        fields = self.fields(loc)
        matters = np.dot(fields.matters, elem)
        attractions = np.dot(fields.attractions.sum(axis=1), elem)
        repulsions = np.dot(fields.repulsions.sum(axis=1), elem)
        energies = repulsions -  attractions
        return Fields(matters, attractions, repulsions, energies)

    def motion(self):
        grad_energies = grad(lambda loc, elem: self.element_weighted_fields(loc, elem).energies)
        return -vmap(grad_energies)(self.atom_locs, self.atom_elems)

    def run(self, n_steps: int = 1):
        # TODO: Purify function
        def step():
            self.atom_locs += self.universe_config.dt * self.motion()

        for _ in range(n_steps):
            step()


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
