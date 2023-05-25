from diotima.world import *
from diotima.physics import *
from einops import repeat
import matplotlib.pyplot as plt

n_steps = 10
universe_config = UniverseConfig(
    n_elems = 4,
    n_atoms = 20
)
universe = seed(universe_config)
universe = run(universe, n_steps)


def get_energy(atom_locs: Array, atom_elems: Array):
    return vmap(lambda loc, elem: compute_element_weighted_fields(
                        loc,
                        elem,
                        atom_locs,
                        universe_config
                    ))(atom_locs, atom_elems).energies.sum()


def get_energies():
    return vmap(lambda atom_locs: get_energy(
        atom_locs,
        universe.atom_elems
    ))(universe.locs_history)


energies = get_energies()
plt.plot(energies)
plt.show()
