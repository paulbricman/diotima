import jax.numpy as np
from jax import Array, vmap, jit
from jax.nn import softmax
from jax.lax import while_loop, cond
from typing import Tuple
from functools import partial
from diotima.utils import norm, normalize
import PIL


def signed_distance(
        atom_locs: Array,
        loc: Array,
        atom_radius: float = 0.5,
        temperature: float = 1e-1,
):
    distances = np.sqrt(np.square(loc - atom_locs).sum(-1).clip(1e-10)) - atom_radius
    weights = softmax(-distances / temperature)
    distance = np.dot(distances, weights) / np.sum(weights)
    return distance


def raymarch(
        sdf,
        init_loc: Array,
        dir: Array,
        steps_threshold: int = 25,
        proximity_threshold: float = 1e-3,
        beyond_threshold: float = 1e7,
):
    def stop(state):
        idx, loc = state
        true_func = lambda: True
        false_func = lambda: False

        # March until close to a surface, exhausted steps, or ended up far off.
        is_exhausted = cond(np.greater(idx, steps_threshold), true_func, false_func)
        is_close = cond(np.isclose(
            sdf(loc),
            0.,
            atol=proximity_threshold
        ), true_func, false_func)
        is_beyond = cond(np.greater(norm(loc), beyond_threshold), true_func, false_func)

        return cond(is_exhausted, false_func,
                    lambda: cond(is_close, false_func,
                                 lambda: cond(is_beyond, false_func, true_func)))

    def step(state):
        idx, loc = state
        return idx + 1, loc + sdf(loc) * dir

    dir = normalize(dir)
    steps, final_loc = while_loop(stop, step, (0, init_loc))
    return steps, final_loc


def radiate(
        camera_forward: Array,
        view_size: Tuple[int, int],
        fx: float = 0.6

):
    # Compute convenient unit vectors.
    world_up = np.array([0., 1., 0.])
    camera_right = np.cross(camera_forward, world_up)
    camera_down = np.cross(camera_right, camera_forward)
    R = normalize(np.vstack([
        camera_right,
        camera_down,
        camera_forward
    ]))
    w, h = view_size
    fy = fx / w * h
    y, x = np.mgrid[fy:-fy:h * 1j, -fx:fx:w * 1j].reshape(2, -1)
    return normalize(np.c_[x, y, np.ones_like(x)]) @ R


def render(
        view_size: Tuple[int, int],
        camera_loc: Array,
        atom_locs: Array
):
    w, h = view_size
    ray_dirs = radiate(-camera_loc, view_size)
    pure_signed_distance = partial(signed_distance, atom_locs)
    pure_raymarch = partial(raymarch, pure_signed_distance, camera_loc)
    steps, hits = vmap(pure_raymarch)(ray_dirs)
    return steps.reshape(h, w), hits.reshape(h, w, 3)

