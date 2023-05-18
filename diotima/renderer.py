import jax.numpy as np
from jax import Array, vmap, jit, grad
from jax.nn import softmax
from jax.lax import while_loop, cond, fori_loop
from typing import Tuple
from functools import partial
from jax._src import prng
from diotima.utils import norm, normalize
import PIL


def signed_distance(
        atom_locs: Array,
        loc: Array,
        atom_radius: float = 0.2,
        temperature: float = 1e-1,
):
    distances = norm(loc - atom_locs) - atom_radius
    weights = softmax(-distances / temperature)
    distance = np.dot(distances, weights) / np.sum(weights)
    return distance


def raymarch(
        atom_locs: Array,
        init_loc: Array,
        dir: Array,
        steps_threshold: int = 25,
        proximity_threshold: float = 1e-3,
        beyond_threshold: float = 1e5,
):
    pure_signed_distance = partial(signed_distance, atom_locs)
    def stop(state):
        idx, loc = state
        true_func = lambda: True
        false_func = lambda: False

        # March until close to a surface, exhausted steps, or ended up far off.
        is_exhausted = cond(np.greater(idx, steps_threshold), true_func, false_func)
        is_close = cond(np.isclose(
            pure_signed_distance(loc),
            0.,
            atol=proximity_threshold
        ), true_func, false_func)
        is_beyond = cond(np.greater(norm(loc), beyond_threshold), true_func, false_func)

        return cond(is_exhausted, false_func,
                    lambda: cond(is_close, false_func,
                                 lambda: cond(is_beyond, false_func, true_func)))

    def step(state):
        idx, loc = state
        return idx + 1, loc + pure_signed_distance(loc) * dir

    dir = normalize(dir)
    steps, final_loc = while_loop(stop, step, (0, init_loc))
    return steps, final_loc


def spawn_rays(
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


def shoot_rays(
        view_size: Tuple[int, int],
        camera_loc: Array,
        atom_locs: Array
):
    w, h = view_size
    ray_dirs = spawn_rays(-camera_loc, view_size)
    pure_signed_distance = partial(signed_distance, atom_locs)
    pure_raymarch = partial(raymarch, atom_locs, camera_loc)
    steps, hits = vmap(pure_raymarch)(ray_dirs)
    return steps, hits


def compute_ambient_lighting(
        hits: Array,
        atom_locs: Array
):
    pure_signed_distance = partial(signed_distance, atom_locs)
    raw_normals = vmap(grad(pure_signed_distance))(hits)
    return raw_normals


def raymarch_lights(
        srcs: Array,
        dir: Array,
        atom_locs: Array,
        steps_threshold: int = 20,
        proximity_threshold: float = 1e-6,
        beyond_threshold: float = 1e7,
        hardness: float = 1.0
):
    pure_signed_distance = partial(signed_distance, atom_locs)
    dir = normalize(dir)

    def raymarch_light(
        src: Array
    ):
        def stop(state):
            idx, traveled, shadow = state
            true_func = lambda: True
            false_func = lambda: False

            # March until close to a surface, or exhausted steps.
            is_exhausted = cond(np.greater(idx, steps_threshold), true_func, false_func)
            is_close = cond(np.isclose(
                pure_signed_distance(src + dir * traveled),
                0.,
                atol=proximity_threshold
            ), true_func, false_func)
            is_beyond = cond(np.greater(traveled, beyond_threshold), true_func, false_func)

            return cond(is_exhausted, false_func,
                        lambda: cond(is_close, false_func,
                                     lambda: cond(is_beyond, false_func, true_func)))

        def step(state):
            idx, traveled, shadow = state
            d = pure_signed_distance(src + dir * traveled)
            return idx + 1, traveled + d, np.clip(hardness * d / traveled, 0, shadow)

        steps, traveled, shadow = while_loop(stop, step, (0, 1., 1.0))
        return steps, traveled, shadow

    shadows = vmap(partial(raymarch_light))(srcs)
    return shadows


def compute_shades(
    colors: Array,
    shadows: Array,
    raw_normals: Array,
    light_dir: Array,
    ray_dirs: Array,
):
    def compute_shade(
        color: Array,
        shadow: Array,
        raw_normal: Array,
        ray_dir: Array,
    ):
        ambient = norm(raw_normal)
        unit_normal = raw_normal / ambient
        diffuse = unit_normal.dot(light_dir).clip(0.0) * shadow
        half = normalize(light_dir - ray_dir)
        spec = 0.3 * shadow * half.dot(unit_normal).clip(0.0) ** 200.0
        light = 0.7 * diffuse + 0.1 * ambient
        return color * light + spec

    shades = vmap(compute_shade)(colors, shadows, raw_normals, ray_dirs)
    shades = shades ** (1.0 / 2.2)
    return shades


def compute_colors(
    atom_locs: Array,
    atom_colors: Array,
    locs: Array,
    atom_radius: float = 0.2,
    temperature: float = 1e-1,
):
    def compute_color(
        loc: Array
    ):
        distances = norm(loc - atom_locs) - atom_radius
        weights = softmax(-distances / temperature)
        color = atom_colors.T @ weights
        return color

    return vmap(compute_color)(locs)


def render_frames(
    locs_history: Array,
    atom_colors_by_timestep: Array,
    view_size: Tuple[int, int],
    camera_loc: Array,
    light_loc: Array
):
    w, h = view_size

    def render_frame(
            atom_locs: Array,
            atom_colors: Array
    ):
        steps, hits = shoot_rays(view_size, camera_loc, atom_locs)
        raw_normals = compute_ambient_lighting(hits, atom_locs)
        steps, traveled, shadows = raymarch_lights(hits, light_loc, atom_locs)
        ray_dirs = spawn_rays(-camera_loc, view_size)
        colors = compute_colors(atom_locs, atom_colors, hits)
        shades = compute_shades(colors, shadows, raw_normals, light_loc, ray_dirs)

        shades = shades.reshape(h, w, 3)
        shades = shades * 255
        shades = shades.astype("uint8")
        shades = np.array(shades)

        return shades

    return vmap(render_frame)(locs_history, atom_colors_by_timestep)
