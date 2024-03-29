from diotima.world.renderer import *
from diotima.world.universe import *

from jax import Array
import jax.numpy as jnp
import numpy as np

import cv2
import io
import pytest
from PIL import Image
from einops import repeat


@pytest.fixture
def camera_loc():
    return jnp.array([10.0, 0.0, 0.0])


@pytest.fixture
def light_loc():
    return jnp.array([1.0, 0.0, -1.2])


@pytest.fixture
def atom_locs_hit():
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.3],
            [-0.4, -0.8, -0.4],
            [0.2, 0.5, 0.6],
        ]
    )


@pytest.fixture
def atom_locs_miss():
    return jnp.array(
        [
            [0.0, 2.0, 0.0],
        ]
    )


@pytest.fixture
def atom_locs_render():
    return jnp.array(
        [
            [0.4, -0.2, -0.4],
            [0.0, 0.2, 0.3],
            [-0.4, -0.8, -0.6],
            [0.2, 0.5, 0.6],
        ]
    )


@pytest.fixture
def universe_config():
    n_elems = 2
    return UniverseConfig(
        n_elems,
        n_atoms=2,
        n_dims=3,
        dt=0.1,
        physics_config=physics.default_physics_config(n_elems),
        elem_distrib=physics.default_elem_distrib(n_elems),
    )


@pytest.fixture
def universe(universe_config: UniverseConfig):
    return seed(universe_config)


def test_signed_distance(camera_loc: Array, atom_locs_render: Array):
    distance = signed_distance(atom_locs_render, camera_loc)
    assert distance.size == 1


def test_raymarch(camera_loc: Array, atom_locs_hit: Array, atom_locs_miss: Array):
    steps, final_loc = raymarch(atom_locs_hit, camera_loc, -camera_loc)
    assert steps < 5

    steps, final_loc = raymarch(atom_locs_miss, camera_loc, -camera_loc)
    assert steps > 10


def test_spawn_rays(camera_loc: Array):
    view_size = 640, 400
    w, h = view_size
    grid = spawn_rays(-camera_loc, view_size)
    assert grid.shape == (w * h, 3)


def test_shoot_rays(camera_loc: Array, atom_locs_render: Array):
    view_size = 640, 400
    w, h = view_size

    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    hits = hits.reshape(h, w, 3)
    steps = steps.reshape(h, w)

    assert steps.shape == (h, w)
    assert hits.shape == (h, w, 3)
    assert steps[0][0] > 10

    hits = hits % 1.0 * 255
    hits = hits.astype("uint8")
    hits = np.array(hits)
    img = Image.fromarray(hits)
    img.save("media/hits.jpg")


def test_ambient_lighting(camera_loc: Array, atom_locs_render: Array):
    view_size = 640, 400
    w, h = view_size
    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    hits = compute_ambient_lighting(hits, atom_locs_render)
    hits = norm(hits)
    hits = hits.reshape(h, w)
    steps = steps.reshape(h, w)

    assert steps.shape == (h, w)
    assert hits.shape == (h, w)

    hits = hits * 255
    hits = hits.astype("uint8")
    hits = np.array(hits)
    img = Image.fromarray(hits)
    img.save("media/ambient_light.jpg")


def test_direct_lighting(camera_loc: Array, atom_locs_render: Array, light_loc: Array):
    view_size = 640, 400
    w, h = view_size
    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    steps, traveled, shadows = raymarch_lights(hits, light_loc, atom_locs_render)
    steps = steps.reshape(h, w)
    traveled = traveled.reshape(h, w)
    shadows = shadows.reshape(h, w)

    assert steps.shape == (h, w)
    assert traveled.shape == (h, w)
    assert shadows.shape == (h, w)

    shadows = shadows * 255
    shadows = shadows.astype("uint8")
    shadows = np.array(shadows)
    img = Image.fromarray(shadows)
    img.save("media/directional_light.jpg")


def test_compute_shades(camera_loc: Array, atom_locs_render: Array, light_loc: Array):
    view_size = 640, 400
    w, h = view_size
    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    raw_normals = compute_ambient_lighting(hits, atom_locs_render)
    steps, traveled, shadows = raymarch_lights(hits, light_loc, atom_locs_render)
    ray_dirs = spawn_rays(-camera_loc, view_size)
    shades = compute_shades(
        np.ones((w * h, 3)), shadows, raw_normals, light_loc, ray_dirs
    )
    shades = shades.reshape(h, w, 3)

    assert shades.shape == (h, w, 3)

    shades = shades * 255
    shades = shades.astype("uint8")
    shades = np.array(shades)
    img = Image.fromarray(shades)
    img.save("media/shades.jpg")


def test_compute_colors(camera_loc: Array, atom_locs_render: Array, light_loc: Array):
    view_size = 640, 400
    w, h = view_size
    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    raw_normals = compute_ambient_lighting(hits, atom_locs_render)
    steps, traveled, shadows = raymarch_lights(hits, light_loc, atom_locs_render)
    ray_dirs = spawn_rays(-camera_loc, view_size)
    atom_colors = jax.random.uniform(
        jax.random.PRNGKey(42), (atom_locs_render.shape[0], 3)
    )
    colors = compute_colors(atom_locs_render, atom_colors, hits)
    shades = compute_shades(colors, shadows, raw_normals, light_loc, ray_dirs)
    shades = shades.reshape(h, w, 3)

    assert shades.shape == (h, w, 3)

    shades = shades * 255
    shades = shades.astype("uint8")
    shades = np.array(shades)
    img = Image.fromarray(shades)
    img.save("media/colors.jpg")


def test_render_frames(
    universe: Universe,
    univers_config: UniverseConfig,
    camera_loc: Array,
    light_loc: Array,
):
    n_frame_chunks = 1
    n_frames_in_chunk = 1
    view_size = 256, 144
    w, h = view_size
    atom_colors = jax.random.uniform(
        jax.random.PRNGKey(0), (universe_config.n_atoms, 3)
    )
    atom_colors = repeat(atom_colors, "a c -> f a c", f=n_frames_in_chunk)
    out = cv2.VideoWriter(
        "media/frames.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
    )

    for _ in range(n_frame_chunks):
        universe = run(universe, n_frames_in_chunk)

        frames = render_frames(
            universe.locs_history[-n_frames_in_chunk:],
            atom_colors,
            view_size,
            camera_loc,
            light_loc,
        )

        assert frames.shape == (n_frames_in_chunk, h, w, 3)

        for frame in np.array(frames):
            out.write(frame)
    out.release()
