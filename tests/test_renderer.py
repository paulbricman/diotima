import io
import jax.numpy as np
import numpy
from diotima.renderer import *
from jax import Array, random
import pytest
from PIL import Image
from diotima.utils import norm
from diotima.world import Universe, UniverseConfig, seed, run
from einops import repeat
import cv2


@pytest.fixture
def camera_loc():
    return np.array([5., 0., 0.])


@pytest.fixture
def light_loc():
    return np.array([1., 0., -1.2])


@pytest.fixture
def atom_locs_hit():
    return np.array([
        [0., 0., 0.],
        [0., 1., 0.3],
        [-0.4, -0.8, -0.4],
        [0.2, 0.5, 0.6],
    ])


@pytest.fixture
def atom_locs_miss():
    return np.array([
        [0., 2., 0.],
    ])


@pytest.fixture
def atom_locs_render():
    return np.array([
        [0.4, -0.2, -0.4],
        [0., 0.2, 0.3],
        [-0.4, -0.8, -0.6],
        [0.2, 0.5, 0.6],
    ])


@pytest.fixture
def universe():
    return seed(UniverseConfig(
        n_dims = 3,
        n_atoms = 8,
    ))


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

    hits = hits % 1. * 255
    hits = hits.astype("uint8")
    hits = numpy.array(hits)
    img = Image.fromarray(hits)
    img.save("hits.jpg")


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
    hits = numpy.array(hits)
    img = Image.fromarray(hits)
    img.save("ambient_light.jpg")


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
    shadows = numpy.array(shadows)
    img = Image.fromarray(shadows)
    img.save("directional_light.jpg")


def test_compute_shades(camera_loc: Array, atom_locs_render: Array, light_loc: Array):
    view_size = 640, 400
    w, h = view_size
    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    raw_normals = compute_ambient_lighting(hits, atom_locs_render)
    steps, traveled, shadows = raymarch_lights(hits, light_loc, atom_locs_render)
    ray_dirs = spawn_rays(-camera_loc, view_size)
    shades = compute_shades(np.ones((w * h, 3)), shadows, raw_normals, light_loc, ray_dirs)
    shades = shades.reshape(h, w, 3)

    assert shades.shape == (h, w, 3)

    shades = shades * 255
    shades = shades.astype("uint8")
    shades = numpy.array(shades)
    img = Image.fromarray(shades)
    img.save("shades.jpg")


def test_compute_colors(camera_loc: Array, atom_locs_render: Array, light_loc: Array):
    view_size = 640, 400
    w, h = view_size
    steps, hits = shoot_rays(view_size, camera_loc, atom_locs_render)
    raw_normals = compute_ambient_lighting(hits, atom_locs_render)
    steps, traveled, shadows = raymarch_lights(hits, light_loc, atom_locs_render)
    ray_dirs = spawn_rays(-camera_loc, view_size)
    atom_colors = random.uniform(random.PRNGKey(42), (
        atom_locs_render.shape[0],
        3
    ))
    colors = compute_colors(atom_locs_render, atom_colors, hits)
    shades = compute_shades(colors, shadows, raw_normals, light_loc, ray_dirs)
    shades = shades.reshape(h, w, 3)

    assert shades.shape == (h, w, 3)

    shades = shades * 255
    shades = shades.astype("uint8")
    shades = numpy.array(shades)
    img = Image.fromarray(shades)
    img.save("colors.jpg")


def test_render_frames(universe: Universe, camera_loc: Array, light_loc: Array):
    n_frames = 20
    view_size = 640, 400
    w, h = view_size

    universe = run(universe, n_frames)
    atom_colors = random.uniform(random.PRNGKey(42), (
        universe.universe_config.n_atoms,
        3
    ))

    atom_colors = repeat(atom_colors, "a c -> f a c", f = n_frames)
    frames = render_frames(
        universe.locs_history,
        atom_colors,
        view_size,
        camera_loc,
        light_loc
    )

    assert frames.shape == (n_frames, h, w, 3)

    out = cv2.VideoWriter('frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    for frame in numpy.array(frames):
        out.write(frame)
    out.release()


