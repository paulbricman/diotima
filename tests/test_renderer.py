import io
import jax.numpy as np
import numpy
from diotima.renderer import *
from jax import Array
import pytest
from PIL import Image


@pytest.fixture
def camera_loc():
    return np.array([5., 0., 0.])


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


def test_signed_distance(camera_loc: Array, atom_locs_render: Array):
    distance = signed_distance(atom_locs_render, camera_loc)
    assert distance.size == 1


def test_raymarch(camera_loc: Array, atom_locs_hit: Array, atom_locs_miss: Array):
    pure_signed_distance = partial(signed_distance, atom_locs_hit)
    steps, final_loc = raymarch(pure_signed_distance, camera_loc, -camera_loc)
    assert steps < 5

    pure_signed_distance = partial(signed_distance, atom_locs_miss)
    steps, final_loc = raymarch(pure_signed_distance, camera_loc, -camera_loc)
    assert steps > 10


def test_radiate(camera_loc: Array):
    view_size = 640, 400
    w, h = view_size
    grid = radiate(-camera_loc, view_size)
    assert grid.shape == (w * h, 3)


def test_render_proper(camera_loc: Array, atom_locs_render: Array):
    view_size = 640, 400
    w, h = view_size

    steps, hits = render(view_size, camera_loc, atom_locs_render)

    assert steps.shape == (h, w)
    assert hits.shape == (h, w, 3)
    assert steps[0][0] > 10

    hits = hits % 1. * 255
    hits = hits.astype("uint8")
    hits = numpy.array(hits)
    img = Image.fromarray(hits)
    img.save("dummy.jpg")
