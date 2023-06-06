# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for io_processors."""

import numpy as np

from diotima.perceiver.perceiver import io_processors


def _create_test_image(shape):
  image = np.arange(np.prod(np.array(shape)))
  return np.reshape(image, shape)


def test_space_to_depth_image():
  image_shape = (2, 3 * 5, 3 * 7, 11)
  image = _create_test_image(image_shape)
  output = io_processors.space_to_depth(image, spatial_block_size=3)
  assert output.shape == (2, 5, 7, 3 * 3 * 11)


def test_space_to_depth_video():
  image_shape = (2, 5 * 7, 3 * 11, 3 * 13, 17)
  image = _create_test_image(image_shape)
  output = io_processors.space_to_depth(image, spatial_block_size=3,
                                        temporal_block_size=5)
  assert output.shape == (2, 7, 11, 13, 5 * 3 * 3 * 17)


def test_reverse_space_to_depth_image():
  image_shape = (2, 5, 7, 3 * 3 * 11)
  image = _create_test_image(image_shape)
  output = io_processors.reverse_space_to_depth(image, spatial_block_size=3)
  assert output.shape == (2, 3 * 5, 3 * 7, 11)


def test_reverse_space_to_depth_video():
  image_shape = (2, 7, 11, 13, 5 * 3 * 3 * 17)
  image = _create_test_image(image_shape)
  output = io_processors.reverse_space_to_depth(
      image, spatial_block_size=3, temporal_block_size=5)
  assert output.shape == (2, 5 * 7, 3 * 11, 3 * 13, 17)
