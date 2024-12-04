"""
Testing functions for random walker segmentation from scikit-image 0.11.3.

Thanks to scikit image.
"""

import numpy as np
import pytest

from nilearn._utils.segmentation import random_walker


def test_modes_in_random_walker_spacing(rng):
    """Smoke test for spacing in random_walker."""
    img = rng.standard_normal(size=(30, 30, 30))
    labels = np.zeros_like(img)
    random_walker(img, labels, beta=90, spacing=(3, 3, 3))


def test_modes_in_random_walker(rng):
    img = np.zeros((30, 30, 30)) + 0.1 * rng.standard_normal(size=(30, 30, 30))
    img[9:21, 9:21, 9:21] = 1
    img[10:20, 10:20, 10:20] = 0
    labels = np.zeros_like(img)
    labels[6, 6, 6] = 1
    labels[14, 15, 16] = 2
    # default mode = cg
    random_walker_cg = random_walker(img, labels, beta=90)
    assert (random_walker_cg.reshape(img.shape)[6, 6, 6] == 1).all()
    assert img.shape == random_walker_cg.shape
    # test `mask` strategy of sub function _mask_edges_weights in laplacian
    labels[5:25, 26:29, 26:29] = -1
    random_walker(img, labels, beta=30)


def test_isolated_pixel(rng):
    data = rng.random((3, 3))

    # Build the following labels with an isolated seed
    # in the bottom-right corner:
    # array([[ 0., -1., -1.],
    #        [-1., -1., -1.],
    #        [-1., -1.,  1.]])
    labels = -np.ones((3, 3))
    # Point
    labels[0, 0] = 0
    # Make a seed
    labels[2, 2] = 1
    # The expected result is:
    # array([[ 0., -1., -1.],
    #        [-1., -1., -1.],
    #        [-1., -1.,  1.]])
    expected = np.array(
        [[0.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]
    )
    np.testing.assert_array_equal(expected, random_walker(data, labels))


def test_isolated_seed(rng):
    data = rng.random((3, 3))

    # Build the following labels with an isolated seed
    # in the bottom-right corner:
    # array([[ 0.,  1., -1.],
    #        [-1., -1., -1.],
    #        [-1., -1.,  2.]])
    labels = -np.ones((3, 3))
    # Make a seed
    labels[0, 1] = 1
    # Point next to the seed
    labels[0, 0] = 0
    # Set a seed in the middle, it is surrounded by masked pixels
    labels[2, 2] = 2
    # The expected result is:
    # array([[ 1.,  1., -1.],
    #        [-1., -1., -1.],
    #        [-1., -1., -1.]])
    expected = np.array(
        [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    )
    np.testing.assert_array_equal(expected, random_walker(data, labels))


def test_trivial_cases():
    # When all voxels are labeled
    img = np.ones((10, 10, 10))
    labels = np.ones((10, 10, 10))

    # It returns same labels which are provided
    pass_through = random_walker(img, labels)
    np.testing.assert_array_equal(pass_through, labels)

    # When there is no seed
    # Case 1: only masked pixels
    labels = -np.ones((10, 10, 10))
    # It returns the labels
    np.testing.assert_array_equal(random_walker(img, labels), labels)

    # Case 2: only unlabeled pixels
    labels = np.zeros((10, 10, 10))
    # It return the labels
    np.testing.assert_array_equal(random_walker(img, labels), labels)


def test_bad_inputs(rng):
    # Too few dimensions
    img = np.ones(10)
    labels = np.arange(10)
    with pytest.raises(ValueError):
        random_walker(img, labels)

    # Too many dimensions
    img = rng.normal(size=(3, 3, 3, 3, 3))
    labels = np.arange(3**5).reshape(img.shape)
    with pytest.raises(ValueError):
        random_walker(img, labels)

    # Spacing incorrect length
    img = rng.normal(size=(10, 10))
    labels = np.zeros((10, 10))
    labels[2, 4] = 2
    labels[6, 8] = 5
    with pytest.raises(ValueError):
        random_walker(img, labels, spacing=(1,))


def test_reorder_labels(rng):
    """When labels have non-consecutive integers, make them consecutive by \
    reordering them to make no gaps/differences between integers.

    We expect labels to be of same shape even if they are reordered.

    Issue #938, comment #14.
    """
    data = np.zeros((5, 5)) + 0.1 * rng.standard_normal(size=(5, 5))
    data[1:5, 1:5] = 1

    labels = np.zeros_like(data)
    labels[3, 3] = 1
    labels[1, 4] = 4  # giving integer which is non-consecutive

    labels = random_walker(data, labels)
    assert data.shape == labels.shape
