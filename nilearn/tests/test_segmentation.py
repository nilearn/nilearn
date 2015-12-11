"""
Testing functions for random walker segmentation from scikit-image 0.11.3.

Thanks to scikit image.
"""

import numpy as np

from nilearn._utils.segmentation import _random_walker


def test_modes_in_random_walker():
    img = np.zeros((30, 30, 30)) + 0.1 * np.random.randn(30, 30, 30)
    img[9:21, 9:21, 9:21] = 1
    img[10:20, 10:20, 10:20] = 0
    labels = np.zeros_like(img)
    labels[6, 6, 6] = 1
    labels[14, 15, 16] = 2
    # default mode = cg
    random_walker_cg = _random_walker(img, labels, beta=90)
    assert (random_walker_cg.reshape(img.shape)[6, 6, 6] == 1).all()
    assert img.shape == random_walker_cg.shape
    # test `mask` strategy of sub function _mask_edges_weights in laplacian
    labels[5:25, 26:29, 26:29] = -1
    random_walker_inactive = _random_walker(img, labels, beta=30)


def test_trivial_cases():
    # When all voxels are labeled
    img = np.ones((10, 10, 10))
    labels = np.ones((10, 10, 10))

    # It returns same labels which are provided
    pass_through = _random_walker(img, labels)
    np.testing.assert_array_equal(pass_through, labels)


def test_bad_inputs():
    # Too few dimensions
    img = np.ones(10)
    labels = np.arange(10)
    np.testing.assert_raises(ValueError, _random_walker, img, labels)

    # Too many dimensions
    np.random.seed(42)
    img = np.random.normal(size=(3, 3, 3, 3, 3))
    labels = np.arange(3 ** 5).reshape(img.shape)
    np.testing.assert_raises(ValueError, _random_walker, img, labels)

    # Spacing incorrect length
    img = np.random.normal(size=(10, 10))
    labels = np.zeros((10, 10))
    labels[2, 4] = 2
    labels[6, 8] = 5
    np.testing.assert_raises(ValueError,
                             _random_walker, img, labels, spacing=(1,))
