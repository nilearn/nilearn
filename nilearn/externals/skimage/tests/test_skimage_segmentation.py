"""
Test functions from skimage testing for random_walker.py
"""

import numpy as np

from nilearn.externals.skimage import random_walker


def make_2d_syntheticdata(lx, ly=None):
    if ly is None:
        ly = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly)) + 0.1 * np.random.randn(lx, ly)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l,
         ly // 2 - small_l:ly // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1,
         ly // 2 - small_l + 1:ly // 2 + small_l - 1] = (
             0.1 * np.random.randn(2 * small_l - 2, 2 * small_l - 2))
    data[lx // 2 - small_l, ly // 2 - small_l // 8:ly // 2 + small_l // 8] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5] = 1
    seeds[lx // 2 + small_l // 4, ly // 2 - small_l // 4] = 2
    return data, seeds


def make_3d_syntheticdata(lx, ly=None, lz=None):
    if ly is None:
        ly = lx
    if lz is None:
        lz = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly, lz)) + 0.1 * np.random.randn(lx, ly, lz)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l,
         ly // 2 - small_l:ly // 2 + small_l,
         lz // 2 - small_l:lz // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1,
         ly // 2 - small_l + 1:ly // 2 + small_l - 1,
         lz // 2 - small_l + 1:lz // 2 + small_l - 1] = 0
    # make a hole
    hole_size = np.max([1, small_l // 8])
    data[lx // 2 - small_l,
         ly // 2 - hole_size:ly // 2 + hole_size,
         lz // 2 - hole_size:lz // 2 + hole_size] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5, lz // 5] = 1
    seeds[lx // 2 + small_l // 4,
          ly // 2 - small_l // 4,
          lz // 2 - small_l // 4] = 2
    return data, seeds


def test_2d_bf():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels_bf = random_walker(data, labels, beta=90, mode='bf')
    assert (labels_bf[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    full_prob_bf = random_walker(data, labels, beta=90, mode='bf',
                                 return_full_prob=True)
    assert (full_prob_bf[1, 25:45, 40:60] >=
            full_prob_bf[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    # Now test with more than two labels
    labels[55, 80] = 3
    full_prob_bf = random_walker(data, labels, beta=90, mode='bf',
                                 return_full_prob=True)
    assert (full_prob_bf[1, 25:45, 40:60] >=
            full_prob_bf[0, 25:45, 40:60]).all()
    assert len(full_prob_bf) == 3
    assert data.shape == labels.shape


def test_2d_cg():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels_cg = random_walker(data, labels, beta=90, mode='cg')
    assert (labels_cg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    full_prob = random_walker(data, labels, beta=90, mode='cg',
                              return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >=
            full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    return data, labels_cg


def test_2d_cg_mg():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels_cg_mg = random_walker(data, labels, beta=90, mode='cg_mg')
    assert (labels_cg_mg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    full_prob = random_walker(data, labels, beta=90, mode='cg_mg',
                              return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >=
            full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    return data, labels_cg_mg


def test_reorder_labels():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels[labels == 2] = 4
    labels_bf = random_walker(data, labels, beta=90, mode='bf')
    assert (labels_bf[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels_bf


def test_spacing_0():
    n = 30
    lx, ly, lz = n, n, n
    data, _ = make_3d_syntheticdata(lx, ly, lz)

    # Rescale `data` along Z axis
    data_aniso = np.zeros((n, n, n // 2))
    for i, yz in enumerate(data):
        data_aniso[i, :, :] = np.resize(yz, (n, n // 2))

    # Generate new labels
    small_l = int(lx // 5)
    labels_aniso = np.zeros_like(data_aniso)
    labels_aniso[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso[lx // 2 + small_l // 4,
                 ly // 2 - small_l // 4,
                 lz // 4 - small_l // 8] = 2

    # Test with `spacing` kwarg
    labels_aniso = random_walker(data_aniso, labels_aniso, mode='cg',
                                 spacing=(1., 1., 0.5))
    assert (labels_aniso[13:17, 13:17, 7:9] == 2).all()


def test_trivial_cases():
    # When all voxels are labeled
    img = np.ones((10, 10))
    labels = np.ones((10, 10))

    pass_through = random_walker(img, labels)
    np.testing.assert_array_equal(pass_through, labels)

    # When all voxels are labeled AND return_full_prob is True
    labels[:, :5] = 3
    expected = np.concatenate(((labels == 1)[..., np.newaxis],
                               (labels == 3)[..., np.newaxis]), axis=2)
    test = random_walker(img, labels, return_full_prob=True)
    np.testing.assert_array_equal(test, expected)


def test_bad_inputs():
    # Too few dimensions
    img = np.ones(10)
    labels = np.arange(10)
    np.testing.assert_raises(ValueError, random_walker, img, labels)
    np.testing.assert_raises(ValueError,
                             random_walker, img, labels, multichannel=True)

    # Too many dimensions
    np.random.seed(42)
    img = np.random.normal(size=(3, 3, 3, 3, 3))
    labels = np.arange(3 ** 5).reshape(img.shape)
    np.testing.assert_raises(ValueError, random_walker, img, labels)
    np.testing.assert_raises(ValueError,
                             random_walker, img, labels, multichannel=True)

    # Spacing incorrect length
    img = np.random.normal(size=(10, 10))
    labels = np.zeros((10, 10))
    labels[2, 4] = 2
    labels[6, 8] = 5
    np.testing.assert_raises(ValueError,
                             random_walker, img, labels, spacing=(1,))
