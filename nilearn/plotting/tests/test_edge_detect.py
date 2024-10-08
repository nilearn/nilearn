import numpy as np

from nilearn.plotting.edge_detect import _edge_detect


def test_edge_detect():
    img = np.zeros((10, 10))
    img[:5] = 1
    _, _ = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)


def test_edge_nan():
    img = np.zeros((10, 10))
    img[:5] = 1
    img[0] = np.nan
    grad_mag, _ = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)
    assert (grad_mag[0] > 2).all()
