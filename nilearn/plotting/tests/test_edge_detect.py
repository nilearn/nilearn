import numpy as np

from nilearn.plotting.edge_detect import _edge_detect
from nose.tools import assert_true

def test_edge_detect():
    img = np.zeros((10, 10))
    img[:5] = 1
    _, edge_mask = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)


def test_edge_nan():
    img = np.zeros((10, 10))
    img[:5] = 1
    img[0] = np.NaN
    grad_mag, edge_mask = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)
    assert_true((grad_mag[0] > 2).all())
