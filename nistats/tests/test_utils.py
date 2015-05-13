#!/usr/bin/env python

import numpy as np
from scipy.stats import norm

from ..utils import multiple_mahalanobis, z_score, multiple_fast_inv
from nose.tools import assert_true
from numpy.testing import assert_almost_equal, assert_array_almost_equal

def test_z_score():
    p = np.random.rand(10)
    z = z_score(p)
    assert_array_almost_equal(norm.sf(z), p) 

def test_mahalanobis():
    x = np.random.rand(100) / 100
    A = np.random.rand(100, 100) / 100
    A = np.dot(A.transpose(), A) + np.eye(100)
    mah = np.dot(x, np.dot(np.linalg.inv(A), x))
    assert_almost_equal(mah, multiple_mahalanobis(x, A), decimal=1) 

def test_mahalanobis2():
    x = np.random.randn(100, 3)
    Aa = np.zeros([100, 100, 3])
    for i in range(3):
        A = np.random.randn(120, 100)
        A = np.dot(A.T, A)
        Aa[:, :, i] = A
    i = np.random.randint(3)
    mah = np.dot(x[:, i], np.dot(np.linalg.inv(Aa[:, :, i]), x[:, i]))
    f_mah = (multiple_mahalanobis(x, Aa))[i]
    assert_true(np.allclose(mah, f_mah))

def test_multiple_fast_inv():
    shape = (10, 20, 20)
    X = np.random.randn(shape[0], shape[1], shape[2])
    X_inv_ref = np.zeros(shape)
    for i in range(shape[0]):
        X[i] = np.dot(X[i], X[i].T)
        X_inv_ref[i] = np.linalg.inv(X[i])
    X_inv = multiple_fast_inv(X)
    assert_almost_equal(X_inv_ref, X_inv)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
