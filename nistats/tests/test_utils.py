#!/usr/bin/env python

import numpy as np
from scipy.stats import norm

from ..utils import multiple_mahalanobis, z_score, multiple_fast_inv
from nose.tools import assert_true
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from ..utils import (matrix_rank, full_rank, pos_recipr, open4csv)
from nose.tools import (assert_true, assert_equal, assert_false,
                        assert_raises)
from nibabel.tmpdirs import InTemporaryDirectory


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

def test_full_rank():
    rng = np.random.RandomState(20110831)
    X = rng.standard_normal((40,5))
    # A quick rank check
    assert_equal(matrix_rank(X), 5)
    X[:,0] = X[:,1] + X[:,2]
    assert_equal(matrix_rank(X), 4)
    Y1 = full_rank(X)
    assert_equal(Y1.shape, (40,4))
    Y2 = full_rank(X, r=3)
    assert_equal(Y2.shape, (40,3))
    Y3 = full_rank(X, r=4)
    assert_equal(Y3.shape, (40,4))
    # Windows - there seems to be some randomness in the SVD result; standardize
    # column signs before comparison
    flipper = np.sign(Y1[0]) * np.sign(Y3[0])
    assert_almost_equal(Y1, Y3 * flipper)


def test_pos_recipr():
    X = np.array([2,1,-1,0], dtype=np.int8)
    eX = np.array([0.5,1,0,0])
    Y = pos_recipr(X)
    yield assert_array_almost_equal, Y, eX
    yield assert_equal, Y.dtype.type, np.float64
    X2 = X.reshape((2,2))
    Y2 = pos_recipr(X2)
    yield assert_array_almost_equal, Y2, eX.reshape((2,2))
    # check that lists have arrived
    XL = [0, 1, -1]
    yield assert_array_almost_equal, pos_recipr(XL), [0, 1, 0]
    # scalars
    yield assert_equal, pos_recipr(-1), 0
    yield assert_equal, pos_recipr(0), 0
    yield assert_equal, pos_recipr(2), 0.5

def test_open4csv():
    # Test opening of csv files
    import csv
    contents = [['oh', 'my', 'G'],
                ['L', 'O', 'L'],
                ['when', 'cleaning', 'windas']]
    with InTemporaryDirectory():
        with open4csv('my.csv', 'w') as fobj:
            writer = csv.writer(fobj)
            writer.writerows(contents)
        with open4csv('my.csv', 'r') as fobj:
            dialect = csv.Sniffer().sniff(fobj.read())
            fobj.seek(0)
            reader = csv.reader(fobj, dialect)
            back = list(reader)
    assert_equal(contents, back)
    assert_raises(ValueError, open4csv, 'my.csv', 'rb')
    assert_raises(ValueError, open4csv, 'my.csv', 'wt')


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
