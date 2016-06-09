#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.linalg as spl
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from nose.tools import assert_true, assert_equal, assert_raises

from nistats.utils import (multiple_mahalanobis, z_score, multiple_fast_inv,
                           pos_recipr, full_rank, _check_run_tables,
                           _check_and_load_tables, _check_list_length_match)


def test_full_rank():
    n, p = 10, 5
    X = np.random.randn(n, p)
    X_, _ = full_rank(X)
    assert_array_almost_equal(X, X_)
    X[:, -1] = X[:, :-1].sum(1)
    X_, cond = full_rank(X)
    assert_true(cond > 1.e10)
    assert_array_almost_equal(X, X_)


def test_z_score():
    p = np.random.rand(10)
    assert_array_almost_equal(norm.sf(z_score(p)), p)
    # check the numerical precision
    for p in [1.e-250, 1 - 1.e-16]:
        assert_array_almost_equal(z_score(p), norm.isf(p))
    assert_array_almost_equal(z_score(np.float32(1.e-100)), norm.isf(1.e-300))


def test_mahalanobis():
    n = 50
    x = np.random.rand(n) / n
    A = np.random.rand(n, n) / n
    A = np.dot(A.transpose(), A) + np.eye(n)
    mah = np.dot(x, np.dot(spl.inv(A), x))
    assert_almost_equal(mah, multiple_mahalanobis(x, A), decimal=1)


def test_mahalanobis2():
    n = 50
    x = np.random.randn(n, 3)
    Aa = np.zeros([n, n, 3])
    for i in range(3):
        A = np.random.randn(120, n)
        A = np.dot(A.T, A)
        Aa[:, :, i] = A
    i = np.random.randint(3)
    mah = np.dot(x[:, i], np.dot(spl.inv(Aa[:, :, i]), x[:, i]))
    f_mah = (multiple_mahalanobis(x, Aa))[i]
    assert_true(np.allclose(mah, f_mah))


def test_multiple_fast_inv():
    shape = (10, 20, 20)
    X = np.random.randn(shape[0], shape[1], shape[2])
    X_inv_ref = np.zeros(shape)
    for i in range(shape[0]):
        X[i] = np.dot(X[i], X[i].T)
        X_inv_ref[i] = spl.inv(X[i])
    X_inv = multiple_fast_inv(X)
    assert_almost_equal(X_inv_ref, X_inv)


def test_pos_recipr():
    X = np.array([2, 1, -1, 0], dtype=np.int8)
    eX = np.array([0.5, 1, 0, 0])
    Y = pos_recipr(X)
    yield assert_array_almost_equal, Y, eX
    yield assert_equal, Y.dtype.type, np.float64
    X2 = X.reshape((2, 2))
    Y2 = pos_recipr(X2)
    yield assert_array_almost_equal, Y2, eX.reshape((2, 2))
    # check that lists have arrived
    XL = [0, 1, -1]
    yield assert_array_almost_equal, pos_recipr(XL), [0, 1, 0]
    # scalars
    yield assert_equal, pos_recipr(-1), 0
    yield assert_equal, pos_recipr(0), 0
    yield assert_equal, pos_recipr(2), 0.5


def test_img_table_checks():
    # check matching lengths
    assert_raises(ValueError, _check_list_length_match, [''] * 2, [''], "", "")
    # check tables type and that can be loaded
    assert_raises(ValueError, _check_and_load_tables, ['.csv', '.csv'], "")
    assert_raises(TypeError, _check_and_load_tables,
                  [np.array([0]), pd.DataFrame()], "")
    assert_raises(ValueError, _check_and_load_tables,
                  ['.csv', pd.DataFrame()], "")
    # check high level wrapper keeps behavior
    assert_raises(ValueError, _check_run_tables, [''] * 2, [''], "")
    assert_raises(ValueError, _check_run_tables, [''] * 2, ['.csv', '.csv'], "")
    assert_raises(TypeError, _check_run_tables, [''] * 2,
                  [np.array([0]), pd.DataFrame()], "")
    assert_raises(ValueError, _check_run_tables, [''] * 2,
                  ['.csv', pd.DataFrame()], "")
