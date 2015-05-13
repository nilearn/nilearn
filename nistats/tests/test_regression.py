# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.regression
"""

import numpy as np

import scipy.linalg as spl

from ..regression import (OLSModel, ARModel,
                          ar_bias_corrector, ar_bias_correct)

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal


RNG = np.random.RandomState(20110902)
X = RNG.standard_normal((40,10))
Y = RNG.standard_normal((40,))


def test_OLS():
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert_equal(results.df_resid, 30)


def test_AR():
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert_equal(results.df_resid, 30)


def test_OLS_degenerate():
    Xd = X.copy()
    Xd[:,0] = Xd[:,1] + Xd[:,2]
    model = OLSModel(design=Xd)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_AR_degenerate():
    Xd = X.copy()
    Xd[:,0] = Xd[:,1] + Xd[:,2]
    model = ARModel(design=Xd, rho=0.9)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_ar_estimator():
    # More or less a smoke test
    rng = np.random.RandomState(20110903)
    N = 100
    Y = rng.normal(size=(N,1)) * 10 + 100
    X = np.c_[np.linspace(-1,1,N), np.ones((N,))]
    my_model = OLSModel(X)
    results = my_model.fit(Y)
    rhos2 = ar_bias_correct(results, 2)
    invM = ar_bias_corrector(my_model.design, my_model.calc_beta, 2)
    rhos3 = ar_bias_correct(results, 2, invM)
    assert_array_almost_equal(rhos2, rhos3)
    # Check orders 1 and 3
    rhos = ar_bias_correct(results, 1)
    assert_equal(rhos.shape, ())
    assert_true(abs(rhos) <= 1)
    rhos = ar_bias_correct(results, 3)
    assert_equal(rhos.shape, (3,))
    assert_true(np.all(np.abs(rhos) <= 1))
    # Make a 2D Y and try that
    Y = rng.normal(size=(N, 12)) * 10 + 100
    results = my_model.fit(Y)
    rhos2 = ar_bias_correct(results, 2)
    rhos3 = ar_bias_correct(results, 2, invM)
    assert_array_almost_equal(rhos2, rhos3)
    # Passing in a simple array
    rhos4 = ar_bias_correct(results.resid, 2, invM)
    assert_array_almost_equal(rhos3, rhos4)
    # Check orders 1 and 3
    rhos = ar_bias_correct(results, 1)
    assert_equal(rhos.shape, (12,))
    assert_true(np.all(np.abs(rhos) <= 1))
    rhos = ar_bias_correct(results, 3)
    assert_equal(rhos.shape, (3, 12))
    assert_true(np.all(np.abs(rhos) <= 1))
    # Try reshaping to 3D
    results.resid = results.resid.reshape((N, 3, 4))
    rhos = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2.reshape((2, 3, 4)))
