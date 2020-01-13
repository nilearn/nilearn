"""
Test functions for models.regression
"""

import numpy as np

from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           )

from nistats.regression import OLSModel, ARModel


RNG = np.random.RandomState(20110902)
X = RNG.standard_normal((40, 10))
Y = RNG.standard_normal((40,))


def test_OLS():
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert results.df_resid == 30
    assert results.resid.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_AR():
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert results.df_resid == 30
    assert results.resid.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_residuals():
    Xintercept = X.copy()

    # If design matrix contains an intercept, the
    # mean of the residuals should be 0 (short of
    # some numerical rounding errors)
    Xintercept[:, 0] = 1
    model = OLSModel(design=Xintercept)
    results = model.fit(Y)
    assert_almost_equal(results.resid.mean(), 0)


def test_predicted_r_square():
    Xshort = X.copy()[:10, :]
    Yshort = Y.copy()[:10]

    # Signal of 10 elements should be completely
    # predicted by 10 predictors (short of some numerical
    # rounding errors)
    model = OLSModel(design=Xshort)
    results = model.fit(Yshort)
    assert_almost_equal(results.resid.sum(), 0)
    assert_array_almost_equal(results.predicted, Yshort)
    assert_almost_equal(results.r_square, 1.0)


def test_OLS_degenerate():
    Xd = X.copy()
    Xd[:, 0] = Xd[:, 1] + Xd[:, 2]
    model = OLSModel(design=Xd)
    results = model.fit(Y)
    assert results.df_resid == 31


def test_AR_degenerate():
    Xd = X.copy()
    Xd[:, 0] = Xd[:, 1] + Xd[:, 2]
    model = ARModel(design=Xd, rho=0.9)
    results = model.fit(Y)
    assert results.df_resid == 31
