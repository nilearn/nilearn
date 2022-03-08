"""
Test functions for models.regression
"""

import numpy as np
import pytest

from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_array_equal,
                           )

from nilearn.glm import OLSModel, ARModel


RNG = np.random.RandomState(42)
X = RNG.standard_normal(size=(40, 10))
Y = RNG.standard_normal(size=(40,))


def test_OLS():
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert results.df_residuals == 30
    assert results.residuals.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_AR():
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert results.df_residuals == 30
    assert results.residuals.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_residuals():
    Xintercept = X.copy()

    # If design matrix contains an intercept, the
    # mean of the residuals should be 0 (short of
    # some numerical rounding errors)
    Xintercept[:, 0] = 1
    model = OLSModel(design=Xintercept)
    results = model.fit(Y)
    assert_almost_equal(results.residuals.mean(), 0)
    assert len(results.whitened_residuals) == 40


def test_predicted_r_square():
    Xshort = X.copy()[:10, :]
    Yshort = Y.copy()[:10]

    # Signal of 10 elements should be completely
    # predicted by 10 predictors (short of some numerical
    # rounding errors)
    model = OLSModel(design=Xshort)
    results = model.fit(Yshort)
    assert_almost_equal(results.residuals.sum(), 0)
    assert_array_almost_equal(results.predicted, Yshort)
    assert_almost_equal(results.r_square, 1.0)


def test_OLS_degenerate():
    Xd = X.copy()
    Xd[:, 0] = Xd[:, 1] + Xd[:, 2]
    model = OLSModel(design=Xd)
    results = model.fit(Y)
    assert results.df_residuals == 31


def test_AR_degenerate():
    Xd = X.copy()
    Xd[:, 0] = Xd[:, 1] + Xd[:, 2]
    model = ARModel(design=Xd, rho=0.9)
    results = model.fit(Y)
    assert results.df_residuals == 31


def test_resid_rename_warnings_ols():
    model = OLSModel(design=X)
    results_ols = model.fit(Y)

    with pytest.warns(FutureWarning,
                      match="'df_resid'"):
        assert_array_equal(results_ols.df_resid,
                           results_ols.df_residuals)

    with pytest.warns(FutureWarning,
                      match="'resid'"):
        assert_array_equal(results_ols.resid,
                           results_ols.residuals)

    with pytest.warns(FutureWarning,
                      match="'wresid'"):
        assert_array_equal(results_ols.wresid,
                           results_ols.whitened_residuals)

    with pytest.warns(FutureWarning,
                      match="'norm_resid'"):
        assert_array_equal(results_ols.norm_resid,
                           results_ols.normalized_residuals)

    with pytest.warns(FutureWarning,
                      match="'wY'"):
        assert_array_equal(results_ols.wY,
                           results_ols.whitened_Y)

    with pytest.warns(FutureWarning,
                      match="'wdesign'"):
        assert_array_equal(results_ols.wdesign,
                           results_ols.whitened_design)

    with pytest.warns(FutureWarning,
                      match="'df_resid'"):
        assert_array_equal(model.df_resid,
                           model.df_residuals)

    with pytest.warns(FutureWarning,
                      match="'wdesign'"):
        assert_array_equal(model.wdesign,
                           model.whitened_design)


def test_resid_rename_warnings_ar():
    model = ARModel(design=X, rho=0.4)
    results_ar = model.fit(Y)
    with pytest.warns(FutureWarning, match="'resid'"):
        assert_array_equal(results_ar.resid, results_ar.residuals)
    with pytest.warns(FutureWarning, match="'wresid"):
        assert_array_equal(results_ar.wresid, results_ar.whitened_residuals)
