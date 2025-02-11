"""Test functions for models.regression."""

import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from nilearn.glm import ARModel, OLSModel, SimpleRegressionResults


@pytest.fixture()
def X(rng):  # noqa: N802
    return rng.standard_normal(size=(40, 10))


@pytest.fixture()
def Y(rng):  # noqa: N802
    return rng.standard_normal(size=(40, 10))


def test_ols(X, Y):
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert results.df_residuals == 30
    assert results.residuals.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_ar(X, Y):
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert results.df_residuals == 30
    assert results.residuals.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_residuals(X, Y):
    # If design matrix contains an intercept, the
    # mean of the residuals should be 0 (short of
    # some numerical rounding errors)
    X[:, 0] = 1
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert_almost_equal(results.residuals.mean(), 0)
    assert len(results.whitened_residuals) == 40


def test_predicted_r_square(X, Y):
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


def test_ols_degenerate(X, Y):
    X[:, 0] = X[:, 1] + X[:, 2]
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert results.df_residuals == 31


def test_ar_degenerate(X, Y):
    X[:, 0] = X[:, 1] + X[:, 2]
    model = ARModel(design=X, rho=0.9)
    results = model.fit(Y)
    assert results.df_residuals == 31


def test_simple_results(X, Y):
    model = OLSModel(X)
    results = model.fit(Y)

    simple_results = SimpleRegressionResults(results)
    assert_array_equal(results.predicted, simple_results.predicted(X))
    assert_array_equal(results.residuals, simple_results.residuals(Y, X))
    assert_array_equal(
        results.normalized_residuals, simple_results.normalized_residuals(Y, X)
    )
