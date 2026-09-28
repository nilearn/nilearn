"""Test functions for models.regression."""

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from nilearn.glm import ARModel, OLSModel, SimpleRegressionResults


@pytest.fixture()
def X(rng) -> np.ndarray:  # noqa: N802
    """Return a random 40x10 design matrix."""
    return rng.standard_normal(size=(40, 10))


@pytest.fixture()
def Y(rng) -> np.ndarray:  # noqa: N802
    """Return a random 40x10 array of observations."""
    return rng.standard_normal(size=(40, 10))


def test_ols(X, Y):
    """Test that OLSModel fits and produces outputs of the expected shape."""
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert results.df_residuals == 30
    assert results.residuals.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_ar(X, Y):
    """Test that ARModel fits and produces outputs of the expected shape."""
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)

    assert results.df_residuals == 30
    assert results.residuals.shape[0] == 40
    assert results.predicted.shape[0] == 40


def test_residuals(X, Y):
    """Test that residuals have zero mean when design has an intercept.

    Short of some numerical rounding errors.
    """
    X[:, 0] = 1
    model = OLSModel(design=X)
    results = model.fit(Y)

    assert_almost_equal(results.residuals.mean(), 0)
    assert len(results.whitened_residuals) == 40


def test_predicted_r_square(X, Y):
    """Test that a fully-determined OLS fit has r_square of 1.

    Signal of 10 elements should be completely predicted by 10
    predictors, short of some numerical rounding errors.
    """
    Xshort = X.copy()[:10, :]
    Yshort = Y.copy()[:10]

    model = OLSModel(design=Xshort)
    results = model.fit(Yshort)

    assert_almost_equal(results.residuals.sum(), 0)
    assert_array_almost_equal(results.predicted, Yshort)
    assert_almost_equal(results.r_square, 1.0)


def test_ols_degenerate(X, Y):
    """Test that OLSModel's degrees of freedom account for collinearity."""
    X[:, 0] = X[:, 1] + X[:, 2]
    model = OLSModel(design=X)
    results = model.fit(Y)

    assert results.df_residuals == 31


def test_ar_degenerate(X, Y):
    """Test that ARModel's degrees of freedom account for collinearity."""
    X[:, 0] = X[:, 1] + X[:, 2]
    model = ARModel(design=X, rho=0.9)
    results = model.fit(Y)

    assert results.df_residuals == 31


@pytest.mark.ai_generated
@pytest.mark.parametrize(
    "model_cls, kwargs", [(OLSModel, {}), (ARModel, {"rho": 0.4})]
)
def test_dispersion_rank_deficient(X, Y, model_cls, kwargs):
    """Check that dispersion and t do not depend on a redundant column.

    Regression test for https://github.com/nilearn/nilearn/issues/6607
    """
    X_redundant = np.column_stack([X, X[:, 1] + X[:, 2]])
    full_rank = model_cls(design=X, **kwargs).fit(Y)
    rank_deficient = model_cls(design=X_redundant, **kwargs).fit(Y)

    assert rank_deficient.df_residuals == full_rank.df_residuals
    assert_array_almost_equal(rank_deficient.dispersion, full_rank.dispersion)
    assert_array_almost_equal(
        rank_deficient.t(column=0), full_rank.t(column=0)
    )


def test_simple_results(X, Y):
    """Test that SimpleRegressionResults matches the full results object."""
    model = OLSModel(X)
    results = model.fit(Y)

    simple_results = SimpleRegressionResults(results)

    assert_array_equal(results.predicted, simple_results.predicted(X))
    assert_array_equal(results.residuals, simple_results.residuals(Y, X))
    assert_array_equal(
        results.normalized_residuals, simple_results.normalized_residuals(Y, X)
    )
