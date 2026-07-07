"""Test module for functions related cost functions (including penalties)."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.optimize import check_grad

from nilearn.decoding._objective_functions import (
    divergence_id,
    gradient_id,
    logistic_loss,
    logistic_loss_grad,
)

L1_RATIO = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


@pytest.mark.parametrize("ndim", range(1, 5))
@pytest.mark.parametrize("l1_ratio", L1_RATIO)
@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.ai_generated
def test_grad_div_adjoint_arbitrary_ndim(rng, ndim, l1_ratio, size):
    """Test that <D x, y> = <x, DT y> for x and y random vectors."""
    shape = tuple([size] * ndim)
    x = rng.normal(size=shape)
    y = rng.normal(size=[ndim + 1, *shape])

    assert_almost_equal(
        np.sum(gradient_id(x, l1_ratio=l1_ratio) * y),
        -np.sum(x * divergence_id(y, l1_ratio=l1_ratio)),
    )


@pytest.mark.parametrize("l1_ratio", L1_RATIO)
@pytest.mark.parametrize("size", [1, 2, 10])
@pytest.mark.ai_generated
def test_1d_gradient_id(l1_ratio, size):
    """Test gradient_id shape and last component on 1D input."""
    img = np.arange(size)

    gid = gradient_id(img, l1_ratio=l1_ratio)

    assert_array_equal(gid.shape, [img.ndim + 1, *img.shape])
    assert_array_equal(l1_ratio * img, gid[-1])


@pytest.mark.parametrize("l1_ratio", L1_RATIO)
@pytest.mark.ai_generated
def test_2d_gradient_id(l1_ratio):
    """Test gradient_id shape and last component on 2D input."""
    img = np.array([[1, 3], [4, 2]])

    gid = gradient_id(img, l1_ratio)

    assert_array_equal(gid.shape, [img.ndim + 1, *img.shape])
    assert_array_equal(l1_ratio * img, gid[-1])


@pytest.mark.parametrize("l1_ratio", L1_RATIO)
@pytest.mark.ai_generated
def test_3d_gradient_id(l1_ratio):
    """Test gradient_id shape on a non-square 2D input."""
    img = np.array([[1, 3], [4, 2], [1, 0]])

    gid = gradient_id(img, l1_ratio)
    assert_array_equal(gid.shape, [img.ndim + 1, *img.shape])


@pytest.mark.ai_generated
def test_logistic_loss_derivative(rng, n_samples=4, n_features=10, decimal=5):
    """Test that logistic_loss_grad matches the numerical gradient."""
    X = rng.standard_normal((n_samples, n_features))
    y = rng.standard_normal(n_samples)
    n_features = X.shape[1]
    w = rng.standard_normal(n_features + 1)
    assert_almost_equal(
        check_grad(
            lambda w: logistic_loss(X, y, w),
            lambda w: logistic_loss_grad(X, y, w),
            w,
        ),
        0.0,
        decimal=decimal,
    )
