"""
Test module for common.py

"""

import numpy as np
from scipy.optimize import check_grad
from ..common import (gradient_id, logistic,
                      logistic_grad, _unmask,
                      test_grad_div_adjoint_arbitrary_ndim)
from ..estimators import _BaseEstimator
from nose.tools import raises


def test_1D_gradient_id():
    for size in [1, 2, 10]:
        img = np.arange(size)
        for l1_ratio in [0., .1, .3, .5, .7, .9, 1.]:
            gid = gradient_id(img, l1_ratio=l1_ratio)

            np.testing.assert_array_equal(
                gid.shape, [img.ndim + 1] + list(img.shape))

            np.testing.assert_array_equal(l1_ratio * img, gid[-1])


def test_2D_gradient_id():
    img = np.array([[1, 3], [4, 2]])
    for l1_ratio in [0., .1, .3, .5, .7, .9, 1.]:
        gid = gradient_id(img, l1_ratio)

        np.testing.assert_array_equal(
            gid.shape, [img.ndim + 1] + list(img.shape))

        np.testing.assert_array_equal(l1_ratio * img, gid[-1])


def test_3D_gradient_id():
    img = np.array([[1, 3], [4, 2], [1, 0]])
    for l1_ratio in [0., .1, .3, .5, .7, .9, 1.]:
        gid = gradient_id(img, l1_ratio)

        np.testing.assert_array_equal(
            gid.shape, [img.ndim + 1] + list(img.shape))


def test_logistic_loss_derivative(n_samples=4, n_features=10, random_state=42,
                                  decimal=5):

    rng = np.random.RandomState(random_state)

    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    n_features = X.shape[1]
    w = rng.randn(n_features + 1)

    np.testing.assert_almost_equal(check_grad(
        lambda w: logistic(X, y, w),
        lambda w: logistic_grad(X, y, w), w), 0., decimal=decimal)

    np.testing.assert_almost_equal(check_grad(
        lambda w: logistic(X, y, w),
        lambda w: logistic_grad(X, y, w), w), 0., decimal=decimal)


def test_grad_div_adjoint_arbitrary_ndim_():
    for size in [3, 4, 5]:
        test_grad_div_adjoint_arbitrary_ndim(size=size)


@raises(ValueError)
def test_baseestimator_invalide_l1_ratio():
    _BaseEstimator(l1_ratio=2.)


def test_unmasl(size=5):
    rng = np.random.RandomState(42)
    for ndim in xrange(1, 4):
        shape = [size] * ndim
        mask = np.zeros(shape).astype(np.bool)
        mask[rng.rand(*shape) > .8] = 1
        support = rng.randn(mask.sum())
        full = _unmask(support, mask)
        np.testing.assert_array_equal(full.shape, shape)
        np.testing.assert_array_equal(full[mask], support)
