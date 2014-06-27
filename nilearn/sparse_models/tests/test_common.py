"""
Test module for common.py

"""

import numpy as np
from scipy.optimize import check_grad
import itertools
from ..common import (gradient_id,
                      tv_l1_reg_objective,
                      tv_l1_from_gradient, compute_mse, logistic,
                      logistic_grad,
                      compute_logistic_lipschitz_constant,
                      check_lipschitz_continuous,
                      compute_mse_lipschitz_constant,
                      test_grad_div_adjoint_arbitrary_ndim)
from ..estimators import _BaseEstimator
from ..prox_tv_l1 import prox_tv_l1
from ..operators import prox_l1
from nose.tools import assert_true, assert_equal, raises


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


def test_prox_l1(n_features=10):

    rng = np.random.RandomState(42)
    x = rng.randn(n_features, 1)
    tau = .3
    s = prox_l1(x.copy(), tau)
    p = x - s  # projection + shrinkage = id

    # We should have ||s(a) - s(b)||^2 <= ||a - b||^2 - ||p(a) - p(b)||^2
    # for all a and b (this is strong non-expansiveness
    for (a, b), (pa, pb), (sa, sb) in zip(*[itertools.product(z[0], z[0])
                                            for z in [x, p, s]]):
        assert_true((sa - sb) ** 2 <= (a - b) ** 2 - (pa - pb) ** 2)


def test_prox_tv_l1_approximates_prox_l1_for_lasso(size=15, random_state=42,
                                                   decimal=4, dgap_tol=1e-7):

    rng = np.random.RandomState(random_state)

    l1_ratio = 1.  # pure LASSO
    for ndim in xrange(1, 4):
        shape = [size] * ndim
        z = rng.randn(*shape)
        for weight in np.logspace(-10, 10, num=10):
            # use prox_tv_l1 approximation to prox_l1
            a = prox_tv_l1(z.copy(), weight=weight, l1_ratio=l1_ratio,
                           dgap_tol=dgap_tol, return_info=False,
                           max_iter=10,
                           )[-1].ravel()

            # use exact closed-form soft shrinkage formula for prox_l1
            b = prox_l1(z.copy(), weight)[-1].ravel()

            # results shoud be close in l-infinity norm
            np.testing.assert_almost_equal(np.abs(a - b).max(),
                                           0., decimal=decimal)


def test_tv_l1_from_gradient(size=5, n_samples=10, random_state=42,
                             decimal=8):

    rng = np.random.RandomState(random_state)

    shape = [size] * 3
    n_voxels = np.prod(shape)
    X = rng.randn(n_samples, n_voxels)
    y = rng.randn(n_samples)
    w = rng.randn(*shape)

    for alpha in [0., 1e-1, 1e-3]:
        for l1_ratio in [0., .5, 1.]:
            gradid = gradient_id(w, l1_ratio=l1_ratio)
            assert_equal(tv_l1_reg_objective(
                X, y, w.copy(), alpha, l1_ratio, shape=shape),
                compute_mse(X, y, w.copy(),
                            compute_grad=False) + alpha * tv_l1_from_gradient(
                    gradid))


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


def test_logistic_lipschitz(n_samples=4, n_features=2, random_state=42):
    rng = np.random.RandomState(random_state)

    for scaling in np.logspace(-3, 3, num=7):
        X = rng.randn(n_samples, n_features) * scaling
        y = rng.randn(n_samples)
        n_features = X.shape[1]

        L = compute_logistic_lipschitz_constant(X)
        check_lipschitz_continuous(lambda w: logistic(
            X, y, w), n_features + 1, L)


def test_mse_lipschitz(n_samples=4, n_features=2, random_state=42):
    rng = np.random.RandomState(random_state)

    for scaling in np.logspace(-3, 3, num=7):
        X = rng.randn(n_samples, n_features) * scaling
        y = rng.randn(n_samples)
        n_features = X.shape[1]

        L = compute_mse_lipschitz_constant(X)
        check_lipschitz_continuous(lambda w: compute_mse(
            X, y, w, compute_energy=False), n_features, L)


def test_grad_div_adjoint_arbitrary_ndim_():
    for size in [3, 4, 5]:
        test_grad_div_adjoint_arbitrary_ndim(size=size)


@raises(ValueError)
def test_baseestimator_invalide_l1_ratio():
    _BaseEstimator(l1_ratio=2.)
