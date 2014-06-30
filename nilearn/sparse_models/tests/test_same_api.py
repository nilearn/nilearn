"""
Make sure all models are using thesame low-level API (
for computing image gradient, loss functins, etc.).

"""

import os
import sys
from nose.tools import nottest
import numpy as np
from sklearn.datasets import load_iris
from ..common import (squared_loss, squared_loss_grad,
                      compute_logistic_lipschitz_constant,
                      squared_loss_lipschitz_constant)
from ..smooth_lasso import (squared_loss_and_spatial_grad,
                            logistic_derivative_lipschitz_constant,
                            squared_loss_derivative_lipschitz_constant,
                            smooth_lasso_squared_loss, smooth_lasso_logistic,
                            squared_loss_and_spatial_grad_derivative)
from ..estimators import SmoothLassoRegressor, SmoothLassoClassifier
from ..tv import tvl1_solver
from ..cv import TVl1Regressor, TVl1Classifier
from nose.tools import assert_equal

fn = lambda f, x, n: f(fn(f, x, n - 1)) if n > 1 else f(x)
ROOT = fn(os.path.dirname, os.path.dirname(__file__), 4)
CACHE = os.path.join(ROOT, "cache")
sys.path.append(os.path.join(ROOT, "examples/proximal"))


def _make_data():
    dim = (16, 16)
    W_init = np.zeros(dim)
    W_init[2:6, 3:7] = 1
    np.random.seed(0)
    n = 10
    p = dim[0] * dim[1]
    X = np.ones((n, 1)) + W_init.ravel().T
    X += np.random.randn(n, p)
    y = np.dot(X, W_init.ravel())
    mask = np.ones(X.shape[1]).astype(np.bool).reshape(dim)

    return X, y, mask


def test_same_energy_calculus_pure_lasso():
    rng = np.random.RandomState(42)
    dim = (16, 16)
    np.random.seed(0)
    n = 40
    p = dim[0] * dim[1]
    w = rng.randn(*dim)
    X = np.ones((n, 1)) + w.ravel().T
    X += np.random.randn(n, p)
    y = np.dot(X, w.ravel())

    # check funcvals
    mask = np.ones(dim).astype(np.bool)
    f1 = squared_loss(X, y, w, mask=mask)
    f2 = squared_loss_and_spatial_grad(X, y, w.ravel(), mask, 0.)
    assert_equal(f1, f2)

    # check derivatives
    g1 = squared_loss_grad(X, y, w, mask=mask)
    g2 = squared_loss_and_spatial_grad_derivative(X, y, w.ravel(), mask, 0.)
    np.testing.assert_array_equal(g1, g2)


def test_lipschitz_constant_lass_mse():
    rng = np.random.RandomState(42)
    l1_ratio = 1.
    alpha = .1
    n, p = 4, 10
    X = rng.randn(n, p)
    mask = np.ones(X.shape[1]).astype(np.bool)
    grad_weight = alpha * X.shape[0] * (1. - l1_ratio)
    a = squared_loss_derivative_lipschitz_constant(X, mask, grad_weight)
    b = squared_loss_lipschitz_constant(X)
    np.testing.assert_almost_equal(a, b)


def test_lipschitz_constant_lass_logreg():
    rng = np.random.RandomState(42)
    l1_ratio = 1.
    alpha = .1
    n, p = 4, 10
    X = rng.randn(n, p)
    mask = np.ones(X.shape[1]).astype(np.bool)
    grad_weight = alpha * X.shape[0] * (1. - l1_ratio)
    a = logistic_derivative_lipschitz_constant(X, mask, grad_weight)
    b = compute_logistic_lipschitz_constant(X)
    assert_equal(a, b)


def test_smoothlasso_and_tvl1_same_for_pure_l1(max_iter=10, decimal=2):
    ###############################################################
    # smoothlasso_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    X, y, _ = _make_data()
    alpha = .1

    # results should be exactly the same for pure lasso
    a = tvl1_solver(X, y, alpha, 1., loss="mse", max_iter=max_iter)[0]
    b = smooth_lasso_squared_loss(X, y, alpha, 1., max_iter=max_iter)[0]
    sl = SmoothLassoRegressor(alpha=alpha, l1_ratio=1.,
                              max_iter=max_iter).fit(X, y)
    tvl1 = TVl1Regressor(alpha=alpha, l1_ratio=1.,
                         max_iter=max_iter).fit(X, y)

    # Should be exactly the same (except for numerical errors).
    # However because of the TV-l1 prox approx, results might be 'slightly'
    # different.
    assert_equal(alpha * X.shape[0], tvl1.alpha_)  # test _prefit
    np.testing.assert_array_almost_equal(a, b, decimal=decimal)
    np.testing.assert_array_almost_equal(sl.coef_, tvl1.coef_, decimal=decimal)


def test_smoothlasso_and_tvl1_same_for_pure_l1_logistic(max_iter=10,
                                                        decimal=3):
    ###############################################################
    # smoothlasso_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    iris = load_iris()
    X, y = iris.data, iris.target
    y = (y > 0)
    alpha = 1. / X.shape[0]
    mask = np.ones(X.shape[1]).astype(np.bool)
    max_iter = 10

    # results should be exactly the same for pure lasso
    for bt in [True, False]:
        a = smooth_lasso_logistic(X, y, alpha, 1., mask=mask,
                                  max_iter=max_iter,
                                  backtracking=bt)[0]
        b = tvl1_solver(X, y, alpha, 1., loss="logistic", mask=mask,
                        max_iter=max_iter, backtracking=bt)[0]
        sl = SmoothLassoClassifier(alpha=alpha, l1_ratio=1., verbose=0,
                                   max_iter=max_iter, mask=mask,
                                   backtracking=bt).fit(X, y)
        tvl1 = TVl1Classifier(alpha=alpha, l1_ratio=1.,
                              max_iter=max_iter, mask=mask,
                              backtracking=bt).fit(X, y)

        # should be exactly the same (except for numerical errors)
        np.testing.assert_array_almost_equal(a, b, decimal=decimal)
        np.testing.assert_array_almost_equal(sl.coef_[0], tvl1.coef_[0],
                                             decimal=decimal)


def test_logreg_with_mask_issue_10():
    rng = np.random.RandomState(42)
    shape = (3, 4, 5)
    n_samples = 10
    mask = np.zeros(np.prod(shape))
    mask[4:21] = 1
    mask = mask.reshape(shape).astype(np.bool)
    X = rng.randn(n_samples, mask.sum())
    y = np.sign(rng.randn(n_samples))
    alpha = 1.
    l1_ratio = .5

    for model in [TVl1Regressor, TVl1Classifier, SmoothLassoRegressor,
                  SmoothLassoClassifier]:
        # ensure that our fix didn't break anythx else
        model(alpha=alpha, l1_ratio=l1_ratio, mask=mask).fit(X, y)


def test_smoothlasso_and_tv_same_for_pure_l1_another_test(decimal=2):
    ###############################################################
    # smoothlasso_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    dim = (16, 16)
    W_init = np.zeros(dim)
    W_init[2:6, 3:7] = 1
    np.random.seed(0)
    n = 10
    p = dim[0] * dim[1]
    X = np.ones((n, 1)) + W_init.ravel().T
    X += np.random.randn(n, p)
    y = np.dot(X, W_init.ravel())
    mask = np.ones(X.shape[1]).astype(np.bool).reshape(dim)
    alpha = .1
    l1_ratio = 1.
    max_iter = 10

    sl = SmoothLassoRegressor(alpha=alpha, l1_ratio=l1_ratio,
                              max_iter=max_iter, mask=mask,
                              verbose=0).fit(X, y)
    tvl1 = TVl1Regressor(alpha=alpha, l1_ratio=l1_ratio,
                         max_iter=max_iter, mask=mask,
                         verbose=0).fit(X, y)

    # should be exactly the same (except for numerical errors)
    np.testing.assert_array_almost_equal(sl.coef_, tvl1.coef_, decimal=decimal)


@nottest
def test_w_shapes():
    """Test that solvers handle w of same shape (during callbacks, etc.)."""
    pass
