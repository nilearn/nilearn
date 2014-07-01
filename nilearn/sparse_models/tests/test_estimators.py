import os
import sys
import itertools
from nose.tools import nottest, assert_equal
import numpy as np
from sklearn.utils import extmath
from sklearn.linear_model import Lasso
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from ..estimators import (TVl1Regressor, TVl1Classifier, SmoothLassoClassifier,
                          SmoothLassoRegressor)

# Data used in almost all tests
fn = lambda f, x, n: f(fn(f, x, n - 1)) if n > 1 else f(x)
ROOT = fn(os.path.dirname, os.path.dirname(__file__), 4)
CACHE = os.path.join(ROOT, "cache")
sys.path.append(os.path.join(ROOT, "examples/proximal"))
from .simulate_smooth_lasso_data import create_smooth_simulation_data
X, y, w, mask = create_smooth_simulation_data(
    snr=1., n_samples=10, size=4, n_points=5, random_state=42)
rng = check_random_state(42)


def test_tv_regression_2D_image_doesnt_crash():
    dim = (16, 16)
    W_init = np.zeros(dim)
    W_init[2:6, 3:7] = 1
    np.random.seed(0)
    n = 40
    p = dim[0] * dim[1]
    X = np.ones((n, 1)) + W_init.ravel().T
    X += np.random.randn(n, p)
    y = np.dot(X, W_init.ravel())
    alpha = 1.

    for l1_ratio in [1.]:
        TVl1Regressor(alpha=alpha, l1_ratio=l1_ratio,
                      max_iter=10).fit(X, y)


def test_tv_regression_3D_image_doesnt_crash():
    dim = (3, 4, 5)
    W_init = np.zeros(dim)
    W_init[2:3, 3:, 1:3] = 1

    np.random.seed(0)
    n = 10
    p = dim[0] * dim[1] * dim[2]
    X = np.ones((n, 1)) + W_init.ravel().T
    X += np.random.randn(n, p)
    y = np.dot(X, W_init.ravel())
    alpha = 1.

    for l1_ratio in [0., .5, 1.]:
        TVl1Regressor(alpha=alpha, l1_ratio=l1_ratio,
                      max_iter=5).fit(X, y)


def test_log_reg_vs_smooth_lasso_two_classes_iris(C=1., tol=1e-10,
                                                  zero_thr=1e-4):
    # Test for one of the extreme cases of Smooth Lasso: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Smooth Lasso's coefficients'
    # performance with the coefficients obtained from Scikit-Learn's
    # LogisticRegression, with l1 penalty, in a 2 classes classification task
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    tvl1 = TVl1Classifier(alpha=1. / C / X.shape[0], l1_ratio=1., tol=tol,
                          verbose=0, max_iter=1000).fit(X, y)
    sklogreg = LogisticRegression(penalty="l1", fit_intercept=True,
                                  tol=tol, C=C).fit(X, y)

    # compare supports
    np.testing.assert_array_equal((np.abs(tvl1.coef_) < zero_thr),
                                  (np.abs(sklogreg.coef_) < zero_thr))

    # compare predictions
    np.testing.assert_array_equal(tvl1.predict(X), sklogreg.predict(X))


def test_smooth_lasso_works_without_mask():
    rng = np.random.RandomState(42)
    n_samples = 10
    n_features = 125
    for l1_ratio in [0., .5, 1.]:
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        SmoothLassoRegressor(l1_ratio=l1_ratio, mask=None).fit(X, y)
        SmoothLassoClassifier(l1_ratio=l1_ratio, mask=None).fit(X, (y > 0))


@nottest
def test_log_reg_vs_smooth_lasso_multiclass(C=1., tol=1e-6):
    # Test for one of the extreme cases of Smooth Lasso: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Smooth Lasso's coefficients'
    # performance with the coefficients obtained from Scikit-Learn's
    # LogisticRegression, with l1 penalty, in a 4 classes classification task
    iris = load_iris()
    sl = SmoothLassoClassifier(alpha=1. / C / iris.data.shape[0],
                               l1_ratio=1., tol=tol).fit(
        iris.data, iris.target)
    sklogreg = LogisticRegression(penalty="l1", C=C, fit_intercept=True,
                                  tol=tol).fit(iris.data, iris.target)

    # compare supports
    np.testing.assert_array_equal((sl.coef_ == 0.), (sklogreg.coef_ == 0.))

    # compare predictions
    np.testing.assert_array_equal(sl.predict(iris.data),
                                  sklogreg.predict(iris.data))


def test_lasso_vs_smooth_lasso():
    # Test for one of the extreme cases of Smooth Lasso: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Smooth Lasso's performance with
    # Scikit-Learn lasso
    lasso = Lasso(max_iter=100, tol=1e-8, normalize=False)
    smooth_lasso = SmoothLassoRegressor(mask=mask, alpha=1, l1_ratio=1,
                                        max_iter=400, normalize=False)
    lasso.fit(X, y)
    smooth_lasso.fit(X, y)
    lasso_perf = 0.5 / y.size * extmath.norm(np.dot(
        X, lasso.coef_) - y) ** 2 + np.sum(np.abs(lasso.coef_))
    smooth_lasso_perf = 0.5 / y.size * extmath.norm(
        np.dot(X, smooth_lasso.coef_) - y) ** 2\
        + np.sum(np.abs(smooth_lasso.coef_))
    np.testing.assert_almost_equal(smooth_lasso_perf, lasso_perf, decimal=3)


def test_params_correctly_propagated_in_constructors():
    for model_class, alpha, l1_ratio in itertools.product(
        [SmoothLassoRegressor, SmoothLassoClassifier, TVl1Regressor,
         TVl1Classifier], [.4, .01], [.5, 1.]):
        cvobj = model_class(alpha=alpha, l1_ratio=l1_ratio)
        assert_equal(cvobj.alpha, alpha)
        assert_equal(cvobj.l1_ratio, l1_ratio)
