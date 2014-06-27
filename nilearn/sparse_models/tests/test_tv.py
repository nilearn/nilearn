# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

# $Id: test_tv.py 336 2010-04-21 18:07:26Z gramfort $

import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from ..estimators import TVl1Regressor, TVl1Classifier

fn = lambda f, x, n: f(fn(f, x, n - 1)) if n > 1 else f(x)
ROOT = fn(os.path.dirname, os.path.dirname(__file__), 4)
CACHE = os.path.join(ROOT, "cache")
sys.path.append(os.path.join(ROOT, "examples/proximal"))


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
