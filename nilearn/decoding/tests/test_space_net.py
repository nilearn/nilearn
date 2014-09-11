import itertools
from functools import partial
from nose import SkipTest
from nose.tools import assert_equal, assert_true
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_iris
from sklearn.utils import extmath
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from ..space_net import (TVl1Regressor, TVl1Classifier, SmoothLassoClassifier,
                         SmoothLassoRegressor,
                         RegressorFeatureSelector, ClassifierFeatureSelector,
                         EarlyStoppingCallback, _my_alpha_grid,
                         path_scores)
from ..space_net_solvers import (smooth_lasso_logistic,
                                 smooth_lasso_squared_loss)

rng = check_random_state(42)
logistic_path_scores = partial(path_scores, logistic=True)
squared_loss_path_scores = partial(path_scores, logistic=False)

# Data used in almost all tests
from .simulate_smooth_lasso_data import create_smooth_simulation_data
X, y, w, mask = create_smooth_simulation_data(
    snr=1., n_samples=10, size=4, n_points=5, random_state=42)


@SkipTest
def test_same_lasso_classifier_cv():
    # XXX test fails with early stopping in CV
    l1_ratio = 1.
    memory = Memory("cache")
    iris = load_iris()
    X = iris.data
    y = iris.target
    y = 2 * (y == 2) - 1
    tol = 1e-3

    # classification
    n_alphas = 5
    tvl1classifiercv = TVl1Classifier(n_alphas=n_alphas, memory=memory, cv=2,
                                        l1_ratio=l1_ratio, tol=tol).fit(X, y)
    slclassifiercv = SmoothLassoClassifier(n_alphas=n_alphas, memory=memory,
                                   cv=2, l1_ratio=l1_ratio, tol=tol).fit(X, y)
    assert_equal(tvl1classifiercv.alpha_, slclassifiercv.alpha_)

    # regression
    tvl1regressorcv = TVl1Regressor(n_alphas=n_alphas, memory=memory, cv=2,
                                      l1_ratio=l1_ratio, tol=tol).fit(X, y)
    slregressorcv = SmoothLassoRegressor(n_alphas=n_alphas, memory=memory,
                                           cv=2, l1_ratio=l1_ratio, tol=tol
                                           ).fit(X, y)
    assert_equal(tvl1regressorcv.alpha_, slregressorcv.alpha_)


def test_my_alpha_grid(n_samples=4, n_features=3):
    X = rng.randn(n_samples, n_features)
    y = np.arange(n_samples)

    for l1_ratio in [.5, 1.]:
        alpha_max = np.max(np.abs(np.dot(X.T, y))) / n_samples / l1_ratio
        assert_equal(_my_alpha_grid(X, y, n_alphas=1, l1_ratio=l1_ratio),
                     alpha_max)

    for standardize in [False, True]:
        for l1_ratio in [.5, 1.]:
            alpha_max = np.max(np.abs(np.dot(X.T, y))) / n_samples / l1_ratio
            for n_alphas in xrange(1, 10):
                alphas = _my_alpha_grid(
                    X, y, n_alphas=n_alphas, l1_ratio=l1_ratio,
                    standardize=standardize)
                if not standardize:
                    assert_equal(alphas.max(), alpha_max)
                assert_equal(n_alphas, len(alphas))


def test_my_alpha_grid_same_as_sk():
    try:
        from sklearn.linear_model.coordinate_descent import _alpha_grid
        iris = load_iris()
        X = iris.data
        y = iris.target
        for normalize in [True]:
            for fit_intercept in [True, False]:
                np.testing.assert_array_equal(_my_alpha_grid(
                        X, y, n_alphas=5, normalize=normalize,
                        fit_intercept=fit_intercept, standardize=True),
                        _alpha_grid(X, y, n_alphas=5, normalize=normalize,
                                    fit_intercept=fit_intercept))
    except ImportError:
        raise SkipTest


def test_featureselectors():
    import random
    iris = load_iris()
    X, y = iris.data, iris.target
    for ndim in range(1, 4):
        shape = [4] * ndim
        for percentile in [0, 10, 100]:
            for n_samples in [X.shape[0], X.shape[1] - 1]:
                for with_mask in [True, False]:
                    if with_mask:
                        mask = np.zeros(np.prod(shape)).astype(np.bool)
                        support = random.sample(xrange(np.prod(mask.shape)),
                                                X.shape[1])
                        mask[support] = 1
                        mask = mask.reshape(shape)
                    else:
                        mask = None

                    for selector_class in [RegressorFeatureSelector,
                                           ClassifierFeatureSelector]:
                        selector = selector_class(percentile=percentile,
                                                  mask=mask)
                        salt = int("Classifier" in selector_class.__name__)
                        X_ = X[:n_samples]
                        y_ = y[:n_samples]
                        X_ = selector.fit_transform(X_, y_)
                        if not mask is None:
                            assert_true(selector.mask_ is not None)
                        else:
                            assert_true(selector.mask_ is None)
                        coef_ = selector.inverse_transform(rng.randn(
                                selector.support_.sum() + salt))
                        assert_equal(len(coef_), X.shape[1] + salt)


def test_earlystoppingcallbackobject(n_samples=10, n_features=30):
    # This test evolves w so that every line of th EarlyStoppingCallback
    # code is executed a some point. This a kind of code fuzzing.
    X_test = rng.randn(n_samples, n_features)
    y_test = np.dot(X_test, np.ones(n_features))
    w = np.zeros(n_features)
    escb = EarlyStoppingCallback(X_test, y_test, verbose=1)
    for counter in xrange(50):
        k = min(counter, n_features - 1)
        w[k] = 1

        # jitter
        if k > 0 and rng.rand() > .9:
            w[k - 1] = 1 - w[k - 1]

        escb(dict(w=w, counter=counter))
        assert_equal(len(escb.test_errors), counter + 1)
        print np.mean(np.diff(escb.test_errors[-5:])), len(escb.test_errors)

        # restart
        if counter > 20:
            w *= 0.


def test_params_correctly_propagated_in_constructors():
    for cv_class, n_alphas, l1_ratio, n_jobs, cv, perc in itertools.product(
        [SmoothLassoRegressor, SmoothLassoClassifier, TVl1Regressor,
         TVl1Classifier], [.1, .01], [.5, 1.], [1, -1], [2, 3], [5, 10]):
        cvobj = cv_class(n_alphas=n_alphas, n_jobs=n_jobs, l1_ratio=l1_ratio,
                         cv=cv, screening_percentile=perc)
        assert_equal(cvobj.n_alphas, n_alphas)
        assert_equal(cvobj.l1_ratio, l1_ratio)
        assert_equal(cvobj.n_jobs, n_jobs)
        assert_equal(cvobj.cv, cv)
        assert_equal(cvobj.screening_percentile, perc)


def test_logistic_path_scores():
    iris = load_iris()
    X, y = iris.data, iris.target
    alphas = [1., .1, .01]
    test_scores, best_w, _ = logistic_path_scores(
        smooth_lasso_logistic, X, y, alphas, .5,
        range(len(X)), range(len(X)))
    assert_equal(len(test_scores), len(alphas))
    assert_equal(X.shape[1] + 1, len(best_w))


def test_squared_loss_path_scores():
    iris = load_iris()
    X, y = iris.data, iris.target
    alphas = [1., .1, .01]
    test_scores, best_w, _ = squared_loss_path_scores(
        smooth_lasso_squared_loss, X, y, alphas, .5,
        range(len(X)), range(len(X)))
    assert_equal(len(test_scores), len(alphas))
    assert_equal(X.shape[1], len(best_w))


def test_estimators_are_special_cv_objects():
    iris = load_iris()
    X, y = iris.data, iris.target
    alpha = 1.
    for cv_class in [SmoothLassoRegressor, SmoothLassoClassifier,
                     TVl1Regressor, TVl1Classifier]:
        cv = cv_class(alpha=alpha)
        cv.fit(X, y)
        np.testing.assert_array_equal([alpha], cv.alphas_)


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
    n_samples = 10
    n_features = 125
    for l1_ratio in [0., .5, 1.]:
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        SmoothLassoRegressor(l1_ratio=l1_ratio, mask=None,
                             max_iter=10, alpha=1.).fit(X, y)
        SmoothLassoClassifier(alpha=1., l1_ratio=l1_ratio, mask=None,
                              max_iter=10).fit(X, (y > 0))


@SkipTest
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
                                          max_iter=100, normalize=False)
    lasso.fit(X, y)
    smooth_lasso.fit(X, y)
    lasso_perf = 0.5 / y.size * extmath.norm(np.dot(
        X, lasso.coef_) - y) ** 2 + np.sum(np.abs(lasso.coef_))
    smooth_lasso_perf = 0.5 / y.size * extmath.norm(
        np.dot(X, smooth_lasso.coef_) - y) ** 2\
        + np.sum(np.abs(smooth_lasso.coef_))
    np.testing.assert_almost_equal(smooth_lasso_perf, lasso_perf, decimal=3)


def test_params_correctly_propagated_in_constructors_biz():
    for model_class, alpha, l1_ratio in itertools.product(
        [SmoothLassoRegressor, SmoothLassoClassifier, TVl1Regressor,
         TVl1Classifier], [.4, .01], [.5, 1.]):
        cvobj = model_class(alpha=alpha, l1_ratio=l1_ratio)
        assert_equal(cvobj.alpha, alpha)
        assert_equal(cvobj.l1_ratio, l1_ratio)
