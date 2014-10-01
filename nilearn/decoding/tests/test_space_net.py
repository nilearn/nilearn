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
from ..space_net import (EarlyStoppingCallback, _my_alpha_grid, path_scores,
                         SpaceNet)
from ..space_net_solvers import (smooth_lasso_logistic,
                                 smooth_lasso_squared_loss)

rng = check_random_state(42)
logistic_path_scores = partial(path_scores, classif=True)
squared_loss_path_scores = partial(path_scores, classif=False)

# Data used in almost all tests
from .test_same_api import to_niimgs
size = 4
from .simulate_smooth_lasso_data import create_smooth_simulation_data
X_, y, w, mask = create_smooth_simulation_data(
    snr=1., n_samples=10, size=size, n_points=5, random_state=42)
X, mask = to_niimgs(X_, [size] * 3)


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
    X, mask = to_niimgs(X, (2, 2, 2))

    # classification
    n_alphas = 5
    tvl1classifiercv = SpaceNet(
        mask=mask, n_alphas=n_alphas, memory=memory, cv=2,
        l1_ratio=l1_ratio, tol=tol, penalty="tvl1",
        classif=True).fit(X, y)
    slclassifiercv = SpaceNet(mask=mask, n_alphas=n_alphas, memory=memory,
                              classif=True, penalty="smooth-lasso",
                              cv=2, l1_ratio=l1_ratio, tol=tol).fit(X, y)
    assert_equal(tvl1classifiercv.alpha_, slclassifiercv.alpha_)

    # regression
    tvl1regressorcv = SpaceNet(
        mask=mask, n_alphas=n_alphas, memory=memory, cv=2,
        l1_ratio=l1_ratio, tol=tol, classif=False,
        penalty="tvl1").fit(X, y)
    slregressorcv = SpaceNet(
        mask=mask, n_alphas=n_alphas, memory=memory, classif=True,
        cv=2, l1_ratio=l1_ratio, tol=tol, penalty="smooth-lasso"
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
    for (penalty, classif, n_alphas, l1_ratio, n_jobs,
         cv, perc) in itertools.product(
        ["smooth-lasso", "tvl1"], [True, False],
        [.1, .01], [.5, 1.], [1, -1], [2, 3], [5, 10]):
        cvobj = SpaceNet(
            mask="dummy", n_alphas=n_alphas, n_jobs=n_jobs, l1_ratio=l1_ratio,
            cv=cv, screening_percentile=perc, penalty=penalty,
            classif=classif)
        assert_equal(cvobj.n_alphas, n_alphas)
        assert_equal(cvobj.l1_ratio, l1_ratio)
        assert_equal(cvobj.n_jobs, n_jobs)
        assert_equal(cvobj.cv, cv)
        assert_equal(cvobj.screening_percentile, perc)


def test_logistic_path_scores():
    iris = load_iris()
    X, y = iris.data, iris.target
    mask = np.ones(X.shape[1]).astype(np.bool)
    alphas = [1., .1, .01]
    test_scores, best_w, _ = logistic_path_scores(
        smooth_lasso_logistic, X, y, alphas, .5,
        range(len(X)), range(len(X)), mask=mask)
    assert_equal(len(test_scores), len(alphas))
    assert_equal(X.shape[1] + 1, len(best_w))


def test_squared_loss_path_scores():
    iris = load_iris()
    X, y = iris.data, iris.target
    mask = np.ones(X.shape[1]).astype(np.bool)
    alphas = [1., .1, .01]
    test_scores, best_w, _ = squared_loss_path_scores(
        smooth_lasso_squared_loss, X, y, alphas, .5,
        range(len(X)), range(len(X)), mask=mask)
    assert_equal(len(test_scores), len(alphas))
    assert_equal(X.shape[1], len(best_w))


def test_estimators_are_special_cv_objects():
    iris = load_iris()
    X, y = iris.data, iris.target
    alpha = 1.
    X, mask = to_niimgs(X, (2, 2, 2))

    for penalty, classif in itertools.product(['smooth-lasso', 'tvl1'],
                                             [True, False]):
        cv = SpaceNet(mask=mask, penalty=penalty, alpha=alpha)
        cv.fit(X, y)
        np.testing.assert_array_equal([alpha], cv.alphas_)


def test_tv_regression_simple():
    dim = (4, 4, 4)
    W_init = np.zeros(dim)
    W_init[2:3, 1:2, -2:] = 1
    np.random.seed(0)
    n = 40
    p = np.prod(dim)
    X = np.ones((n, 1)) + W_init.ravel().T
    X += np.random.randn(n, p)
    y = np.dot(X, W_init.ravel())
    X, mask = to_niimgs(X, dim)
    alpha = 1.

    for l1_ratio in [1.]:
        SpaceNet(mask=mask, alpha=alpha, l1_ratio=l1_ratio,
                 penalty="tvl1", classif=False, max_iter=10).fit(X, y)


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
    X, mask = to_niimgs(X, dim)

    for l1_ratio in [0., .5, 1.]:
        SpaceNet(mask=mask, alpha=alpha, l1_ratio=l1_ratio, penalty="tvl1",
                 classif=False, max_iter=10).fit(X, y)


def test_log_reg_vs_smooth_lasso_two_classes_iris(C=1., tol=1e-10,
                                                  zero_thr=1e-4):
    # Test for one of the extreme cases of Smooth Lasso: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Smooth Lasso's coefficients'
    # performance with the coefficients obtained from Scikit-Learn's
    # LogisticRegression, with l1 penalty, in a 2 classes classification task
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))
    tvl1 = SpaceNet(mask=mask, alpha=1. / C / X.shape[0], l1_ratio=1., tol=tol,
                    verbose=0, max_iter=1000, penalty="tvl1",
                    classif=True, screening_percentile=100.).fit(X_, y)
    sklogreg = LogisticRegression(penalty="l1", fit_intercept=True,
                                  tol=tol, C=C).fit(X, y)

    # compare supports
    np.testing.assert_array_equal((np.abs(tvl1.coef_) < zero_thr),
                                  (np.abs(sklogreg.coef_) < zero_thr))

    # compare predictions
    np.testing.assert_array_equal(tvl1.predict(X_), sklogreg.predict(X))


@SkipTest
def test_log_reg_vs_smooth_lasso_multiclass(C=1., tol=1e-6):
    # Test for one of the extreme cases of Smooth Lasso: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Smooth Lasso's coefficients'
    # performance with the coefficients obtained from Scikit-Learn's
    # LogisticRegression, with l1 penalty, in a 4 classes classification task
    iris = load_iris()
    mask = np.ones(X.shape[1]).astype(np.bool)
    sl = SpaceNet(mask=mask, alpha=1. / C / iris.data.shape[0],
                  l1_ratio=1., tol=tol, classif=True).fit(
        iris.data, iris.target)
    sklogreg = LogisticRegression(
        mask=mask, penalty="l1", C=C, fit_intercept=True,
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
    smooth_lasso = SpaceNet(mask=mask, alpha=1, l1_ratio=1, classif=False,
                            penalty="smooth-lasso",
                            max_iter=100, normalize=False)
    lasso.fit(X_, y)
    smooth_lasso.fit(X, y)
    lasso_perf = 0.5 / y.size * extmath.norm(np.dot(
        X_, lasso.coef_) - y) ** 2 + np.sum(np.abs(lasso.coef_))
    smooth_lasso_perf = 0.5 / y.size * extmath.norm(
        np.dot(X_, smooth_lasso.coef_) - y) ** 2\
        + np.sum(np.abs(smooth_lasso.coef_))
    np.testing.assert_almost_equal(smooth_lasso_perf, lasso_perf, decimal=3)


def test_params_correctly_propagated_in_constructors_biz():
    for penalty, classif, alpha, l1_ratio in itertools.product(
        ["smooth-lasso", "tvl1"], [True, False], [.4, .01], [.5, 1.]):
        cvobj = SpaceNet(
            mask="dummy", penalty=penalty, classif=classif, alpha=alpha,
            l1_ratio=l1_ratio)
        assert_equal(cvobj.alpha, alpha)
        assert_equal(cvobj.l1_ratio, l1_ratio)
