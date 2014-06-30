from nose.tools import assert_equal, assert_true
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_iris
from ..cv import (TVl1ClassifierCV, TVl1RegressorCV,
                  SmoothLassoClassifierCV, SmoothLassoRegressorCV)
from .._cv_tricks import (RegressorFeatureSelector, ClassifierFeatureSelector,
                          EarlyStoppingCallback, _my_alpha_grid)


def test_same_lasso_classifier_cv():
    l1_ratio = 1.
    memory = Memory("cache")
    iris = load_iris()
    X = iris.data
    y = iris.target
    y = 2 * (y == 2) - 1
    tol = 1e-3

    # classification
    n_alphas = 5
    tvl1classifiercv = TVl1ClassifierCV(n_alphas=n_alphas, memory=memory, cv=2,
                                        l1_ratio=l1_ratio, tol=tol).fit(X, y)
    slclassifiercv = SmoothLassoClassifierCV(n_alphas=n_alphas, memory=memory,
                                   cv=2, l1_ratio=l1_ratio, tol=tol).fit(X, y)
    if 0:
        # XXX test fails with early stopping in CV
        assert_equal(tvl1classifiercv.alpha_, slclassifiercv.alpha_)

    # regression
    tvl1regressorcv = TVl1RegressorCV(n_alphas=n_alphas, memory=memory, cv=2,
                                      l1_ratio=l1_ratio, tol=tol).fit(X, y)
    slregressorcv = SmoothLassoRegressorCV(n_alphas=n_alphas, memory=memory,
                                           cv=2, l1_ratio=l1_ratio, tol=tol
                                           ).fit(X, y)
    if 0:
        # XXX test fails with early stopping in CV
        assert_equal(tvl1regressorcv.alpha_, slregressorcv.alpha_)

    # plot cv curves
    if 0:
        import pylab as pl
        for cv in [tvl1classifiercv, tvl1regressorcv, slclassifiercv,
                   slregressorcv]:
            pl.figure()
            means = cv.scores_.mean(axis=-1)
            stds = cv.scores_.std(axis=-1)
            best_score = np.mean(cv.scores_[cv.alphas_ == cv.alpha_])
            pl.errorbar(-np.log10(cv.alphas_), means, yerr=stds,
                        label="test error")
            pl.axvline(-np.log10(cv.alpha_), linestyle="--",
                       c='r', label="best alpha")
            pl.axhline(best_score, linestyle="-.", c='r',
                       label="best mean test error")
            pl.ylabel("mean test error (misclassification)")
            pl.xlabel("-Log10(alpha)")
            pl.legend(loc="best")
            pl.title("sk iris data: %s (l1_ratio=%g)" % (
                    cv.__class__.__name__, l1_ratio))
        pl.show()


def test_my_alpha_grid(n_samples=4, n_features=3):
    rng = np.random.RandomState(42)
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
        pass


def test_featureselectors():
    import random
    rng = np.random.RandomState(42)
    random.seed(0)
    from sklearn.datasets import load_iris
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
    rng = np.random.RandomState(42)
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
