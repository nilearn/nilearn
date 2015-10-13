import os
import warnings
import itertools
from functools import partial
from nose import SkipTest
from nose.tools import (assert_equal, assert_true, assert_false,
                        assert_raises)
import numpy as np
import nibabel
from sklearn.datasets import load_iris
from sklearn.utils import extmath
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from nilearn.decoding.space_net import (
    _EarlyStoppingCallback, _space_net_alpha_grid, MNI152_BRAIN_VOLUME,
    path_scores, BaseSpaceNet, _crop_mask, _univariate_feature_screening,
    _get_mask_volume, SpaceNetClassifier, SpaceNetRegressor)
from nilearn.decoding.space_net_solvers import (_graph_net_logistic,
                                                _graph_net_squared_loss)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")
logistic_path_scores = partial(path_scores, is_classif=True)
squared_loss_path_scores = partial(path_scores, is_classif=False)

# Data used in almost all tests
from .test_same_api import to_niimgs
size = 4
from .simulate_graph_net_data import create_graph_net_simulation_data
X_, y, w, mask = create_graph_net_simulation_data(
    snr=1., n_samples=10, size=size, n_points=5, random_state=42)
X, mask = to_niimgs(X_, [size] * 3)


def test_space_net_alpha_grid(n_samples=4, n_features=3):
    rng = check_random_state(42)
    X = rng.randn(n_samples, n_features)
    y = np.arange(n_samples)

    for l1_ratio, is_classif in itertools.product([.5, 1.], [True, False]):
        alpha_max = np.max(np.abs(np.dot(X.T, y))) / l1_ratio
        np.testing.assert_almost_equal(_space_net_alpha_grid(
            X, y, n_alphas=1, l1_ratio=l1_ratio,
            logistic=is_classif), alpha_max)

    for l1_ratio, is_classif in itertools.product([.5, 1.], [True, False]):
        alpha_max = np.max(np.abs(np.dot(X.T, y))) / l1_ratio
        for n_alphas in range(1, 10):
            alphas = _space_net_alpha_grid(
                X, y, n_alphas=n_alphas, l1_ratio=l1_ratio,
                logistic=is_classif)
            np.testing.assert_almost_equal(alphas.max(), alpha_max)
            np.testing.assert_almost_equal(n_alphas, len(alphas))


def test_space_net_alpha_grid_same_as_sk():
    try:
        from sklearn.linear_model.coordinate_descent import _alpha_grid
        iris = load_iris()
        X = iris.data
        y = iris.target
        np.testing.assert_almost_equal(_space_net_alpha_grid(
            X, y, n_alphas=5), X.shape[0] * _alpha_grid(X, y, n_alphas=5,
                                                        fit_intercept=False))
    except ImportError:
        raise SkipTest


def test_early_stopping_callback_object(n_samples=10, n_features=30):
    # This test evolves w so that every line of th _EarlyStoppingCallback
    # code is executed a some point. This a kind of code fuzzing.
    rng = check_random_state(42)
    X_test = rng.randn(n_samples, n_features)
    y_test = np.dot(X_test, np.ones(n_features))
    w = np.zeros(n_features)
    escb = _EarlyStoppingCallback(X_test, y_test, False)
    for counter in range(50):
        k = min(counter, n_features - 1)
        w[k] = 1

        # jitter
        if k > 0 and rng.rand() > .9:
            w[k - 1] = 1 - w[k - 1]

        escb(dict(w=w, counter=counter))
        assert_equal(len(escb.test_scores), counter + 1)

        # restart
        if counter > 20:
            w *= 0.


def test_params_correctly_propagated_in_constructors():
    for (penalty, is_classif, n_alphas, l1_ratio, n_jobs,
         cv, perc) in itertools.product(["graph-net", "tv-l1"],
                                        [True, False], [.1, .01],
                                        [.5, 1.], [1, -1], [2, 3],
                                        [5, 10]):
        cvobj = BaseSpaceNet(
            mask="dummy", n_alphas=n_alphas, n_jobs=n_jobs, l1_ratios=l1_ratio,
            cv=cv, screening_percentile=perc, penalty=penalty,
            is_classif=is_classif)
        assert_equal(cvobj.n_alphas, n_alphas)
        assert_equal(cvobj.l1_ratios, l1_ratio)
        assert_equal(cvobj.n_jobs, n_jobs)
        assert_equal(cvobj.cv, cv)
        assert_equal(cvobj.screening_percentile, perc)


def testlogistic_path_scores():
    iris = load_iris()
    X, y = iris.data, iris.target
    _, mask = to_niimgs(X, [2, 2, 2])
    mask = mask.get_data().astype(np.bool)
    alphas = [1., .1, .01]
    test_scores, best_w = logistic_path_scores(
        _graph_net_logistic, X, y, mask, alphas, .5,
        np.arange(len(X)), np.arange(len(X)), {})[:2]
    test_scores = test_scores[0]
    assert_equal(len(test_scores), len(alphas))
    assert_equal(X.shape[1] + 1, len(best_w))


def test_squared_loss_path_scores():
    iris = load_iris()
    X, y = iris.data, iris.target
    _, mask = to_niimgs(X, [2, 2, 2])
    mask = mask.get_data().astype(np.bool)
    alphas = [1., .1, .01]
    test_scores, best_w = squared_loss_path_scores(
        _graph_net_squared_loss, X, y, mask, alphas, .5,
        np.arange(len(X)), np.arange(len(X)), {})[:2]
    test_scores = test_scores[0]
    assert_equal(len(test_scores), len(alphas))
    assert_equal(X.shape[1] + 1, len(best_w))


def test_tv_regression_simple():
    rng = check_random_state(42)
    dim = (4, 4, 4)
    W_init = np.zeros(dim)
    W_init[2:3, 1:2, -2:] = 1
    n = 10
    p = np.prod(dim)
    X = np.ones((n, 1)) + W_init.ravel().T
    X += rng.randn(n, p)
    y = np.dot(X, W_init.ravel())
    X, mask = to_niimgs(X, dim)
    print("%s %s" % (X.shape, mask.get_data().sum()))
    alphas = [.1, 1.]

    for l1_ratio in [1.]:
        for debias in [True]:
            BaseSpaceNet(mask=mask, alphas=alphas, l1_ratios=l1_ratio,
                         penalty="tv-l1", is_classif=False, max_iter=10,
                         debias=debias).fit(X, y)


def test_tv_regression_3D_image_doesnt_crash():
    rng = check_random_state(42)
    dim = (3, 4, 5)
    W_init = np.zeros(dim)
    W_init[2:3, 3:, 1:3] = 1

    n = 10
    p = dim[0] * dim[1] * dim[2]
    X = np.ones((n, 1)) + W_init.ravel().T
    X += rng.randn(n, p)
    y = np.dot(X, W_init.ravel())
    alpha = 1.
    X, mask = to_niimgs(X, dim)

    for l1_ratio in [0., .5, 1.]:
        BaseSpaceNet(mask=mask, alphas=alpha, l1_ratios=l1_ratio,
                     penalty="tv-l1", is_classif=False, max_iter=10).fit(X, y)


def test_log_reg_vs_graph_net_two_classes_iris(C=.01, tol=1e-10,
                                               zero_thr=1e-4):
    # Test for one of the extreme cases of Graph-Net: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Graph-Net's coefficients'
    # performance with the coefficients obtained from Scikit-Learn's
    # LogisticRegression, with L1 penalty, in a 2 classes classification task
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))
    tvl1 = SpaceNetClassifier(
        mask=mask, alphas=1. / C / X.shape[0], l1_ratios=1., tol=tol,
        verbose=0, max_iter=1000, penalty="tv-l1", standardize=False,
        screening_percentile=100.).fit(X_, y)
    sklogreg = LogisticRegression(penalty="l1", fit_intercept=True,
                                  tol=tol, C=C).fit(X, y)

    # compare supports
    np.testing.assert_array_equal((np.abs(tvl1.coef_) < zero_thr),
                                  (np.abs(sklogreg.coef_) < zero_thr))

    # compare predictions
    np.testing.assert_array_equal(tvl1.predict(X_), sklogreg.predict(X))


def test_lasso_vs_graph_net():
    # Test for one of the extreme cases of Graph-Net: That is, with
    # l1_ratio = 1 (pure Lasso), we compare Graph-Net's performance with
    # Scikit-Learn lasso
    lasso = Lasso(max_iter=100, tol=1e-8, normalize=False)
    graph_net = BaseSpaceNet(mask=mask, alphas=1. * X_.shape[0],
                             l1_ratios=1, is_classif=False,
                             penalty="graph-net", max_iter=100)
    lasso.fit(X_, y)
    graph_net.fit(X, y)
    lasso_perf = 0.5 / y.size * extmath.norm(np.dot(
        X_, lasso.coef_) - y) ** 2 + np.sum(np.abs(lasso.coef_))
    graph_net_perf = 0.5 * ((graph_net.predict(X) - y) ** 2).mean()
    np.testing.assert_almost_equal(graph_net_perf, lasso_perf, decimal=3)


def test_params_correctly_propagated_in_constructors_biz():
    for penalty, is_classif, alpha, l1_ratio in itertools.product(
            ["graph-net", "tv-l1"], [True, False], [.4, .01], [.5, 1.]):
        cvobj = BaseSpaceNet(
            mask="dummy", penalty=penalty, is_classif=is_classif, alphas=alpha,
            l1_ratios=l1_ratio)
        assert_equal(cvobj.alphas, alpha)
        assert_equal(cvobj.l1_ratios, l1_ratio)


def test_crop_mask():
    rng = np.random.RandomState(42)
    mask = np.zeros((3, 4, 5), dtype=np.bool)
    box = mask[:2, :3, :4]
    box[rng.rand(*box.shape) < 3.] = 1  # mask covers 30% of brain
    idx = np.where(mask)
    assert_true(idx[1].max() < 3)
    tight_mask = _crop_mask(mask)
    assert_equal(mask.sum(), tight_mask.sum())
    assert_true(np.prod(tight_mask.shape) <= np.prod(box.shape))


def test_univariate_feature_screening(dim=(11, 12, 13), n_samples=10):
    rng = np.random.RandomState(42)
    mask = rng.rand(*dim) > 100. / np.prod(dim)
    assert_true(mask.sum() >= 100.)
    mask[dim[0] // 2, dim[1] // 3:, -dim[2] // 2:] = 1  # put spatial structure
    n_features = mask.sum()
    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_features)
    w[rng.rand(n_features) > .8] = 0.
    y = X.dot(w)
    for is_classif in [True, False]:
        X_, mask_, support_ = _univariate_feature_screening(
            X, y, mask, is_classif, 20.)
        n_features_ = support_.sum()
        assert_equal(X_.shape[1], n_features_)
        assert_equal(mask_.sum(), n_features_)
        assert_true(n_features_ <= n_features)


def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (
            mni152_brain_mask))


def test_space_net_classifier_subclass():
    for penalty, alpha, l1_ratio, verbose in itertools.product(
            ["graph-net", "tv-l1"], [.4, .01], [.5, 1.], [True, False]):
        cvobj = SpaceNetClassifier(
            mask="dummy", penalty=penalty, alphas=alpha, l1_ratios=l1_ratio,
            verbose=verbose)
        assert_equal(cvobj.alphas, alpha)
        assert_equal(cvobj.l1_ratios, l1_ratio)


def test_space_net_regressor_subclass():
    for penalty, alpha, l1_ratio, verbose in itertools.product(
            ["graph-net", "tv-l1"], [.4, .01], [.5, 1.], [True, False]):
        cvobj = SpaceNetRegressor(
            mask="dummy", penalty=penalty, alphas=alpha, l1_ratios=l1_ratio,
            verbose=verbose)
        assert_equal(cvobj.alphas, alpha)
        assert_equal(cvobj.l1_ratios, l1_ratio)


def test_space_net_alpha_grid_pure_spatial():
    rng = check_random_state(42)
    X = rng.randn(10, 100)
    y = np.arange(X.shape[0])
    for is_classif in [True, False]:
        assert_false(np.any(np.isnan(_space_net_alpha_grid(
            X, y, l1_ratio=0., logistic=is_classif))))


def test_string_params_case():
    # penalty
    assert_raises(ValueError, BaseSpaceNet, penalty='TV-L1')
    assert_raises(ValueError, BaseSpaceNet, penalty='Graph-Net')

    # loss
    assert_raises(ValueError, SpaceNetClassifier, loss="MSE")
    assert_raises(ValueError, SpaceNetClassifier, loss="Logistic")
