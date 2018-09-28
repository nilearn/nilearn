"""
Make sure all models are using thesame low-level API (
for computing image gradient, loss functins, etc.).

"""

from nose.tools import nottest, assert_equal, assert_true
import numpy as np
import nibabel
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from nilearn.decoding.objective_functions import (
    _squared_loss, _squared_loss_grad, _logistic_loss_lipschitz_constant,
    spectral_norm_squared, _unmask)
from nilearn.decoding.space_net_solvers import (
    _squared_loss_and_spatial_grad,
    _logistic_derivative_lipschitz_constant,
    _squared_loss_derivative_lipschitz_constant,
    _graph_net_squared_loss,
    _graph_net_logistic,
    _squared_loss_and_spatial_grad_derivative,
    tvl1_solver)
from nilearn.decoding.space_net import (BaseSpaceNet, SpaceNetClassifier,
                                        SpaceNetRegressor)


def _make_data(rng=None, masked=False, dim=(2, 2, 2)):
    if rng is None:
        rng = check_random_state(42)
    mask = np.ones(dim).astype(np.bool)
    mask[rng.rand(*dim) < .7] = 0
    w = np.zeros(dim)
    w[dim[0] // 2:, dim[1] // 2:, :dim[2] // 2] = 1
    n = 5
    X = np.ones([n] + list(dim))
    X += rng.randn(*X.shape)
    y = np.dot([x[mask] for x in X], w[mask])
    if masked:
        X = np.array([x[mask] for x in X])
        w = w[mask]
    else:
        X = np.rollaxis(X, 0, start=4)
        assert_equal(X.shape[-1], n)
    return X, y, w, mask


def to_niimgs(X, dim):
    p = np.prod(dim)
    assert_equal(len(dim), 3)
    assert_true(X.shape[-1] <= p)
    mask = np.zeros(p).astype(np.bool)
    mask[:X.shape[-1]] = 1
    assert_equal(mask.sum(), X.shape[1])
    mask = mask.reshape(dim)
    X = np.rollaxis(np.array([_unmask(x, mask) for x in X]), 0, start=4)
    affine = np.eye(4)
    return nibabel.Nifti1Image(X, affine), nibabel.Nifti1Image(
        mask.astype(np.float), affine)


def test_same_energy_calculus_pure_lasso():
    rng = check_random_state(42)
    X, y, w, mask = _make_data(rng=rng, masked=True)

    # check funcvals
    f1 = _squared_loss(X, y, w)
    f2 = _squared_loss_and_spatial_grad(X, y, w.ravel(), mask, 0.)
    assert_equal(f1, f2)

    # check derivatives
    g1 = _squared_loss_grad(X, y, w)
    g2 = _squared_loss_and_spatial_grad_derivative(X, y, w.ravel(), mask, 0.)
    np.testing.assert_array_equal(g1, g2)


def test_lipschitz_constant_loss_mse():
    rng = check_random_state(42)
    X, _, w, mask = _make_data(rng=rng, masked=True)
    l1_ratio = 1.
    alpha = .1
    mask = np.ones(X.shape[1]).astype(np.bool)
    grad_weight = alpha * X.shape[0] * (1. - l1_ratio)
    a = _squared_loss_derivative_lipschitz_constant(X, mask, grad_weight)
    b = spectral_norm_squared(X)
    np.testing.assert_almost_equal(a, b)


def test_lipschitz_constant_loss_logreg():
    rng = check_random_state(42)
    X, _, w, mask = _make_data(rng=rng, masked=True)
    l1_ratio = 1.
    alpha = .1
    grad_weight = alpha * X.shape[0] * (1. - l1_ratio)
    a = _logistic_derivative_lipschitz_constant(X, mask, grad_weight)
    b = _logistic_loss_lipschitz_constant(X)
    assert_equal(a, b)


def test_graph_net_and_tvl1_same_for_pure_l1(max_iter=100, decimal=2):
    ###############################################################
    # graph_net_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    X, y, _, mask = _make_data()
    alpha = .1
    unmasked_X = np.rollaxis(X, -1, start=0)
    unmasked_X = np.array([x[mask] for x in unmasked_X])

    # results should be exactly the same for pure lasso
    a = tvl1_solver(unmasked_X, y, alpha, 1., mask, loss="mse",
                    max_iter=max_iter)[0]
    b = _graph_net_squared_loss(unmasked_X, y, alpha, 1.,
                                max_iter=max_iter,
                                mask=mask)[0]

    mask = nibabel.Nifti1Image(mask.astype(np.float), np.eye(4))
    X = nibabel.Nifti1Image(X.astype(np.float), np.eye(4))
    for standardize in [True, False]:
        sl = BaseSpaceNet(
            alphas=alpha, l1_ratios=1., mask=mask, penalty="graph-net",
            max_iter=max_iter, standardize=standardize).fit(X, y)
        tvl1 = BaseSpaceNet(
            alphas=alpha, l1_ratios=1., mask=mask, penalty="tv-l1",
            max_iter=max_iter, standardize=standardize).fit(X, y)

        # Should be exactly the same (except for numerical errors).
        # However because of the TV-L1 prox approx, results might be 'slightly'
        # different.
        np.testing.assert_array_almost_equal(a, b, decimal=decimal)
        np.testing.assert_array_almost_equal(sl.coef_, tvl1.coef_,
                                             decimal=decimal)


def test_graph_net_and_tvl1_same_for_pure_l1_logistic(max_iter=20,
                                                      decimal=2):
    ###############################################################
    # graph_net_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    iris = load_iris()
    X, y = iris.data, iris.target
    y = y > 0.
    alpha = 1. / X.shape[0]
    X_, mask_ = to_niimgs(X, (2, 2, 2))
    mask = mask_.get_data().astype(np.bool).ravel()

    # results should be exactly the same for pure lasso
    a = _graph_net_logistic(X, y, alpha, 1., mask=mask,
                            max_iter=max_iter)[0]
    b = tvl1_solver(X, y, alpha, 1., loss="logistic", mask=mask,
                    max_iter=max_iter)[0]
    for standardize in [True, False]:
        sl = SpaceNetClassifier(
            alphas=alpha,
            l1_ratios=1.,
            max_iter=max_iter,
            mask=mask_,
            penalty="graph-net",
            standardize=standardize).fit(
            X_,
            y)
        tvl1 = SpaceNetClassifier(
            alphas=alpha,
            l1_ratios=1.,
            max_iter=max_iter,
            mask=mask_,
            penalty="tv-l1",
            standardize=standardize).fit(
            X_,
            y)

    # should be exactly the same (except for numerical errors)
    np.testing.assert_array_almost_equal(a, b, decimal=decimal)
    np.testing.assert_array_almost_equal(sl.coef_[0], tvl1.coef_[0],
                                         decimal=decimal)


def test_graph_net_and_tv_same_for_pure_l1_another_test(decimal=1):
    ###############################################################
    # graph_net_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    dim = (3, 3, 3)
    X, y, _, mask = _make_data(masked=True, dim=dim)
    X, mask = to_niimgs(X, dim)
    alpha = .1
    l1_ratio = 1.
    max_iter = 20

    for standardize in [True, False]:
        sl = BaseSpaceNet(alphas=alpha, l1_ratios=l1_ratio,
                          penalty="graph-net", max_iter=max_iter,
                          mask=mask, is_classif=False,
                          standardize=standardize, verbose=0).fit(X, y)
        tvl1 = BaseSpaceNet(alphas=alpha, l1_ratios=l1_ratio, penalty="tv-l1",
                            max_iter=max_iter, mask=mask, is_classif=False,
                            standardize=standardize, verbose=0).fit(X, y)

    # should be exactly the same (except for numerical errors)
    np.testing.assert_array_almost_equal(sl.coef_, tvl1.coef_, decimal=decimal)


def test_coef_shape():
    iris = load_iris()
    X, y = iris.data, iris.target
    X, mask = to_niimgs(X, (2, 2, 2))
    for penalty in ["graph-net", "tv-l1"]:
        for cls in [SpaceNetRegressor, SpaceNetClassifier]:
            model = cls(
                mask=mask, max_iter=3, penalty=penalty, alphas=1.).fit(X, y)
            assert_equal(model.coef_.ndim, 2)


@nottest
def test_w_shapes():
    """Test that solvers handle w of same shape (during callbacks, etc.)."""
    pass
