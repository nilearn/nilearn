"""
Make sure all models are using thesame low-level API (
for computing image gradient, loss functins, etc.).

"""

import random
from nose.tools import nottest
import numpy as np
import nibabel
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from ..objective_functions import (squared_loss, squared_loss_grad,
                                   logistic_loss_lipschitz_constant,
                                   spectral_norm_squared, _unmask)
from ..space_net_solvers import (squared_loss_and_spatial_grad,
                                 logistic_derivative_lipschitz_constant,
                                 squared_loss_derivative_lipschitz_constant,
                                 smooth_lasso_squared_loss,
                                 smooth_lasso_logistic,
                                 squared_loss_and_spatial_grad_derivative,
                                 tvl1_solver)
from ..space_net import SpaceNet
from nose.tools import assert_equal


def _make_data(rng=None, masked=False):
    if rng is None:
        rng = check_random_state(42)
    dim = (3, 4, 5)
    mask = np.ones(dim).astype(np.bool)
    mask[rng.rand() < .7] = 0
    w = np.zeros(dim)
    w[2:, 3:, :2] = 1
    n = 10
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


def to_niimgs(X, dim, rng=None):
    if rng is None:
        rng = random.Random(42)
    p = np.prod(dim)
    assert_equal(len(dim), 3)
    assert X.shape[-1] <= p
    mask = np.zeros(p).astype(np.bool)
    mask[rng.sample(np.arange(p), X.shape[-1])] = 1
    mask = mask.reshape(dim)
    X = np.rollaxis(np.array([_unmask(x, mask) for x in X]), 0, start=4)
    affine = np.eye(4)
    return nibabel.Nifti1Image(X, affine), nibabel.Nifti1Image(
        mask.astype(np.float), affine)


def test_same_energy_calculus_pure_lasso():
    rng = check_random_state(42)
    X, y, w, mask = _make_data(rng=rng, masked=True)

    # check funcvals
    f1 = squared_loss(X, y, w)
    f2 = squared_loss_and_spatial_grad(X, y, w.ravel(), mask, 0.)
    assert_equal(f1, f2)

    # check derivatives
    g1 = squared_loss_grad(X, y, w)
    g2 = squared_loss_and_spatial_grad_derivative(X, y, w.ravel(), mask, 0.)
    np.testing.assert_array_equal(g1, g2)


def test_lipschitz_constant_lass_mse():
    rng = check_random_state(42)
    X, y, w, mask = _make_data(rng=rng, masked=True)
    l1_ratio = 1.
    alpha = .1
    mask = np.ones(X.shape[1]).astype(np.bool)
    grad_weight = alpha * X.shape[0] * (1. - l1_ratio)
    a = squared_loss_derivative_lipschitz_constant(X, mask, grad_weight)
    b = spectral_norm_squared(X)
    np.testing.assert_almost_equal(a, b)


def test_lipschitz_constant_lass_logreg():
    rng = check_random_state(42)
    X, y, w, mask = _make_data(rng=rng, masked=True)
    l1_ratio = 1.
    alpha = .1
    grad_weight = alpha * X.shape[0] * (1. - l1_ratio)
    a = logistic_derivative_lipschitz_constant(X, mask, grad_weight)
    b = logistic_loss_lipschitz_constant(X)
    assert_equal(a, b)


def test_smoothlasso_and_tvl1_same_for_pure_l1(max_iter=20, decimal=2):
    ###############################################################
    # smoothlasso_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    X, y, _, mask = _make_data()
    alpha = .1
    unmasked_X = np.rollaxis(X, -1, start=0)
    unmasked_X = np.array([x[mask] for x in unmasked_X])

    # results should be exactly the same for pure lasso
    a = tvl1_solver(unmasked_X, y, alpha, 1., mask, loss="mse",
                    max_iter=max_iter)[0]
    b = smooth_lasso_squared_loss(unmasked_X, y, alpha, 1.,
                                  max_iter=max_iter,
                                  mask=mask)[0]

    mask = nibabel.Nifti1Image(mask.astype(np.float), np.eye(4))
    X = nibabel.Nifti1Image(X.astype(np.float), np.eye(4))
    sl = SpaceNet(
        alpha=alpha, l1_ratio=1., mask=mask, penalty="smooth-lasso",
        max_iter=max_iter).fit(X, y)
    tvl1 = SpaceNet(alpha=alpha, l1_ratio=1., mask=mask, penalty="tv-l1",
                    max_iter=max_iter).fit(X, y)

    # Should be exactly the same (except for numerical errors).
    # However because of the TV-l1 prox approx, results might be 'slightly'
    # different.
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
    X_, mask_ = to_niimgs(X, (2, 2, 2))
    mask = mask_.get_data().astype(np.bool).ravel()
    max_iter = 10

    # results should be exactly the same for pure lasso
    a = smooth_lasso_logistic(X, y, alpha, 1., mask=mask,
                              max_iter=max_iter)[0]
    b = tvl1_solver(X, y, alpha, 1., loss="logistic", mask=mask,
                    max_iter=max_iter)[0]
    sl = SpaceNet(alpha=alpha, l1_ratio=1., verbose=0, is_classif=True,
                  max_iter=max_iter, mask=mask_, penalty="smooth-lasso",
                  ).fit(X_, y)
    tvl1 = SpaceNet(alpha=alpha, l1_ratio=1., verbose=0, is_classif=True,
                  max_iter=max_iter, mask=mask_, penalty="tv-l1",
                  ).fit(X_, y)

    # should be exactly the same (except for numerical errors)
    np.testing.assert_array_almost_equal(a, b, decimal=decimal)
    np.testing.assert_array_almost_equal(sl.coef_[0], tvl1.coef_[0],
                                         decimal=decimal)


def test_logreg_with_mask_issue_10():
    rng = check_random_state(42)
    shape = (3, 4, 5)
    n_samples = 10
    mask = np.zeros(shape)
    mask[:2, 3:, 3:4] = 1
    X = rng.randn(n_samples, mask.sum())
    X, mask = to_niimgs(X, shape)
    y = np.sign(rng.randn(n_samples))
    alpha = 1.
    l1_ratio = .5

    for penalty in ["smooth-lasso", "tv-l1"]:
        for is_classif in [True, False]:
            # ensure that our fix didn't break anythx else
            SpaceNet(penalty=penalty, is_classif=is_classif, alpha=alpha,
                     l1_ratio=l1_ratio, mask=mask).fit(X, y)


def test_smoothlasso_and_tv_same_for_pure_l1_another_test(decimal=2):
    ###############################################################
    # smoothlasso_solver and tvl1_solver should give same results
    # when l1_ratio = 1.
    ###############################################################

    X, y, _, mask = _make_data(masked=True)
    X, mask = to_niimgs(X, (4, 5, 6))
    alpha = .1
    l1_ratio = 1.
    max_iter = 20

    sl = SpaceNet(alpha=alpha, l1_ratio=l1_ratio, penalty="smooth-lasso",
                  max_iter=max_iter, mask=mask, is_classif=False,
                  verbose=0).fit(X, y)
    tvl1 = SpaceNet(alpha=alpha, l1_ratio=l1_ratio, penalty="tv-l1",
                  max_iter=max_iter, mask=mask, is_classif=False,
                  verbose=0).fit(X, y)

    # should be exactly the same (except for numerical errors)
    np.testing.assert_array_almost_equal(sl.coef_, tvl1.coef_, decimal=decimal)


def test_coef_shape():
    iris = load_iris()
    X, y = iris.data, iris.target
    X, mask = to_niimgs(X, (2, 2, 2))
    for penalty in ["smooth-lasso", "tv-l1"]:
        cv = SpaceNet(
            mask=mask, max_iter=3, penalty=penalty, is_classif=False).fit(X, y)
        assert_equal(cv.coef_.ndim, 1)

    for penalty in ["smooth-lasso", "tv-l1"]:
        cv = SpaceNet(mask=mask,
                      max_iter=3, penalty=penalty, is_classif=True).fit(X, y)
        assert_equal(cv.coef_.ndim, 2)


@nottest
def test_w_shapes():
    """Test that solvers handle w of same shape (during callbacks, etc.)."""
    pass
