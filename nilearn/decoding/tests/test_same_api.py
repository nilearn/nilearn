"""Make sure all models are using the same low-level API.

for computing image gradient, loss functions, etc.
"""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.datasets import load_iris

from nilearn.decoding._objective_functions import (
    logistic_loss_lipschitz_constant,
    spectral_norm_squared,
    squared_loss,
    squared_loss_grad,
)
from nilearn.decoding.space_net import (
    BaseSpaceNet,
    SpaceNetClassifier,
    SpaceNetRegressor,
)
from nilearn.decoding.space_net_solvers import (
    _logistic_derivative_lipschitz_constant,
    _squared_loss_and_spatial_grad,
    _squared_loss_and_spatial_grad_derivative,
    _squared_loss_derivative_lipschitz_constant,
    graph_net_logistic,
    graph_net_squared_loss,
    tvl1_solver,
)
from nilearn.image import get_data
from nilearn.masking import unmask_from_to_3d_array


def _make_data(rng=None, masked=False, dim=(2, 2, 2)):
    if rng is None:
        rng = np.random.default_rng(42)
    mask = np.ones(dim).astype(bool)
    mask[rng.random(dim) < 0.7] = 0
    w = np.zeros(dim)
    w[dim[0] // 2 :, dim[1] // 2 :, : dim[2] // 2] = 1
    n = 5
    X = np.ones([n, *dim])
    X += rng.standard_normal(X.shape)
    y = np.dot([x[mask] for x in X], w[mask])
    if masked:
        X = np.array([x[mask] for x in X])
        w = w[mask]
    else:
        X = np.rollaxis(X, 0, start=4)
        assert X.shape[-1] == n
    return X, y, w, mask


def to_niimgs(X, dim):
    p = np.prod(dim)

    assert len(dim) == 3
    assert X.shape[-1] <= p

    mask = np.zeros(p).astype(bool)
    mask[: X.shape[-1]] = 1

    assert mask.sum() == X.shape[1]

    mask = mask.reshape(dim)
    X = np.rollaxis(
        np.array([unmask_from_to_3d_array(x, mask) for x in X]), 0, start=4
    )
    affine = np.eye(4)

    return Nifti1Image(X, affine), Nifti1Image(mask.astype(np.float64), affine)


def test_same_energy_calculus_pure_lasso(rng):
    X, y, w, mask = _make_data(rng=rng, masked=True)

    # check funcvals
    f1 = squared_loss(X, y, w)
    f2 = _squared_loss_and_spatial_grad(X, y, w.ravel(), mask, 0.0)

    assert f1 == f2

    # check derivatives
    g1 = squared_loss_grad(X, y, w)
    g2 = _squared_loss_and_spatial_grad_derivative(X, y, w.ravel(), mask, 0.0)

    assert_array_equal(g1, g2)


def test_lipschitz_constant_loss_mse(rng):
    X, _, _, mask = _make_data(rng=rng, masked=True)
    l1_ratio = 1.0
    alpha = 0.1
    mask = np.ones(X.shape[1]).astype(bool)
    grad_weight = alpha * X.shape[0] * (1.0 - l1_ratio)

    a = _squared_loss_derivative_lipschitz_constant(X, mask, grad_weight)
    b = spectral_norm_squared(X)

    assert_almost_equal(a, b)


def test_lipschitz_constant_loss_logreg(rng):
    X, _, _, mask = _make_data(rng=rng, masked=True)
    l1_ratio = 1.0
    alpha = 0.1
    grad_weight = alpha * X.shape[0] * (1.0 - l1_ratio)

    a = _logistic_derivative_lipschitz_constant(X, mask, grad_weight)
    b = logistic_loss_lipschitz_constant(X)

    assert a == b


def test_graph_net_and_tvl1_same_for_pure_l1(max_iter=100, decimal=2):
    """Check that graph_net_solver and tvl1_solver give same results \
    when l1_ratio = 1.

    Results should be exactly the same for pure lasso
    However because of the TV-L1 prox approx, results might be 'slightly'
    different.
    """
    X, y, _, mask = _make_data(dim=(3, 3, 3))
    y = np.round(y)
    alpha = 0.01
    unmasked_X = np.rollaxis(X, -1, start=0)
    unmasked_X = np.array([x[mask] for x in unmasked_X])

    a = tvl1_solver(
        unmasked_X,
        y,
        alpha,
        l1_ratio=1.0,
        mask=mask,
        loss="mse",
        max_iter=max_iter,
        verbose=1,
    )[0]
    b = graph_net_squared_loss(
        unmasked_X,
        y,
        alpha,
        l1_ratio=1.0,
        max_iter=max_iter,
        mask=mask,
        verbose=0,
    )[0]

    assert_array_almost_equal(a, b, decimal=decimal)


@pytest.mark.parametrize("standardize", [True, False])
def test_graph_net_and_tvl1_same_for_pure_l1_base_space_net(
    affine_eye,
    standardize,
    max_iter=100,
    decimal=2,
):
    """Check that graph_net_solver and tvl1_solver give same results \
    when l1_ratio = 1.

    Results should be exactly the same for pure lasso
    However because of the TV-L1 prox approx, results might be 'slightly'
    different.
    """
    X, y, _, mask = _make_data(dim=(3, 3, 3))
    y = np.round(y)
    alpha = 0.01
    unmasked_X = np.rollaxis(X, -1, start=0)
    unmasked_X = np.array([x[mask] for x in unmasked_X])

    mask = Nifti1Image(mask.astype(np.float64), affine_eye)
    X = Nifti1Image(X.astype(np.float64), affine_eye)

    sl = BaseSpaceNet(
        alphas=alpha,
        l1_ratios=1.0,
        mask=mask,
        penalty="graph-net",
        max_iter=max_iter,
        standardize=standardize,
        verbose=0,
    ).fit(X, y)
    tvl1 = BaseSpaceNet(
        alphas=alpha,
        l1_ratios=1.0,
        mask=mask,
        penalty="tv-l1",
        max_iter=max_iter,
        standardize=standardize,
        verbose=0,
    ).fit(X, y)

    assert_array_almost_equal(sl.coef_, tvl1.coef_, decimal=decimal)


def test_graph_net_and_tvl1_same_for_pure_l1_logistic(max_iter=20, decimal=2):
    """Check graph_net_solver and tvl1_solver should give same results \
    when l1_ratio = 1.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    y = y > 0.0
    alpha = 1.0 / X.shape[0]
    _, mask_ = to_niimgs(X, (2, 2, 2))
    mask = get_data(mask_).astype(bool).ravel()

    a = graph_net_logistic(
        X, y, alpha, l1_ratio=1.0, mask=mask, max_iter=max_iter, verbose=0
    )[0]
    b = tvl1_solver(
        X,
        y,
        alpha,
        l1_ratio=1.0,
        loss="logistic",
        mask=mask,
        max_iter=max_iter,
        verbose=1,
    )[0]

    assert_array_almost_equal(a, b, decimal=decimal)


@pytest.mark.parametrize("standardize", [True, False])
def test_graph_net_and_tvl1_same_for_pure_l1_logistic_spacenet_classifier(
    standardize, max_iter=20, decimal=2
):
    """Check graph_net_solver and tvl1_solver should give same results \
    when l1_ratio = 1.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    y = y > 0.0
    alpha = 1.0 / X.shape[0]
    X_, mask_ = to_niimgs(X, (2, 2, 2))

    sl = SpaceNetClassifier(
        alphas=alpha,
        l1_ratios=1.0,
        max_iter=max_iter,
        mask=mask_,
        penalty="graph-net",
        standardize=standardize,
        verbose=0,
    ).fit(X_, y)
    tvl1 = SpaceNetClassifier(
        alphas=alpha,
        l1_ratios=1.0,
        max_iter=max_iter,
        mask=mask_,
        penalty="tv-l1",
        standardize=standardize,
        verbose=0,
    ).fit(X_, y)

    assert_array_almost_equal(sl.coef_[0], tvl1.coef_[0], decimal=decimal)


@pytest.mark.parametrize("standardize", [True, False])
def test_graph_net_and_tv_same_for_pure_l1_another_test(
    standardize, decimal=1
):
    """Check that graph_net_solver and tvl1_solver give same results \
    when l1_ratio = 1.
    """
    dim = (3, 3, 3)
    X, y, _, mask = _make_data(masked=True, dim=dim)
    X, mask = to_niimgs(X, dim)
    alpha = 0.1
    l1_ratio = 1.0
    max_iter = 20

    sl = BaseSpaceNet(
        alphas=alpha,
        l1_ratios=l1_ratio,
        penalty="graph-net",
        max_iter=max_iter,
        mask=mask,
        is_classif=False,
        standardize=standardize,
        verbose=0,
    ).fit(X, y)
    tvl1 = BaseSpaceNet(
        alphas=alpha,
        l1_ratios=l1_ratio,
        penalty="tv-l1",
        max_iter=max_iter,
        mask=mask,
        is_classif=False,
        standardize=standardize,
        verbose=0,
    ).fit(X, y)

    assert_array_almost_equal(sl.coef_, tvl1.coef_, decimal=decimal)


@pytest.mark.parametrize("penalty", ["graph-net", "tv-l1"])
@pytest.mark.parametrize("cls", [SpaceNetRegressor, SpaceNetClassifier])
def test_coef_shape(penalty, cls):
    iris = load_iris()
    X, y = iris.data, iris.target
    X, mask = to_niimgs(X, (2, 2, 2))

    model = cls(
        mask=mask, max_iter=3, penalty=penalty, alphas=1.0, verbose=0
    ).fit(X, y)

    assert model.coef_.ndim == 2
