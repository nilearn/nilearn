from nose.tools import assert_equal, assert_raises
import numpy as np
from nilearn.decoding.objective_functions import _gradient_id, _squared_loss
from nilearn.decoding.space_net_solvers import (
    _tvl1_objective, _tvl1_objective_from_gradient, tvl1_solver)


def test_tvl1_from_gradient(size=5, n_samples=10, random_state=42):
    rng = np.random.RandomState(random_state)
    shape = [size] * 3
    n_voxels = np.prod(shape)
    X = rng.randn(n_samples, n_voxels)
    y = rng.randn(n_samples)
    w = rng.randn(*shape)
    mask = np.ones_like(w).astype(np.bool)
    for alpha in [0., 1e-1, 1e-3]:
        for l1_ratio in [0., .5, 1.]:
            gradid = _gradient_id(w, l1_ratio=l1_ratio)
            assert_equal(_tvl1_objective(
                X, y, w.copy().ravel(), alpha, l1_ratio, mask),
                _squared_loss(X, y, w.copy().ravel(),
                              compute_grad=False
                              ) + alpha * _tvl1_objective_from_gradient(
                    gradid))


def test_tvl1_objective_raises_value_error_if_invalid_loss():
    assert_raises(ValueError, lambda loss: _tvl1_objective(
        None, None, None, None, None, None, loss=loss),
        "invalidloss")


def test_tvl1_solver_raises_value_error_if_invalid_loss():
    assert_raises(ValueError, lambda loss: tvl1_solver(
        np.array([[1]]), None, None, None, None, loss=loss),
        "invalidloss")
