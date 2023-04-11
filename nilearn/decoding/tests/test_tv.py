import numpy as np
import pytest
from nilearn.decoding.objective_functions import _gradient_id, _squared_loss
from nilearn.decoding.space_net_solvers import (
    _tvl1_objective,
    _tvl1_objective_from_gradient,
    tvl1_solver,
)


@pytest.mark.parametrize("alpha", [0.0, 1e-1, 1e-3])
@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_tvl1_from_gradient(
    alpha, l1_ratio, size=5, n_samples=10, random_state=42
):
    rng = np.random.RandomState(random_state)
    shape = [size] * 3
    n_voxels = np.prod(shape)
    X = rng.randn(n_samples, n_voxels)
    y = rng.randn(n_samples)
    w = rng.randn(*shape)
    mask = np.ones_like(w).astype(bool)

    gradid = _gradient_id(w, l1_ratio=l1_ratio)

    assert _tvl1_objective(
        X, y, w.copy().ravel(), alpha, l1_ratio, mask
    ) == _squared_loss(
        X, y, w.copy().ravel(), compute_grad=False
    ) + alpha * _tvl1_objective_from_gradient(
        gradid
    )


def test_tvl1_objective_raises_value_error_if_invalid_loss():
    with pytest.raises(ValueError, match="mse' or 'logistic"):
        _tvl1_objective(None, None, None, None, None, None, loss="invalidloss")


def test_tvl1_solver_raises_value_error_if_invalid_loss():
    with pytest.raises(ValueError, match="mse' or 'logistic"):
        tvl1_solver(
            np.array([[1]]), None, None, None, None, loss="invalidloss"
        )
