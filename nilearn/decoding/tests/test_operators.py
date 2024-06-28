import itertools

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from nilearn.decoding._proximal_operators import prox_l1, prox_tvl1


def test_prox_l1_nonexpansiveness(rng, n_features=10):
    x = rng.standard_normal((n_features, 1))
    tau = 0.3
    s = prox_l1(x.copy(), tau)
    p = x - s  # projection + shrinkage = id

    # We should have ||s(a) - s(b)||^2 <= ||a - b||^2 - ||p(a) - p(b)||^2
    # for all a and b (this is strong non-expansiveness
    for (a, b), (pa, pb), (sa, sb) in zip(
        *[itertools.product(z[0], z[0]) for z in [x, p, s]]
    ):
        assert (sa - sb) ** 2 <= (a - b) ** 2 - (pa - pb) ** 2


@pytest.mark.parametrize("ndim", range(3, 4))
@pytest.mark.parametrize("weight", np.logspace(-10, 10, num=10))
def test_prox_tvl1_approximates_prox_l1_for_lasso(
    rng, ndim, weight, size=15, decimal=4, dgap_tol=1e-7
):
    l1_ratio = 1.0  # pure LASSO

    shape = [size] * ndim
    z = rng.standard_normal(shape)

    # use prox_tvl1 approximation to prox_l1
    a = prox_tvl1(
        z.copy(),
        weight=weight,
        l1_ratio=l1_ratio,
        dgap_tol=dgap_tol,
        max_iter=10,
    )[0][-1].ravel()

    # use exact closed-form soft shrinkage formula for prox_l1
    b = prox_l1(z.copy(), weight)[-1].ravel()

    # results should be close in l-infinity norm
    assert_almost_equal(np.abs(a - b).max(), 0.0, decimal=decimal)


@pytest.mark.parametrize("verbose", [True, False])
def test_prox_tvl1_verbose(rng, verbose):
    l1_ratio = 1.0  # pure LASSO

    size = 15
    dgap_tol = 1e-7
    ndim = 3
    weight = -10

    shape = [size] * ndim
    z = rng.standard_normal(shape)

    prox_tvl1(
        z.copy(),
        weight=weight,
        l1_ratio=l1_ratio,
        dgap_tol=dgap_tol,
        max_iter=10,
        val_min=-np.inf,
        val_max=np.inf,
        verbose=verbose,
        x_tol=1e-7,
    )
