# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

# $Id: test_tv.py 336 2010-04-21 18:07:26Z gramfort $

import os
import sys
from nose.tools import assert_equal
import numpy as np
from ..common import gradient_id, compute_mse
from ..tv import tvl1_objective, _tvl1_objective_from_gradient

fn = lambda f, x, n: f(fn(f, x, n - 1)) if n > 1 else f(x)
ROOT = fn(os.path.dirname, os.path.dirname(__file__), 4)
CACHE = os.path.join(ROOT, "cache")
sys.path.append(os.path.join(ROOT, "examples/proximal"))


def test_tv_l1_from_gradient(size=5, n_samples=10, random_state=42,
                             decimal=8):

    rng = np.random.RandomState(random_state)

    shape = [size] * 3
    n_voxels = np.prod(shape)
    X = rng.randn(n_samples, n_voxels)
    y = rng.randn(n_samples)
    w = rng.randn(*shape)

    for alpha in [0., 1e-1, 1e-3]:
        for l1_ratio in [0., .5, 1.]:
            gradid = gradient_id(w, l1_ratio=l1_ratio)
            assert_equal(tvl1_objective(
                X, y, w.copy(), alpha, l1_ratio, shape=shape),
                compute_mse(X, y, w.copy(), compute_grad=False
                            ) + alpha * _tvl1_objective_from_gradient(
                    gradid))
