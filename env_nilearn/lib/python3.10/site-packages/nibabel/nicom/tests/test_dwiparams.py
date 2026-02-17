"""Testing diffusion parameter processing"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal as np_assert_equal

from ..dwiparams import B2q, q2bg


def test_b2q():
    # conversion of b matrix to q
    q = np.array([1, 2, 3])
    s = np.sqrt(np.sum(q * q))  # vector norm
    B = np.outer(q, q)
    assert_array_almost_equal(q * s, B2q(B))
    q = np.array([1, 2, 3])
    # check that the sign of the vector as positive x convention
    B = np.outer(-q, -q)
    assert_array_almost_equal(q * s, B2q(B))
    q = np.array([-1, 2, 3])
    B = np.outer(q, q)
    assert_array_almost_equal(-q * s, B2q(B))
    # Massive negative eigs
    B = np.eye(3) * -1
    with pytest.raises(ValueError):
        B2q(B)
    # no error if we up the tolerance
    q = B2q(B, tol=1)
    # Less massive negativity, dropping tol
    B = np.diag([-1e-14, 10.0, 1])
    with pytest.raises(ValueError):
        B2q(B)
    assert_array_almost_equal(B2q(B, tol=5e-13), [0, 10, 0])
    # Confirm that we assume symmetric
    B = np.eye(3)
    B[0, 1] = 1e-5
    with pytest.raises(ValueError):
        B2q(B)


def test_q2bg():
    # Conversion of q vector to b value and unit vector
    for pos in range(3):
        q_vec = np.zeros((3,))
        q_vec[pos] = 10.0
        np_assert_equal(q2bg(q_vec), (10, q_vec / 10.0))
    # Also - check array-like
    q_vec = [0, 1e-6, 0]
    np_assert_equal(q2bg(q_vec), (0, 0))
    q_vec = [0, 1e-4, 0]
    b, g = q2bg(q_vec)
    assert_array_almost_equal(b, 1e-4)
    assert_array_almost_equal(g, [0, 1, 0])
    np_assert_equal(q2bg(q_vec, tol=5e-4), (0, 0))
