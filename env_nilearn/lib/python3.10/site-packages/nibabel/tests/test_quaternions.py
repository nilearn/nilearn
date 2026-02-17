# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test quaternion calculations"""

import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal

from .. import eulerangles as nea
from .. import quaternions as nq


def norm(vec):
    # Return unit vector with same orientation as input vector
    return vec / np.sqrt(vec @ vec)


def gen_vec(dtype):
    # Generate random 3-vector in [-1, 1]^3
    rand = np.random.default_rng()
    return rand.uniform(low=-1.0, high=1.0, size=(3,)).astype(dtype)


# Example rotations
eg_rots = [
    nea.euler2mat(z, y, x)
    for z in np.arange(-pi, pi, pi / 2)
    for y in np.arange(-pi, pi, pi / 2)
    for x in np.arange(-pi, pi, pi / 2)
]

# Example quaternions (from rotations)
eg_quats = [nq.mat2quat(M) for M in eg_rots]
# M, quaternion pairs
eg_pairs = list(zip(eg_rots, eg_quats))

# Set of arbitrary unit quaternions
unit_quats = set(
    tuple(norm(np.r_[w, x, y, z]))
    for w in range(-2, 3)
    for x in range(-2, 3)
    for y in range(-2, 3)
    for z in range(-2, 3)
    if (w, x, y, z) != (0, 0, 0, 0)
)


def test_fillpos():
    # Takes np array
    xyz = np.zeros((3,))
    w, x, y, z = nq.fillpositive(xyz)
    assert w == 1
    # Or lists
    xyz = [0] * 3
    w, x, y, z = nq.fillpositive(xyz)
    assert w == 1
    # Errors with wrong number of values
    with pytest.raises(ValueError):
        nq.fillpositive([0, 0])
    with pytest.raises(ValueError):
        nq.fillpositive([0] * 4)
    # Errors with negative w2
    with pytest.raises(ValueError):
        nq.fillpositive([1.0] * 3)
    # Test corner case where w is near zero
    wxyz = nq.fillpositive([1, 0, 0])
    assert wxyz[0] == 0.0


@pytest.mark.parametrize('dtype', ('f4', 'f8'))
def test_fillpositive_plus_minus_epsilon(dtype):
    # Deterministic test for fillpositive threshold
    # We are trying to fill (x, y, z) with a w such that |(w, x, y, z)| == 1
    # If |(x, y, z)| is slightly off one, w should still be 0
    nptype = np.dtype(dtype).type

    # Obviously, |(x, y, z)| == 1
    baseline = np.array([0, 0, 1], dtype=dtype)

    # Obviously, |(x, y, z)| ~ 1
    plus = baseline * nptype(1 + np.finfo(dtype).eps)
    minus = baseline * nptype(1 - np.finfo(dtype).eps)

    assert nq.fillpositive(plus)[0] == 0.0
    assert nq.fillpositive(minus)[0] == 0.0

    # |(x, y, z)| > 1, no real solutions
    plus = baseline * nptype(1 + 2 * np.finfo(dtype).eps)
    with pytest.raises(ValueError):
        nq.fillpositive(plus)

    # |(x, y, z)| < 1, two real solutions, we choose positive
    minus = baseline * nptype(1 - 2 * np.finfo(dtype).eps)
    assert nq.fillpositive(minus)[0] > 0.0


@pytest.mark.parametrize('dtype', ('f4', 'f8'))
def test_fillpositive_simulated_error(dtype):
    # Nondeterministic test for fillpositive threshold
    # Create random vectors, normalize to unit length, and count on floating point
    # error to result in magnitudes larger/smaller than one
    # This is to simulate cases where a unit quaternion with w == 0 would be encoded
    # as xyz with small error, and we want to recover the w of 0

    # Permit 1 epsilon per value (default, but make explicit here)
    w2_thresh = 3 * np.finfo(dtype).eps

    for _ in range(50):
        xyz = norm(gen_vec(dtype))

        assert nq.fillpositive(xyz, w2_thresh)[0] == 0.0


def test_conjugate():
    # Takes sequence
    cq = nq.conjugate((1, 0, 0, 0))
    # Returns float type
    assert cq.dtype.kind == 'f'


def test_quat2mat():
    # also tested in roundtrip case below
    M = nq.quat2mat([1, 0, 0, 0])
    assert_array_almost_equal, M, np.eye(3)
    M = nq.quat2mat([3, 0, 0, 0])
    assert_array_almost_equal, M, np.eye(3)
    M = nq.quat2mat([0, 1, 0, 0])
    assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = nq.quat2mat([0, 2, 0, 0])
    assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = nq.quat2mat([0, 0, 0, 0])
    assert_array_almost_equal, M, np.eye(3)


def test_inverse_0():
    # Takes sequence
    iq = nq.inverse((1, 0, 0, 0))
    # Returns float type
    assert iq.dtype.kind == 'f'


@pytest.mark.parametrize(('M', 'q'), eg_pairs)
def test_inverse_1(M, q):
    iq = nq.inverse(q)
    iqM = nq.quat2mat(iq)
    iM = np.linalg.inv(M)
    assert np.allclose(iM, iqM)


def test_eye():
    qi = nq.eye()
    assert qi.dtype.kind == 'f'
    assert np.all([1, 0, 0, 0] == qi)
    assert np.allclose(nq.quat2mat(qi), np.eye(3))


def test_norm():
    qi = nq.eye()
    assert nq.norm(qi) == 1
    assert nq.isunit(qi)
    qi[1] = 0.2
    assert not nq.isunit(qi)


@pytest.mark.parametrize(('M1', 'q1'), eg_pairs[0::4])
@pytest.mark.parametrize(('M2', 'q2'), eg_pairs[1::4])
def test_mult(M1, q1, M2, q2):
    # Test that quaternion * same as matrix *
    q21 = nq.mult(q2, q1)
    assert_array_almost_equal, M2 @ M1, nq.quat2mat(q21)


@pytest.mark.parametrize(('M', 'q'), eg_pairs)
def test_inverse(M, q):
    iq = nq.inverse(q)
    iqM = nq.quat2mat(iq)
    iM = np.linalg.inv(M)
    assert np.allclose(iM, iqM)


@pytest.mark.parametrize('vec', np.eye(3))
@pytest.mark.parametrize(('M', 'q'), eg_pairs)
def test_qrotate(vec, M, q):
    vdash = nq.rotate_vector(vec, q)
    vM = M @ vec
    assert_array_almost_equal(vdash, vM)


@pytest.mark.parametrize('q', unit_quats)
def test_quaternion_reconstruction(q):
    # Test reconstruction of arbitrary unit quaternions
    M = nq.quat2mat(q)
    qt = nq.mat2quat(M)
    # Accept positive or negative match
    posm = np.allclose(q, qt)
    negm = np.allclose(q, -qt)
    assert posm or negm


def test_angle_axis2quat():
    q = nq.angle_axis2quat(0, [1, 0, 0])
    assert_array_equal(q, [1, 0, 0, 0])
    q = nq.angle_axis2quat(np.pi, [1, 0, 0])
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = nq.angle_axis2quat(np.pi, [1, 0, 0], True)
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = nq.angle_axis2quat(np.pi, [2, 0, 0], False)
    assert_array_almost_equal(q, [0, 1, 0, 0])


def test_angle_axis():
    for M, q in eg_pairs:
        theta, vec = nq.quat2angle_axis(q)
        q2 = nq.angle_axis2quat(theta, vec)
        nq.nearly_equivalent(q, q2)
        aa_mat = nq.angle_axis2mat(theta, vec)
        assert_array_almost_equal(aa_mat, M)
        unit_vec = norm(vec)
        aa_mat2 = nq.angle_axis2mat(theta, unit_vec, is_normalized=True)
        assert_array_almost_equal(aa_mat2, M)
