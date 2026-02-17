# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for Euler angles"""

import math

import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal

from .. import eulerangles as nea
from .. import quaternions as nq

FLOAT_EPS = np.finfo(np.float64).eps

# Example rotations """
params = np.arange(-pi * 2, pi * 2.5, pi / 2)
eg_rots = [(x, y, z) for x in params for y in params for z in params]


def x_only(x):
    cosx = np.cos(x)
    sinx = np.sin(x)
    return np.array(
        [
            [1, 0, 0],
            [0, cosx, -sinx],
            [0, sinx, cosx],
        ]
    )


def y_only(y):
    cosy = np.cos(y)
    siny = np.sin(y)
    return np.array(
        [
            [cosy, 0, siny],
            [0, 1, 0],
            [-siny, 0, cosy],
        ]
    )


def z_only(z):
    cosz = np.cos(z)
    sinz = np.sin(z)
    return np.array(
        [
            [cosz, -sinz, 0],
            [sinz, cosz, 0],
            [0, 0, 1],
        ]
    )


def sympy_euler(z, y, x):
    # The whole matrix formula for z,y,x rotations from Sympy
    cos = math.cos
    sin = math.sin
    # the following copy / pasted from Sympy - see derivations subdirectory
    return [
        [cos(y) * cos(z), -cos(y) * sin(z), sin(y)],
        [
            cos(x) * sin(z) + cos(z) * sin(x) * sin(y),
            cos(x) * cos(z) - sin(x) * sin(y) * sin(z),
            -cos(y) * sin(x),
        ],
        [
            sin(x) * sin(z) - cos(x) * cos(z) * sin(y),
            cos(z) * sin(x) + cos(x) * sin(y) * sin(z),
            cos(x) * cos(y),
        ],
    ]


def is_valid_rotation(M):
    if not np.allclose(np.linalg.det(M), 1):
        return False
    return np.allclose(np.eye(3), np.dot(M, M.T))


def test_basic_euler():
    # some example rotations, in radians
    zr = 0.05
    yr = -0.4
    xr = 0.2
    # Rotation matrix composing the three rotations
    M = nea.euler2mat(zr, yr, xr)
    # Corresponding individual rotation matrices
    M1 = nea.euler2mat(zr)
    M2 = nea.euler2mat(0, yr)
    M3 = nea.euler2mat(0, 0, xr)
    # which are all valid rotation matrices
    assert is_valid_rotation(M)
    assert is_valid_rotation(M1)
    assert is_valid_rotation(M2)
    assert is_valid_rotation(M3)
    # Full matrix is composition of three individual matrices
    assert np.allclose(M, np.dot(M3, np.dot(M2, M1)))
    # Rotations can be specified with named args, default 0
    assert np.all(nea.euler2mat(zr) == nea.euler2mat(z=zr))
    assert np.all(nea.euler2mat(0, yr) == nea.euler2mat(y=yr))
    assert np.all(nea.euler2mat(0, 0, xr) == nea.euler2mat(x=xr))
    # Applying an opposite rotation same as inverse (the inverse is
    # the same as the transpose, but just for clarity)
    assert np.allclose(nea.euler2mat(x=-xr), np.linalg.inv(nea.euler2mat(x=xr)))


def test_euler_mat_1():
    M = nea.euler2mat()
    assert_array_equal(M, np.eye(3))


@pytest.mark.parametrize(('x', 'y', 'z'), eg_rots)
def test_euler_mat_2(x, y, z):
    M1 = nea.euler2mat(z, y, x)
    M2 = sympy_euler(z, y, x)
    assert_array_almost_equal(M1, M2)
    M3 = np.dot(x_only(x), np.dot(y_only(y), z_only(z)))
    assert_array_almost_equal(M1, M3)
    zp, yp, xp = nea.mat2euler(M1)
    # The parameters may not be the same as input, but they give the
    # same rotation matrix
    M4 = nea.euler2mat(zp, yp, xp)
    assert_array_almost_equal(M1, M4)


def sympy_euler2quat(z=0, y=0, x=0):
    # direct formula for z,y,x quaternion rotations using sympy
    # see derivations subfolder
    cos = math.cos
    sin = math.sin
    # the following copy / pasted from Sympy output
    return (
        cos(0.5 * x) * cos(0.5 * y) * cos(0.5 * z) - sin(0.5 * x) * sin(0.5 * y) * sin(0.5 * z),
        cos(0.5 * x) * sin(0.5 * y) * sin(0.5 * z) + cos(0.5 * y) * cos(0.5 * z) * sin(0.5 * x),
        cos(0.5 * x) * cos(0.5 * z) * sin(0.5 * y) - cos(0.5 * y) * sin(0.5 * x) * sin(0.5 * z),
        cos(0.5 * x) * cos(0.5 * y) * sin(0.5 * z) + cos(0.5 * z) * sin(0.5 * x) * sin(0.5 * y),
    )


def crude_mat2euler(M):
    """The simplest possible - ignoring atan2 instability"""
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    return math.atan2(-r12, r11), math.asin(r13), math.atan2(-r23, r33)


def test_euler_instability():
    # Test for numerical errors in mat2euler
    # problems arise for cos(y) near 0
    po2 = pi / 2
    zyx = po2, po2, po2
    M = nea.euler2mat(*zyx)
    # Round trip
    M_back = nea.euler2mat(*nea.mat2euler(M))
    assert np.allclose(M, M_back)
    # disturb matrix slightly
    M_e = M - FLOAT_EPS
    # round trip to test - OK
    M_e_back = nea.euler2mat(*nea.mat2euler(M_e))
    assert np.allclose(M_e, M_e_back)
    # not so with crude routine
    M_e_back = nea.euler2mat(*crude_mat2euler(M_e))
    assert not np.allclose(M_e, M_e_back)


@pytest.mark.parametrize(('x', 'y', 'z'), eg_rots)
def test_quats(x, y, z):
    M1 = nea.euler2mat(z, y, x)
    quatM = nq.mat2quat(M1)
    quat = nea.euler2quat(z, y, x)
    assert nq.nearly_equivalent(quatM, quat)
    quatS = sympy_euler2quat(z, y, x)
    assert nq.nearly_equivalent(quat, quatS)
    zp, yp, xp = nea.quat2euler(quat)
    # The parameters may not be the same as input, but they give the
    # same rotation matrix
    M2 = nea.euler2mat(zp, yp, xp)
    assert_array_almost_equal(M1, M2)
