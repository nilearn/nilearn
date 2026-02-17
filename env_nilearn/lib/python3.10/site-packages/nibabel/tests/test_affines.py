# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

from ..affines import (
    AffineError,
    append_diag,
    apply_affine,
    dot_reduce,
    from_matvec,
    obliquity,
    rescale_affine,
    to_matvec,
    voxel_sizes,
)
from ..eulerangles import euler2mat
from ..orientations import aff2axcodes


def validated_apply_affine(T, xyz):
    # This was the original apply_affine implementation that we've stashed here
    # to test against
    xyz = np.asarray(xyz)
    shape = xyz.shape[0:-1]
    XYZ = np.dot(np.reshape(xyz, (np.prod(shape), 3)), T[0:3, 0:3].T)
    XYZ[:, 0] += T[0, 3]
    XYZ[:, 1] += T[1, 3]
    XYZ[:, 2] += T[2, 3]
    XYZ = np.reshape(XYZ, shape + (3,))
    return XYZ


def test_apply_affine():
    rng = np.random.RandomState(20110903)
    aff = np.diag([2, 3, 4, 1])
    pts = rng.uniform(size=(4, 3))
    assert_array_equal(apply_affine(aff, pts), pts * [[2, 3, 4]])
    aff[:3, 3] = [10, 11, 12]
    assert_array_equal(apply_affine(aff, pts), pts * [[2, 3, 4]] + [[10, 11, 12]])
    aff[:3, :] = rng.normal(size=(3, 4))
    exp_res = np.concatenate((pts.T, np.ones((1, 4))), axis=0)
    exp_res = np.dot(aff, exp_res)[:3, :].T
    assert_array_equal(apply_affine(aff, pts), exp_res)
    # Check we get the same result as the previous implementation
    assert_almost_equal(validated_apply_affine(aff, pts), apply_affine(aff, pts))
    # Check that lists work for inputs
    assert_array_equal(apply_affine(aff.tolist(), pts.tolist()), exp_res)
    # Check that it's the same as a banal implementation in the simple case
    aff = np.array([[0, 2, 0, 10], [3, 0, 0, 11], [0, 0, 4, 12], [0, 0, 0, 1]])
    pts = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6], [6, 7, 8]])
    exp_res = (np.dot(aff[:3, :3], pts.T) + aff[:3, 3:4]).T
    assert_array_equal(apply_affine(aff, pts), exp_res)
    # That points can be reshaped and you'll get the same shape output
    pts = pts.reshape((2, 2, 3))
    exp_res = exp_res.reshape((2, 2, 3))
    assert_array_equal(apply_affine(aff, pts), exp_res)

    # Check inplace modification.
    res = apply_affine(aff, pts, inplace=True)
    assert_array_equal(res, exp_res)
    assert np.shares_memory(res, pts)

    # That ND also works
    for N in range(2, 6):
        aff = np.eye(N)
        nd = N - 1
        aff[:nd, :nd] = rng.normal(size=(nd, nd))
        pts = rng.normal(size=(2, 3, nd))
        res = apply_affine(aff, pts)
        # crude apply
        new_pts = np.ones((N, 6))
        new_pts[:-1, :] = np.rollaxis(pts, -1).reshape((nd, 6))
        exp_pts = np.dot(aff, new_pts)
        exp_pts = np.rollaxis(exp_pts[:-1, :], 0, 2)
        exp_res = exp_pts.reshape((2, 3, nd))
        assert_array_almost_equal(res, exp_res)


def test_matrix_vector():
    for M, N in ((4, 4), (5, 4), (4, 5)):
        xform = np.zeros((M, N))
        xform[:-1, :] = np.random.normal(size=(M - 1, N))
        xform[-1, -1] = 1
        newmat, newvec = to_matvec(xform)
        mat = xform[:-1, :-1]
        vec = xform[:-1, -1]
        assert_array_equal(newmat, mat)
        assert_array_equal(newvec, vec)
        assert newvec.shape == (M - 1,)
        assert_array_equal(from_matvec(mat, vec), xform)
        # Check default translation works
        xform_not = xform[:]
        xform_not[:-1, :] = 0
        assert_array_equal(from_matvec(mat), xform)
        assert_array_equal(from_matvec(mat, None), xform)
    # Check array-like works
    newmat, newvec = to_matvec(xform.tolist())
    assert_array_equal(newmat, mat)
    assert_array_equal(newvec, vec)
    assert_array_equal(from_matvec(mat.tolist(), vec.tolist()), xform)


def test_append_diag():
    # Routine for appending diagonal elements
    assert_array_equal(append_diag(np.diag([2, 3, 1]), [1]), np.diag([2, 3, 1, 1]))
    assert_array_equal(append_diag(np.diag([2, 3, 1]), [1, 1]), np.diag([2, 3, 1, 1, 1]))
    aff = np.array(
        [
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    assert_array_equal(
        append_diag(aff, [5], [9]),
        [
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 5, 9],
            [0, 0, 0, 1],
        ],
    )
    assert_array_equal(
        append_diag(aff, [5, 6], [9, 10]),
        [
            [2, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 5, 0, 9],
            [0, 0, 0, 6, 10],
            [0, 0, 0, 0, 1],
        ],
    )
    aff = np.array(
        [
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    assert_array_equal(
        append_diag(aff, [5], [9]),
        [
            [2, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 0, 5, 9],
            [0, 0, 0, 0, 1],
        ],
    )
    # Length of starts has to match length of steps
    with pytest.raises(AffineError):
        append_diag(aff, [5, 6], [9])


def test_dot_reduce():
    # Chaining numpy dot
    # Error for no arguments
    with pytest.raises(TypeError):
        dot_reduce()
    # Anything at all on its own, passes through
    assert dot_reduce(1) == 1
    assert dot_reduce(None) is None
    assert dot_reduce([1, 2, 3]) == [1, 2, 3]
    # Two or more -> dot product
    vec = [1, 2, 3]
    mat = np.arange(4, 13).reshape((3, 3))
    assert_array_equal(dot_reduce(vec, mat), np.dot(vec, mat))
    assert_array_equal(dot_reduce(mat, vec), np.dot(mat, vec))
    mat2 = np.arange(13, 22).reshape((3, 3))
    assert_array_equal(dot_reduce(mat2, vec, mat), mat2 @ (vec @ mat))
    assert_array_equal(dot_reduce(mat, vec, mat2), mat @ (vec @ mat2))


def test_voxel_sizes():
    affine = np.diag([2, 3, 4, 1])
    assert_almost_equal(voxel_sizes(affine), [2, 3, 4])
    # Some example rotations
    rotations = []
    for x_rot, y_rot, z_rot in product((0, 0.4), (0, 0.6), (0, 0.8)):
        rotations.append(euler2mat(z_rot, y_rot, x_rot))
    # Works on any size of array
    for n in range(2, 10):
        vox_sizes = np.arange(n) + 4.1
        aff = np.diag(list(vox_sizes) + [1])
        assert_almost_equal(voxel_sizes(aff), vox_sizes)
        # Translations make no difference
        aff[:-1, -1] = np.arange(n) + 10
        assert_almost_equal(voxel_sizes(aff), vox_sizes)
        # Does not have to be square
        new_row = np.vstack((np.zeros(n + 1), aff))
        assert_almost_equal(voxel_sizes(new_row), vox_sizes)
        new_col = np.c_[np.zeros(n + 1), aff]
        assert_almost_equal(voxel_sizes(new_col), [0] + list(vox_sizes))
        if n < 3:
            continue
        # Rotations do not change the voxel size
        for rotation in rotations:
            rot_affine = np.eye(n + 1)
            rot_affine[:3, :3] = rotation
            full_aff = rot_affine.dot(aff)
            assert_almost_equal(voxel_sizes(full_aff), vox_sizes)


def test_obliquity():
    """Check the calculation of inclination of an affine axes."""
    from math import pi

    aligned = np.diag([2.0, 2.0, 2.3, 1.0])
    aligned[:-1, -1] = [-10, -10, -7]
    R = from_matvec(euler2mat(x=0.09, y=0.001, z=0.001), [0.0, 0.0, 0.0])
    oblique = R.dot(aligned)
    assert_almost_equal(obliquity(aligned), [0.0, 0.0, 0.0])
    assert_almost_equal(obliquity(oblique) * 180 / pi, [0.0810285, 5.1569949, 5.1569376])


def test_rescale_affine():
    rng = np.random.RandomState(20200415)
    orig_shape = rng.randint(low=20, high=512, size=(3,))
    orig_aff = np.eye(4)
    orig_aff[:3, :] = rng.normal(size=(3, 4))
    orig_axcodes = aff2axcodes(orig_aff)
    orig_centroid = apply_affine(orig_aff, (orig_shape - 1) // 2)

    for new_shape in (None, tuple(orig_shape), (256, 256, 256), (64, 64, 40)):
        for new_zooms in ((1, 1, 1), (2, 2, 3), (0.5, 0.5, 0.5)):
            new_aff = rescale_affine(orig_aff, orig_shape, new_zooms, new_shape)
            assert aff2axcodes(new_aff) == orig_axcodes
            if new_shape is None:
                new_shape = tuple(orig_shape)
            new_centroid = apply_affine(new_aff, (np.array(new_shape) - 1) // 2)
            assert_almost_equal(new_centroid, orig_centroid)
