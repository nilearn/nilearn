# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Testing for orientations module"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..affines import from_matvec, to_matvec
from ..orientations import (
    OrientationError,
    aff2axcodes,
    apply_orientation,
    axcodes2ornt,
    flip_axis,
    inv_ornt_aff,
    io_orientation,
    ornt2axcodes,
    ornt_transform,
)
from ..testing import deprecated_to, expires

IN_ARRS = [
    np.eye(4),
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    [
        [3, 1, 0, 0],
        [1, 3, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    [
        [1, 3, 0, 0],
        [3, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
]

OUT_ORNTS = [
    [
        [0, 1],
        [1, 1],
        [2, 1],
    ],
    [
        [2, 1],
        [1, 1],
        [0, 1],
    ],
    [
        [2, 1],
        [0, 1],
        [1, 1],
    ],
    [
        [0, 1],
        [1, 1],
        [2, 1],
    ],
    [
        [1, 1],
        [0, 1],
        [2, 1],
    ],
]

IN_ARRS.extend(
    [
        [np.cos(np.pi / 6 + i * np.pi / 2), np.sin(np.pi / 6 + i * np.pi / 2), 0, 0],
        [-np.sin(np.pi / 6 + i * np.pi / 2), np.cos(np.pi / 6 + i * np.pi / 2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    for i in range(4)
)

OUT_ORNTS += [
    [
        [0, 1],
        [1, 1],
        [2, 1],
    ],
    [
        [1, -1],
        [0, 1],
        [2, 1],
    ],
    [
        [0, -1],
        [1, -1],
        [2, 1],
    ],
    [
        [1, 1],
        [0, -1],
        [2, 1],
    ],
]


IN_ARRS = [np.array(arr) for arr in IN_ARRS]
OUT_ORNTS = [np.array(ornt) for ornt in OUT_ORNTS]


_LABELS = ['RL', 'AP', 'SI']
ALL_AXCODES = [
    (_LABELS[i0][j0], _LABELS[i1][j1], _LABELS[i2][j2])
    for i0 in range(3)
    for i1 in range(3)
    for i2 in range(3)
    if i0 != i1 != i2 != i0
    for j0 in range(2)
    for j1 in range(2)
    for j2 in range(2)
]

ALL_ORNTS = [
    [[i0, j0], [i1, j1], [i2, j2]]
    for i0 in range(3)
    for i1 in range(3)
    for i2 in range(3)
    if i0 != i1 != i2 != i0
    for j0 in [1, -1]
    for j1 in [1, -1]
    for j2 in [1, -1]
]


def same_transform(taff, ornt, shape):
    # Applying transformations implied by `ornt` to a made-up array
    # ``arr`` of shape `shape`, results in ``t_arr``. When the point
    # indices from ``arr`` are transformed by (the inverse of) `taff`,
    # and we index into ``t_arr`` with these transformed points, then we
    # should get the same values as we would from indexing into arr with
    # the untransformed points.
    shape = np.array(shape)
    size = np.prod(shape)
    arr = np.arange(size).reshape(shape)
    # apply ornt transformations
    t_arr = apply_orientation(arr, ornt)
    # get all point indices in arr
    i, j, k = shape
    arr_pts = np.mgrid[:i, :j, :k].reshape((3, -1))
    # inverse of taff takes us from point index in arr to point index in
    # t_arr
    itaff = np.linalg.inv(taff)
    # apply itaff so that points indexed in t_arr should correspond
    o2t_pts = np.dot(itaff[:3, :3], arr_pts) + itaff[:3, 3][:, None]
    assert np.allclose(np.round(o2t_pts), o2t_pts)
    # fancy index out the t_arr values
    vals = t_arr[tuple(o2t_pts.astype('i'))]
    return np.all(vals == arr.ravel())


def test_apply():
    # most tests are in ``same_transform`` above, via the
    # test_io_orientations
    a = np.arange(24).reshape((2, 3, 4))
    # Test 4D with an example orientation
    ornt = OUT_ORNTS[-1]
    t_arr = apply_orientation(a[:, :, :, None], ornt)
    assert t_arr.ndim == 4
    # Orientation errors
    with pytest.raises(OrientationError):
        apply_orientation(a[:, :, 1], ornt)
    with pytest.raises(OrientationError):
        apply_orientation(a, [[0, 1], [np.nan, np.nan], [2, 1]])
    for ornt in ALL_ORNTS:
        t_arr = apply_orientation(a, ornt)
        assert_array_equal(a.shape, np.array(t_arr.shape)[np.array(ornt)[:, 0]])


def test_io_orientation():
    for shape in ((2, 3, 4), (20, 15, 7)):
        for in_arr, out_ornt in zip(IN_ARRS, OUT_ORNTS):
            ornt = io_orientation(in_arr)
            assert_array_equal(ornt, out_ornt)
            taff = inv_ornt_aff(ornt, shape)
            assert same_transform(taff, ornt, shape)
            for axno in range(3):
                arr = in_arr.copy()
                ex_ornt = out_ornt.copy()
                # flip the input axis in affine
                arr[:, axno] *= -1
                # check that result shows flip
                ex_ornt[axno, 1] *= -1
                ornt = io_orientation(arr)
                assert_array_equal(ornt, ex_ornt)
                taff = inv_ornt_aff(ornt, shape)
                assert same_transform(taff, ornt, shape)
    # Test nasty hang for zero columns
    rzs = np.c_[np.diag([2, 3, 4, 5]), np.zeros((4, 3))]
    arr = from_matvec(rzs, [15, 16, 17, 18])
    ornt = io_orientation(arr)
    assert_array_equal(
        ornt,
        [
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ],
    )
    # Test behavior of thresholding
    def_aff = np.array(
        [
            [1.0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    fail_tol = np.array(
        [
            [0, 1],
            [np.nan, np.nan],
            [2, 1],
        ]
    )
    pass_tol = np.array(
        [
            [0, 1],
            [1, 1],
            [2, 1],
        ]
    )
    eps = np.finfo(float).eps
    # Test that a Y axis appears as we increase the difference between the
    # first two columns
    for y_val, has_y in (
        (0, False),
        (eps, False),
        (eps * 5, False),
        (eps * 10, True),
    ):
        def_aff[1, 1] = y_val
        res = pass_tol if has_y else fail_tol
        assert_array_equal(io_orientation(def_aff), res)
    # Test tol input argument
    def_aff[1, 1] = eps
    assert_array_equal(io_orientation(def_aff, tol=0), pass_tol)
    def_aff[1, 1] = eps * 10
    assert_array_equal(io_orientation(def_aff, tol=1e-5), fail_tol)
    # Test drop of rows, columns
    mat, vec = to_matvec(def_aff)
    aff_extra_col = np.zeros((4, 5))
    aff_extra_col[-1, -1] = 1  # Not strictly necessary, but for completeness
    aff_extra_col[:3, :3] = mat
    aff_extra_col[:3, -1] = vec
    assert_array_equal(
        io_orientation(aff_extra_col, tol=1e-5),
        [
            [0, 1],
            [np.nan, np.nan],
            [2, 1],
            [np.nan, np.nan],
        ],
    )
    aff_extra_row = np.zeros((5, 4))
    aff_extra_row[-1, -1] = 1  # Not strictly necessary, but for completeness
    aff_extra_row[:3, :3] = mat
    aff_extra_row[:3, -1] = vec
    assert_array_equal(
        io_orientation(aff_extra_row, tol=1e-5),
        [
            [0, 1],
            [np.nan, np.nan],
            [2, 1],
        ],
    )


def test_ornt_transform():
    assert_array_equal(
        ornt_transform(
            [[0, 1], [1, 1], [2, -1]],
            [[1, 1], [0, 1], [2, 1]],
        ),
        [[1, 1], [0, 1], [2, -1]],
    )
    assert_array_equal(
        ornt_transform(
            [[0, 1], [1, 1], [2, 1]],
            [[2, 1], [0, -1], [1, 1]],
        ),
        [[1, -1], [2, 1], [0, 1]],
    )
    # Must have same shape
    with pytest.raises(ValueError):
        ornt_transform([[0, 1], [1, 1]], [[0, 1], [1, 1], [2, 1]])

    # Must be (N,2) in shape
    with pytest.raises(ValueError):
        ornt_transform(
            [[0, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [1, 1, 1]],
        )

    # Target axes must exist in source
    with pytest.raises(ValueError):
        ornt_transform(
            [[0, 1], [1, 1], [1, 1]],
            [[0, 1], [1, 1], [2, 1]],
        )


def test_ornt2axcodes():
    # Recoding orientation to axis codes
    labels = (('left', 'right'), ('back', 'front'), ('down', 'up'))
    assert ornt2axcodes([[0, 1], [1, 1], [2, 1]], labels) == ('right', 'front', 'up')
    assert ornt2axcodes([[0, -1], [1, -1], [2, -1]], labels) == ('left', 'back', 'down')
    assert ornt2axcodes([[2, -1], [1, -1], [0, -1]], labels) == ('down', 'back', 'left')
    assert ornt2axcodes([[1, 1], [2, -1], [0, 1]], labels) == ('front', 'down', 'right')
    # default is RAS output directions
    assert ornt2axcodes([[0, 1], [1, 1], [2, 1]]) == ('R', 'A', 'S')
    # dropped axes produce None
    assert ornt2axcodes([[0, 1], [np.nan, np.nan], [2, 1]]) == ('R', None, 'S')
    # Non integer axes raises error
    with pytest.raises(ValueError):
        ornt2axcodes([[0.1, 1]])
    # As do directions not in range
    with pytest.raises(ValueError):
        ornt2axcodes([[0, 0]])

    for axcodes, ornt in zip(ALL_AXCODES, ALL_ORNTS):
        assert ornt2axcodes(ornt) == axcodes


def test_axcodes2ornt():
    # Go from axcodes back to orientations
    labels = (('left', 'right'), ('back', 'front'), ('down', 'up'))
    assert_array_equal(axcodes2ornt(('right', 'front', 'up'), labels), [[0, 1], [1, 1], [2, 1]])
    assert_array_equal(axcodes2ornt(('left', 'back', 'down'), labels), [[0, -1], [1, -1], [2, -1]])
    assert_array_equal(axcodes2ornt(('down', 'back', 'left'), labels), [[2, -1], [1, -1], [0, -1]])
    assert_array_equal(axcodes2ornt(('front', 'down', 'right'), labels), [[1, 1], [2, -1], [0, 1]])

    # default is RAS output directions
    default = np.c_[range(3), [1] * 3]
    assert_array_equal(axcodes2ornt(('R', 'A', 'S')), default)

    # dropped axes produce None
    assert_array_equal(axcodes2ornt(('R', None, 'S')), [[0, 1], [np.nan, np.nan], [2, 1]])

    # Missing axcodes raise an error
    assert_array_equal(axcodes2ornt('RAS'), default)
    with pytest.raises(ValueError):
        axcodes2ornt('rAS')
    # None is OK as axis code
    assert_array_equal(axcodes2ornt(('R', None, 'S')), [[0, 1], [np.nan, np.nan], [2, 1]])
    # Bad axis code with None also raises error.
    with pytest.raises(ValueError):
        axcodes2ornt(('R', None, 's'))
    # Axis codes checked with custom labels
    labels = ('SD', 'BF', 'lh')
    assert_array_equal(axcodes2ornt('BlD', labels), [[1, -1], [2, -1], [0, 1]])
    with pytest.raises(ValueError):
        axcodes2ornt('blD', labels)

    # Duplicate labels
    for labels in [('SD', 'BF', 'lD'), ('SD', 'SF', 'lD')]:
        with pytest.raises(ValueError):
            axcodes2ornt('blD', labels)

    for axcodes, ornt in zip(ALL_AXCODES, ALL_ORNTS):
        assert_array_equal(axcodes2ornt(axcodes), ornt)


def test_aff2axcodes():
    assert aff2axcodes(np.eye(4)) == tuple('RAS')
    aff = [[0, 1, 0, 10], [-1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
    assert aff2axcodes(aff, (('L', 'R'), ('B', 'F'), ('D', 'U'))) == ('B', 'R', 'U')
    assert aff2axcodes(aff, (('L', 'R'), ('B', 'F'), ('D', 'U'))) == ('B', 'R', 'U')


def test_inv_ornt_aff():
    # Extra tests for inv_ornt_aff routines (also tested in
    # io_orientations test)
    with pytest.raises(OrientationError):
        inv_ornt_aff([[0, 1], [1, -1], [np.nan, np.nan]], (3, 4, 5))


@expires('5.0.0')
def test_flip_axis_deprecation():
    a = np.arange(24).reshape((2, 3, 4))
    axis = 1
    with deprecated_to('5.0.0'):
        a_flipped = flip_axis(a, axis)
    assert_array_equal(a_flipped, np.flip(a, axis))
