"""Tests for spaces module"""

import numpy as np
import numpy.linalg as npl
import pytest
from numpy.testing import assert_almost_equal

from ..affines import apply_affine, from_matvec
from ..eulerangles import euler2mat
from ..nifti1 import Nifti1Image
from ..spaces import slice2volume, vox2out_vox


def assert_all_in(in_shape, in_affine, out_shape, out_affine):
    slices = tuple(slice(N) for N in in_shape)
    n_axes = len(in_shape)
    in_grid = np.mgrid[slices]
    in_grid = np.rollaxis(in_grid, 0, n_axes + 1)
    v2v = npl.inv(out_affine).dot(in_affine)
    if n_axes < 3:  # reduced dimensions case
        new_v2v = np.eye(n_axes + 1)
        new_v2v[:n_axes, :n_axes] = v2v[:n_axes, :n_axes]
        new_v2v[:n_axes, -1] = v2v[:n_axes, -1]
        v2v = new_v2v
    out_grid = apply_affine(v2v, in_grid)
    TINY = 1e-12
    assert np.all(out_grid > -TINY)
    assert np.all(out_grid < np.array(out_shape) + TINY)


def get_outspace_params():
    # Return in_shape, in_aff, vox, out_shape, out_aff for output space tests
    # Put in function to use also for resample_to_output tests
    # Some affines as input to the tests
    trans_123 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    trans_m123 = [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]
    rot_3 = from_matvec(euler2mat(np.pi / 4), [0, 0, 0])
    return (  # in_shape, in_aff, vox, out_shape, out_aff
        # Identity
        ((2, 3, 4), np.eye(4), None, (2, 3, 4), np.eye(4)),
        # Flip first axis
        (
            (2, 3, 4),
            np.diag([-1, 1, 1, 1]),
            None,
            (2, 3, 4),
            [
                [1, 0, 0, -1],  # axis reversed -> -ve offset
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
        # zooms for affine > 1 -> larger grid with default 1mm output voxels
        ((2, 3, 4), np.diag([4, 5, 6, 1]), None, (5, 11, 19), np.eye(4)),
        # set output voxels to be same size as input. back to original shape
        ((2, 3, 4), np.diag([4, 5, 6, 1]), (4, 5, 6), (2, 3, 4), np.diag([4, 5, 6, 1])),
        # Translation preserved in output
        ((2, 3, 4), trans_123, None, (2, 3, 4), trans_123),
        ((2, 3, 4), trans_m123, None, (2, 3, 4), trans_m123),
        # rotation around 3rd axis
        (
            (2, 3, 4),
            rot_3,
            None,
            # x diff, y diff now 3 cos pi / 4 == 2.12, ceil to 3, add 1
            # most negative x now 2 cos pi / 4
            (4, 4, 4),
            [
                [1, 0, 0, -2 * np.cos(np.pi / 4)],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
        # Less than 3 axes
        ((2, 3), np.eye(4), None, (2, 3), np.eye(4)),
        ((2,), np.eye(4), None, (2,), np.eye(4)),
        # Number of voxel sizes matches length
        ((2, 3), np.diag([4, 5, 6, 1]), (4, 5), (2, 3), np.diag([4, 5, 1, 1])),
    )


def test_vox2out_vox():
    # Test world space bounding box
    # Test basic case, identity, no voxel sizes passed
    shape, aff = vox2out_vox(((2, 3, 4), np.eye(4)))
    assert shape == (2, 3, 4)
    assert (aff == np.eye(4)).all()
    for in_shape, in_aff, vox, out_shape, out_aff in get_outspace_params():
        img = Nifti1Image(np.ones(in_shape), in_aff)
        for input in ((in_shape, in_aff), img):
            shape, aff = vox2out_vox(input, vox)
            assert_all_in(in_shape, in_aff, shape, aff)
            assert shape == out_shape
            assert_almost_equal(aff, out_aff)
            assert isinstance(shape, tuple)
            assert isinstance(shape[0], int)
    # Enforce number of axes
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4, 5), np.eye(4)))
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4, 5, 6), np.eye(4)))
    # Voxel sizes must be positive
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4), np.eye(4), [-1, 1, 1]))
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4), np.eye(4), [1, 0, 1]))


def test_slice2volume():
    # Get affine expressing selection of single slice from volume
    for axis, def_aff in zip(
        (0, 1, 2),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
        ),
    ):
        for val in (0, 5, 10):
            exp_aff = np.array(def_aff)
            exp_aff[axis, -1] = val
            assert (slice2volume(val, axis) == exp_aff).all()


@pytest.mark.parametrize(
    ('index', 'axis'),
    [
        [-1, 0],
        [0, -1],
        [0, 3],
    ],
)
def test_slice2volume_exception(index, axis):
    with pytest.raises(ValueError):
        slice2volume(index, axis)
