"""Test the resampling code."""
import copy
import math
import os
from pathlib import Path

import numpy as np
import pytest
from nibabel import Nifti1Header, Nifti1Image
from nibabel.freesurfer import MGHImage
from nilearn import _utils
from nilearn._utils import testing
from nilearn.image import get_data
from nilearn.image.image import _pad_array, crop_img
from nilearn.image.resampling import (
    BoundingBoxError,
    coord_transform,
    from_matrix_vector,
    get_bounds,
    reorder_img,
    resample_img,
    resample_to_img,
)
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)


###############################################################################
# Helper function
def rotation(theta, phi):
    """Returns a rotation 3x3 matrix."""
    cos = np.cos
    sin = np.sin
    a1 = np.array(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
    )
    a2 = np.array(
        [[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]]
    )
    return np.dot(a1, a2)


###############################################################################
# Tests
def test_identity_resample():
    """Test resampling with an identity affine."""
    rng = np.random.RandomState(42)
    shape = (3, 2, 5, 2)
    data = rng.randint(0, 10, shape, dtype="int32")
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=affine,
        interpolation="nearest",
    )
    np.testing.assert_almost_equal(data, get_data(rot_img))
    # Smoke-test with a list affine
    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=affine.tolist(),
        interpolation="nearest",
    )
    # Test with a 3x3 affine
    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=affine[:3, :3],
        interpolation="nearest",
    )
    np.testing.assert_almost_equal(data, get_data(rot_img))

    # Test with non native endian data

    # Test with big endian data ('>f8')
    for interpolation in ["nearest", "linear", "continuous"]:
        rot_img = resample_img(
            Nifti1Image(data.astype(">f8"), affine),
            target_affine=affine.tolist(),
            interpolation=interpolation,
        )
        np.testing.assert_almost_equal(data, get_data(rot_img))

    # Test with little endian data ('<f8')
    for interpolation in ["nearest", "linear", "continuous"]:
        rot_img = resample_img(
            Nifti1Image(data.astype("<f8"), affine),
            target_affine=affine.tolist(),
            interpolation=interpolation,
        )
        np.testing.assert_almost_equal(data, get_data(rot_img))


def test_downsample():
    """Test resampling with a 1/2 down-sampling affine."""
    rng = np.random.RandomState(42)
    shape = (6, 3, 6, 2)
    data = rng.random_sample(shape)
    affine = np.eye(4)
    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=2 * affine,
        interpolation="nearest",
    )
    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    np.testing.assert_almost_equal(
        downsampled, get_data(rot_img)[:x, :y, :z, ...]
    )

    rot_img_2 = resample_img(
        Nifti1Image(data, affine),
        target_affine=2 * affine,
        interpolation="nearest",
        force_resample=True,
    )
    np.testing.assert_almost_equal(get_data(rot_img_2), get_data(rot_img))
    # Test with non native endian data

    # Test to check that if giving non native endian data as input should
    # work as normal and expected to return the same output as above tests.

    # Big endian data ('>f8')
    for copy_data in [True, False]:
        rot_img = resample_img(
            Nifti1Image(data.astype(">f8"), affine),
            target_affine=2 * affine,
            interpolation="nearest",
            copy=copy_data,
        )
        np.testing.assert_almost_equal(
            downsampled, get_data(rot_img)[:x, :y, :z, ...]
        )

    # Little endian data
    for copy_data in [True, False]:
        rot_img = resample_img(
            Nifti1Image(data.astype("<f8"), affine),
            target_affine=2 * affine,
            interpolation="nearest",
            copy=copy_data,
        )
        np.testing.assert_almost_equal(
            downsampled, get_data(rot_img)[:x, :y, :z, ...]
        )


def test_resampling_fill_value():
    """Test resampling with a non-zero fill value"""
    rng = np.random.RandomState(42)

    data_3d = rng.uniform(size=(1, 4, 4))
    data_4d = rng.uniform(size=(1, 4, 4, 3))

    angle = np.pi / 4
    rot = rotation(0, angle)

    # Try a few different fill values
    for data in [data_3d, data_4d]:
        for val in (-3.75, 0):
            if val:
                rot_img = resample_img(
                    Nifti1Image(data, np.eye(4)),
                    target_affine=rot,
                    interpolation="nearest",
                    fill_value=val,
                    clip=False,
                )
            else:
                rot_img = resample_img(
                    Nifti1Image(data, np.eye(4)),
                    target_affine=rot,
                    interpolation="nearest",
                    clip=False,
                )
            assert get_data(rot_img).flatten()[0] == val

            rot_img2 = resample_to_img(
                Nifti1Image(data, np.eye(4)),
                rot_img,
                interpolation="nearest",
                fill_value=val,
            )
            assert get_data(rot_img2).flatten()[0] == val


def test_resampling_with_affine():
    """Test resampling with a given rotation part of the affine."""
    rng = np.random.RandomState(42)

    data_4d = rng.randint(4, size=(1, 4, 4, 3), dtype="int32")
    data_3d = rng.randint(4, size=(1, 4, 4), dtype="int32")

    for data in [data_3d, data_4d]:
        for angle in (0, np.pi, np.pi / 2.0, np.pi / 4.0, np.pi / 3.0):
            rot = rotation(0, angle)
            rot_img = resample_img(
                Nifti1Image(data, np.eye(4)),
                target_affine=rot,
                interpolation="nearest",
            )
            assert np.max(data) == np.max(get_data(rot_img))
            assert get_data(rot_img).dtype == data.dtype

    # We take the same rotation logic as above and test with nonnative endian
    # data as input
    for data in [data_3d, data_4d]:
        img = Nifti1Image(data.astype(">f8"), np.eye(4))
        for angle in (0, np.pi, np.pi / 2.0, np.pi / 4.0, np.pi / 3.0):
            rot = rotation(0, angle)
            rot_img = resample_img(
                img, target_affine=rot, interpolation="nearest"
            )
            assert np.max(data) == np.max(get_data(rot_img))


def test_resampling_continuous_with_affine():
    rng = np.random.RandomState(42)

    data_3d = rng.randint(1, 4, size=(1, 10, 10), dtype="int32")
    data_4d = rng.randint(1, 4, size=(1, 10, 10, 3), dtype="int32")

    for data in [data_3d, data_4d]:
        for angle in (0, np.pi / 2.0, np.pi, 3 * np.pi / 2.0):
            rot = rotation(0, angle)

            img = Nifti1Image(data, np.eye(4))
            rot_img = resample_img(
                img, target_affine=rot, interpolation="continuous"
            )
            rot_img_back = resample_img(
                rot_img, target_affine=np.eye(4), interpolation="continuous"
            )

            center = slice(1, 9)
            # values on the edges are wrong for some reason
            mask = (0, center, center)
            np.testing.assert_allclose(
                get_data(img)[mask], get_data(rot_img_back)[mask]
            )
            assert get_data(rot_img).dtype == np.dtype(
                data.dtype.name.replace("int", "float")
            )


def test_resampling_error_checks():
    rng = np.random.RandomState(42)
    shape = (3, 2, 5, 2)
    target_shape = (5, 3, 2)
    affine = np.eye(4)
    data = rng.randint(0, 10, shape, dtype="int32")
    img = Nifti1Image(data, affine)

    # Correct parameters: no exception
    resample_img(img, target_shape=target_shape, target_affine=affine)
    resample_img(img, target_affine=affine)

    with testing.write_tmp_imgs(img) as filename:
        resample_img(filename, target_shape=target_shape, target_affine=affine)

    # Missing parameter
    pytest.raises(ValueError, resample_img, img, target_shape=target_shape)

    # Invalid shape
    pytest.raises(
        ValueError,
        resample_img,
        img,
        target_shape=(2, 3),
        target_affine=affine,
    )

    # Invalid interpolation
    with pytest.raises(ValueError, match="interpolation must be one of"):
        resample_img(
            img,
            target_shape=target_shape,
            target_affine=affine,
            interpolation="an_invalid_interpolation",
        )

    # Resampling a binary image with continuous or
    # linear interpolation should raise a warning.
    data_binary = rng.randint(4, size=(1, 4, 4), dtype="int32")
    data_binary[data_binary > 0] = 1
    assert sorted(list(np.unique(data_binary))) == [0, 1]

    rot = rotation(0, np.pi / 4)
    img_binary = Nifti1Image(data_binary, np.eye(4))
    assert _utils.niimg._is_binary_niimg(img_binary)

    with pytest.warns(Warning, match="Resampling binary images with"):
        resample_img(img_binary, target_affine=rot, interpolation="continuous")

    with pytest.warns(Warning, match="Resampling binary images with"):
        resample_img(img_binary, target_affine=rot, interpolation="linear")
    img_no_sform = Nifti1Image(data, affine)
    img_no_sform.set_sform(None)
    with pytest.warns(Warning, match="The provided image has no sform"):
        resample_img(img_no_sform, target_affine=affine)

    # Noop
    target_shape = shape[:3]

    img_r = resample_img(img, copy=False)
    assert img_r == img

    img_r = resample_img(img, copy=True)
    assert not np.may_share_memory(get_data(img_r), get_data(img))

    np.testing.assert_almost_equal(get_data(img_r), get_data(img))
    np.testing.assert_almost_equal(img_r.affine, img.affine)

    img_r = resample_img(
        img, target_affine=affine, target_shape=target_shape, copy=False
    )
    assert img_r == img

    img_r = resample_img(
        img, target_affine=affine, target_shape=target_shape, copy=True
    )
    assert not np.may_share_memory(get_data(img_r), get_data(img))
    np.testing.assert_almost_equal(get_data(img_r), get_data(img))
    np.testing.assert_almost_equal(img_r.affine, img.affine)


def test_4d_affine_bounding_box_error():
    small_data = np.ones([4, 4, 4])
    small_data_4D_affine = np.eye(4)
    small_data_4D_affine[:3, -1] = np.array([5, 4, 5])

    small_img = Nifti1Image(small_data, small_data_4D_affine)

    bigger_data_4D_affine = np.eye(4)
    bigger_data = np.zeros([10, 10, 10])
    bigger_img = Nifti1Image(bigger_data, bigger_data_4D_affine)

    # We would like to check whether all/most of the data
    # will be contained in the resampled image
    # The measure will be the l2 norm, since some resampling
    # schemes approximately conserve it

    def l2_norm(arr):
        return (arr**2).sum()

    # resample using 4D affine and specified target shape
    small_to_big_with_shape = resample_img(
        small_img,
        target_affine=bigger_img.affine,
        target_shape=bigger_img.shape,
    )
    # resample using 3D affine and no target shape
    small_to_big_without_shape_3D_affine = resample_img(
        small_img, target_affine=bigger_img.affine[:3, :3]
    )
    # resample using 4D affine and no target shape
    small_to_big_without_shape = resample_img(
        small_img, target_affine=bigger_img.affine
    )

    # The first 2 should pass
    assert_almost_equal(
        l2_norm(small_data), l2_norm(get_data(small_to_big_with_shape))
    )
    assert_almost_equal(
        l2_norm(small_data),
        l2_norm(get_data(small_to_big_without_shape_3D_affine)),
    )

    # After correcting decision tree for 4x4 affine given + no target shape
    # from "use initial shape" to "calculate minimal bounding box respecting
    # the affine anchor and the data"
    assert_almost_equal(
        l2_norm(small_data), l2_norm(get_data(small_to_big_without_shape))
    )

    assert_array_equal(
        small_to_big_without_shape.shape,
        small_data_4D_affine[:3, -1] + np.array(small_img.shape),
    )


def test_raises_upon_3x3_affine_and_no_shape():
    img = Nifti1Image(np.zeros([8, 9, 10]), affine=np.eye(4))
    exception = ValueError
    message = (
        "Given target shape without anchor "
        "vector: Affine shape should be \\(4, 4\\) and "
        "not \\(3, 3\\)"
    )
    with pytest.raises(exception, match=message):
        resample_img(
            img, target_affine=np.eye(3) * 2, target_shape=(10, 10, 10)
        )


def test_3x3_affine_bbox():
    # Test that the bounding-box is properly computed when
    # transforming with a negative affine component
    # This is specifically to test for a change in behavior between
    # scipy < 0.18 and scipy >= 0.18, which is an interaction between
    # offset and a diagonal affine
    image = np.ones((20, 30))
    source_affine = np.eye(4)
    # Give the affine an offset
    source_affine[:2, 3] = np.array([96, 64])

    # We need to turn this data into a nibabel image
    img = Nifti1Image(image[:, :, np.newaxis], affine=source_affine)

    target_affine_3x3 = np.eye(3) * 2
    # One negative axes
    target_affine_3x3[1] *= -1

    img_3d_affine = resample_img(img, target_affine=target_affine_3x3)

    # If the bounding box is computed wrong, the image will be only
    # zeros
    np.testing.assert_allclose(get_data(img_3d_affine).max(), image.max())


def test_raises_bbox_error_if_data_outside_box():
    # Make some cases which should raise exceptions

    # original image
    data = np.zeros([8, 9, 10])
    affine = np.eye(4)
    affine_offset = np.array([1, 1, 1])
    affine[:3, 3] = affine_offset

    img = Nifti1Image(data, affine)

    # some axis flipping affines
    diag = [
        [-1, 1, 1, 1],
        [1, -1, 1, 1],
        [1, 1, -1, 1],
        [-1, -1, 1, 1],
        [-1, 1, -1, 1],
        [1, -1, -1, 1],
    ]
    axis_flips = np.array(list(map(np.diag, diag)))

    # some in plane 90 degree rotations base on these
    # (by permuting two lines)
    af = axis_flips
    rotations = np.array(
        [
            af[0][[1, 0, 2, 3]],
            af[0][[2, 1, 0, 3]],
            af[1][[1, 0, 2, 3]],
            af[1][[0, 2, 1, 3]],
            af[2][[2, 1, 0, 3]],
            af[2][[0, 2, 1, 3]],
        ]
    )

    new_affines = np.concatenate([axis_flips, rotations])
    new_offset = np.array([0.0, 0.0, 0.0])
    new_affines[:, :3, 3] = new_offset[np.newaxis, :]

    exception = BoundingBoxError
    message = (
        "The field of view given "
        "by the target affine does "
        "not contain any of the data"
    )
    for new_affine in new_affines:
        with pytest.raises(exception, match=message):
            resample_img(img, target_affine=new_affine)


def test_resampling_result_axis_permutation():
    # Transform real data using easily checkable transformations
    # For now: axis permutations
    # create a cuboid full of deterministic data, padded with one
    # voxel thickness of zeros
    core_shape = (3, 5, 4)
    core_data = np.arange(np.prod(core_shape)).reshape(core_shape)
    full_data_shape = np.array(core_shape) + 2
    full_data = np.zeros(full_data_shape)
    full_data[tuple(slice(1, 1 + s) for s in core_shape)] = core_data

    source_img = Nifti1Image(full_data, np.eye(4))

    axis_permutations = [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]]

    # check 3x3 transformation matrix
    for ap in axis_permutations:
        target_affine = np.eye(3)[ap]
        resampled_img = resample_img(source_img, target_affine=target_affine)

        resampled_data = get_data(resampled_img)
        what_resampled_data_should_be = full_data.transpose(ap)
        assert_array_almost_equal(
            resampled_data, what_resampled_data_should_be
        )

    # check 4x4 transformation matrix
    offset = np.array([-2, 1, -3])
    for ap in axis_permutations:
        target_affine = np.eye(4)
        target_affine[:3, :3] = np.eye(3)[ap]
        target_affine[:3, 3] = offset

        resampled_img = resample_img(source_img, target_affine=target_affine)
        resampled_data = get_data(resampled_img)
        offset_cropping = (
            np.vstack([-offset[ap][np.newaxis, :], np.zeros([1, 3])])
            .T.ravel()
            .astype(int)
        )
        what_resampled_data_should_be = _pad_array(
            full_data.transpose(ap), list(offset_cropping)
        )

        assert_array_almost_equal(
            resampled_data, what_resampled_data_should_be
        )


def test_resampling_nan():
    # Test that when the data has NaNs they do not propagate to the
    # whole image

    for core_shape in [(3, 5, 4), (3, 5, 4, 2)]:
        # create deterministic data, padded with one
        # voxel thickness of zeros
        core_data = (
            np.arange(np.prod(core_shape))
            .reshape(core_shape)
            .astype(np.float64)
        )
        # Introduce a nan
        core_data[2, 2:4, 1] = np.nan
        full_data_shape = np.array(core_shape) + 2
        full_data = np.zeros(full_data_shape)
        full_data[tuple(slice(1, 1 + s) for s in core_shape)] = core_data

        source_img = Nifti1Image(full_data, np.eye(4))

        # Transform real data using easily checkable transformations
        # For now: axis permutations
        axis_permutation = [0, 1, 2]

        # check 3x3 transformation matrix
        target_affine = np.eye(3)[axis_permutation]
        with pytest.warns(Warning, match=r"(\bnan\b|invalid value)"):
            resampled_img = resample_img(
                source_img, target_affine=target_affine
            )

        resampled_data = get_data(resampled_img)
        if full_data.ndim == 4:
            axis_permutation.append(3)
        what_resampled_data_should_be = full_data.transpose(axis_permutation)
        non_nan = np.isfinite(what_resampled_data_should_be)

        # Check that the input data hasn't been modified:
        assert not np.all(non_nan)

        # Check that for finite value resampling works without problems
        assert_array_almost_equal(
            resampled_data[non_nan], what_resampled_data_should_be[non_nan]
        )

        # Check that what was not finite is still not finite
        assert not np.any(np.isfinite(resampled_data[np.logical_not(non_nan)]))

    # Test with an actual resampling, in the case of a bigish hole
    # This checks the extrapolation mechanism: if we don't do any
    # extrapolation before resampling, the hole creates big
    # artefacts
    data = 10 * np.ones((10, 10, 10))
    data[4:6, 4:6, 4:6] = np.nan
    source_img = Nifti1Image(data, 2 * np.eye(4))
    with pytest.warns(RuntimeWarning):
        resampled_img = resample_img(source_img, target_affine=np.eye(4))

    resampled_data = get_data(resampled_img)
    np.testing.assert_allclose(10, resampled_data[np.isfinite(resampled_data)])


def test_resample_to_img():
    # Testing resample to img function
    rng = np.random.RandomState(42)
    shape = (6, 3, 6, 3)
    data = rng.random_sample(shape)

    source_affine = np.eye(4)
    source_img = Nifti1Image(data, source_affine)

    target_affine = 2 * source_affine
    target_img = Nifti1Image(data, target_affine)

    result_img = resample_to_img(
        source_img, target_img, interpolation="nearest"
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    np.testing.assert_almost_equal(
        downsampled, get_data(result_img)[:x, :y, :z, ...]
    )


def test_crop():
    # Testing that padding of arrays and cropping of images work symmetrically
    shape = (4, 6, 2)
    data = np.ones(shape)
    padded = _pad_array(data, [3, 2, 4, 4, 5, 7])
    padd_nii = Nifti1Image(padded, np.eye(4))

    cropped = crop_img(padd_nii, pad=False)
    np.testing.assert_equal(get_data(cropped), data)


def test_resample_identify_affine_int_translation():
    # Testing resample to img function
    rng = np.random.RandomState(42)

    source_shape = (6, 4, 6)
    source_affine = np.eye(4)
    source_affine[:, 3] = np.append(np.random.randint(0, 4, 3), 1)
    source_data = rng.random_sample(source_shape)
    source_img = Nifti1Image(source_data, source_affine)

    target_shape = (11, 10, 9)
    target_data = np.zeros(target_shape)
    target_affine = source_affine
    target_affine[:3, 3] -= 3  # add an offset of 3 in x, y, z
    target_data[
        3:9, 3:7, 3:9
    ] = source_data  # put the data at the offset location
    target_img = Nifti1Image(target_data, target_affine)

    result_img = resample_to_img(
        source_img, target_img, interpolation="nearest"
    )
    np.testing.assert_almost_equal(get_data(target_img), get_data(result_img))

    result_img_2 = resample_to_img(
        result_img, source_img, interpolation="nearest"
    )
    np.testing.assert_almost_equal(
        get_data(source_img), get_data(result_img_2)
    )

    result_img_3 = resample_to_img(
        result_img, source_img, interpolation="nearest", force_resample=True
    )
    np.testing.assert_almost_equal(
        get_data(result_img_2), get_data(result_img_3)
    )

    result_img_4 = resample_to_img(
        source_img, target_img, interpolation="nearest", force_resample=True
    )
    np.testing.assert_almost_equal(
        get_data(target_img), get_data(result_img_4)
    )


def test_resample_clip():
    # Resample and image and get larger and smaller
    # value than in the original. Use clip to get rid of these images

    shape = (6, 3, 6)
    data = np.zeros(shape=shape)
    data[1:-2, 1:-1, 1:-2] = 1

    source_affine = np.diag((2, 2, 2, 1))
    source_img = Nifti1Image(data, source_affine)

    target_affine = np.eye(4)
    no_clip_data = get_data(
        resample_img(source_img, target_affine, clip=False)
    )
    clip_data = get_data(resample_img(source_img, target_affine, clip=True))

    not_clip = np.where(
        (no_clip_data > data.min()) & (no_clip_data < data.max())
    )

    assert np.any(no_clip_data > data.max())
    assert np.any(no_clip_data < data.min())
    assert np.all(clip_data <= data.max())
    assert np.all(clip_data >= data.min())
    assert_array_equal(no_clip_data[not_clip], clip_data[not_clip])


def test_reorder_img():
    # We need to test on a square array, as rotation does not change
    # shape, whereas reordering does.
    shape = (5, 5, 5, 2, 2)
    rng = np.random.RandomState(42)
    data = rng.uniform(size=shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    ref_img = Nifti1Image(data, affine)
    # Test with purely positive matrices and compare to a rotation
    for theta, phi in rng.randint(4, size=(5, 2)):
        rot = rotation(theta * np.pi / 2, phi * np.pi / 2)
        rot[np.abs(rot) < 0.001] = 0
        rot[rot > 0.9] = 1
        rot[rot < -0.9] = 1
        b = 0.5 * np.array(shape[:3])
        new_affine = from_matrix_vector(rot, b)
        rot_img = resample_img(ref_img, target_affine=new_affine)
        np.testing.assert_array_equal(rot_img.affine, new_affine)
        np.testing.assert_array_equal(get_data(rot_img).shape, shape)
        reordered_img = reorder_img(rot_img)
        np.testing.assert_array_equal(reordered_img.affine[:3, :3], np.eye(3))
        np.testing.assert_almost_equal(get_data(reordered_img), data)

    # Create a non-diagonal affine, and check that we raise a sensible
    # exception
    affine[1, 0] = 0.1
    ref_img = Nifti1Image(data, affine)
    with pytest.raises(ValueError, match="Cannot reorder the axes"):
        reorder_img(ref_img)

    # Test that no exception is raised when resample='continuous'
    reorder_img(ref_img, resample="continuous")

    # Test that resample args gets passed to resample_img
    interpolation = "nearest"
    reordered_img = reorder_img(ref_img, resample=interpolation)
    resampled_img = resample_img(
        ref_img,
        target_affine=reordered_img.affine,
        interpolation=interpolation,
    )
    np.testing.assert_array_equal(
        get_data(reordered_img), get_data(resampled_img)
    )

    # Make sure invalid resample argument is included in the error message
    interpolation = "an_invalid_interpolation"
    with pytest.raises(ValueError, match="interpolation must be one of"):
        reorder_img(ref_img, resample=interpolation)

    # Test flipping an axis
    data = rng.uniform(size=shape)
    for i in (0, 1, 2):
        # Make a diagonal affine with a negative axis, and check that
        # can be reordered, also vary the shape
        shape = (i + 1, i + 2, 3 - i)
        affine = np.eye(4)
        affine[i, i] *= -1
        img = Nifti1Image(data, affine)
        orig_img = copy.copy(img)
        # x, y, z = img.get_world_coords()
        # sample = img.values_in_world(x, y, z)
        img2 = reorder_img(img)
        # Check that img has not been changed
        np.testing.assert_array_equal(img.affine, orig_img.affine)
        np.testing.assert_array_equal(get_data(img), get_data(orig_img))
        # Test that the affine is indeed diagonal:
        np.testing.assert_array_equal(
            img2.affine[:3, :3], np.diag(np.diag(img2.affine[:3, :3]))
        )
        assert np.all(np.diag(img2.affine) >= 0)


def test_reorder_img_non_native_endianness():
    def _get_resampled_img(dtype):
        data = np.ones((10, 10, 10), dtype=dtype)
        data[3:7, 3:7, 3:7] = 2

        affine = np.eye(4)

        theta = math.pi / 6.0
        c = math.cos(theta)
        s = math.sin(theta)

        affine = np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
        )

        img = Nifti1Image(data, affine)
        return resample_img(img, target_affine=np.eye(4))

    img_1 = _get_resampled_img("<f8")
    img_2 = _get_resampled_img(">f8")

    np.testing.assert_equal(get_data(img_1), get_data(img_2))


def test_reorder_img_mirror():
    affine = np.array(
        [
            [-1.1, -0.0, 0.0, 0.0],
            [-0.0, -1.2, 0.0, 0.0],
            [-0.0, -0.0, 1.3, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = Nifti1Image(np.zeros((4, 6, 8)), affine=affine)
    reordered = reorder_img(img)
    np.testing.assert_allclose(
        get_bounds(reordered.shape, reordered.affine),
        get_bounds(img.shape, img.affine),
    )


def test_coord_transform_trivial():
    rng = np.random.RandomState(42)
    sform = np.eye(4)
    x = rng.random_sample((10,))
    y = rng.random_sample((10,))
    z = rng.random_sample((10,))

    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x, x_)
    np.testing.assert_array_equal(y, y_)
    np.testing.assert_array_equal(z, z_)

    sform[:, -1] = 1
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x + 1, x_)
    np.testing.assert_array_equal(y + 1, y_)
    np.testing.assert_array_equal(z + 1, z_)

    # Test the output in case of one item array
    x, y, z = x[:1], y[:1], z[:1]
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x + 1, x_)
    np.testing.assert_array_equal(y + 1, y_)
    np.testing.assert_array_equal(z + 1, z_)

    # Test the output in case of simple items
    x, y, z = x[0], y[0], z[0]
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x + 1, x_)
    np.testing.assert_array_equal(y + 1, y_)
    np.testing.assert_array_equal(z + 1, z_)

    # Test the outputs have the same shape as the inputs
    x = np.ones((3, 2, 4))
    y = np.ones((3, 2, 4))
    z = np.ones((3, 2, 4))
    x_, y_, z_ = coord_transform(x, y, z, sform)
    assert x.shape == x_.shape


@pytest.mark.skipif(
    not testing.is_64bit(), reason="This test only runs on 64bits machines."
)
@pytest.mark.skipif(
    os.environ.get("APPVEYOR") == "True",
    reason="This test too slow (7-8 minutes) on AppVeyor",
)
@pytest.mark.skipif(
    (
        os.environ.get("TRAVIS") == "true"
        and os.environ.get("TRAVIS_CPU_ARCH") == "arm64"
    ),
    reason="This test does not run on ARM arch.",
)
def test_resample_img_segmentation_fault():
    # see https://github.com/nilearn/nilearn/issues/346
    shape_in = (64, 64, 64)
    aff_in = np.diag([2.0, 2.0, 2.0, 1.0])
    aff_out = np.diag([3.0, 3.0, 3.0, 1.0])
    # fourth_dim = 1024 works fine but for 1025 creates a segmentation
    # fault with scipy < 0.14.1
    fourth_dim = 1025

    try:
        data = np.ones(shape_in + (fourth_dim,), dtype=np.float64)
    except MemoryError:
        # This can happen on AppVeyor and for 32-bit Python on Windows
        pytest.skip("Not enough RAM to run this test")
    else:
        img_in = Nifti1Image(data, aff_in)

        resample_img(img_in, target_affine=aff_out, interpolation="nearest")


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.uint8,
        np.uint16,
        np.uint32,
        np.float32,
        np.float64,
        float,
        ">i4",
        "<i4",
    ],
)
def test_resampling_with_int_types_no_crash(dtype):
    affine = np.eye(4)
    data = np.zeros((2, 2, 2))
    img = Nifti1Image(data.astype(dtype), affine)
    resample_img(img, target_affine=2.0 * affine)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "<i8", ">i8"])
@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_resampling_with_int64_types_no_crash(dtype):
    affine = np.eye(4)
    data = np.zeros((2, 2, 2))
    # Passing dtype or header is required when using int64
    # https://nipy.org/nibabel/changelog.html#api-changes-and-deprecations
    hdr = Nifti1Header()
    hdr.set_data_dtype(dtype)
    img = Nifti1Image(data.astype(dtype), affine, header=hdr)
    resample_img(img, target_affine=2.0 * affine)


def test_resample_input():
    rng = np.random.RandomState(42)
    shape = (3, 2, 5, 2)
    data = rng.randint(0, 10, shape, dtype="int32")
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    img = Nifti1Image(data, affine)

    with testing.write_tmp_imgs(img, create_files=True) as filename:
        filename = Path(filename)
        resample_img(filename, target_affine=affine, interpolation="nearest")


def test_smoke_resampling_non_nifti():
    rng = np.random.RandomState(42)
    shape = (3, 2, 5, 2)
    affine = np.eye(4)
    target_affine = 2 * affine
    data = rng.randint(0, 10, shape, dtype="int32")
    img = MGHImage(data, affine)

    resample_img(img, target_affine=target_affine, interpolation="nearest")
