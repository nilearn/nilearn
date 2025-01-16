"""Test the resampling code."""

import copy
import math
import os
from pathlib import Path

import numpy as np
import pytest
from nibabel import Nifti1Header, Nifti1Image
from nibabel.freesurfer import MGHImage
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from nilearn import _utils
from nilearn._utils import testing
from nilearn._utils.exceptions import DimensionError
from nilearn.image import get_data
from nilearn.image.image import _pad_array, crop_img
from nilearn.image.resampling import (
    BoundingBoxError,
    coord_transform,
    from_matrix_vector,
    get_bounds,
    get_mask_bounds,
    reorder_img,
    resample_img,
    resample_to_img,
)
from nilearn.image.tests._testing import match_headers_keys

ANGLES_TO_TEST = (0, np.pi, np.pi / 2.0, np.pi / 4.0, np.pi / 3.0)

SHAPE = (3, 2, 5, 2)


def rotation(theta, phi):
    """Return a rotation 3x3 matrix."""
    cos = np.cos
    sin = np.sin
    a1 = np.array(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
    )
    a2 = np.array(
        [[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]]
    )
    return np.dot(a1, a2)


@pytest.fixture
def shape():
    return SHAPE


@pytest.fixture
def data(rng, shape):
    return rng.random(shape)


def test_resample_deprecation_force_resample(data, shape, affine_eye):
    """Test change of value of force_resample."""
    affine_eye[:3, -1] = 0.5 * np.array(shape[:3])

    with pytest.warns(FutureWarning, match="force_resample"):
        resample_img(
            Nifti1Image(data, affine_eye),
            target_affine=affine_eye,
            interpolation="nearest",
            force_resample=None,
        )


@pytest.mark.parametrize("force_resample", [False, True])
def test_identity_resample(data, force_resample, shape, affine_eye):
    """Test resampling with an identity affine."""
    affine_eye[:3, -1] = 0.5 * np.array(shape[:3])

    rot_img = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=affine_eye,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    assert_almost_equal(data, get_data(rot_img))

    # Test with a 3x3 affine
    rot_img = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=affine_eye[:3, :3],
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    assert_almost_equal(data, get_data(rot_img))

    # Smoke-test with a list affine
    rot_img = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=affine_eye.tolist(),
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("endian_type", [">f8", "<f8"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear", "continuous"])
def test_identity_resample_non_native_endians(
    data, force_resample, shape, affine_eye, endian_type, interpolation
):
    """Test resampling with an identity affine with non native endians.

    with big endian data ('>f8')
    with little endian data ('<f8')
    """
    affine_eye[:3, -1] = 0.5 * np.array(shape[:3])

    rot_img = resample_img(
        Nifti1Image(data.astype(endian_type), affine_eye),
        target_affine=affine_eye.tolist(),
        interpolation=interpolation,
        force_resample=force_resample,
        copy_header=True,
    )

    assert_almost_equal(data, get_data(rot_img))


@pytest.mark.parametrize("force_resample", [False, True])
def test_downsample(data, force_resample, affine_eye):
    """Test resampling with a 1/2 down-sampling affine."""
    rot_img = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=2 * affine_eye,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]

    assert_almost_equal(downsampled, get_data(rot_img)[:x, :y, :z, ...])

    rot_img_2 = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=2 * affine_eye,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    assert_almost_equal(get_data(rot_img_2), get_data(rot_img))


@pytest.mark.parametrize("endian_type", [">f8", "<f8"])
@pytest.mark.parametrize("copy_data", [True, False])
@pytest.mark.parametrize("force_resample", [True, False])
def test_downsample_non_native_endian_data(
    data, affine_eye, endian_type, copy_data, force_resample
):
    """Test resampling with a 1/2 down-sampling affine with non native endians.

    Test to check that if giving non native endian data as input should
    work as normal and expected to return the same output as above tests.

    Big endian data ">f8"
    Little endian data "<f8"
    """
    rot_img = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=2 * affine_eye,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]

    assert_almost_equal(downsampled, get_data(rot_img)[:x, :y, :z, ...])

    rot_img = resample_img(
        Nifti1Image(data.astype(endian_type), affine_eye),
        target_affine=2 * affine_eye,
        interpolation="nearest",
        copy=copy_data,
        force_resample=force_resample,
        copy_header=True,
    )

    assert_almost_equal(downsampled, get_data(rot_img)[:x, :y, :z, ...])


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("shape", [(1, 4, 4), (1, 4, 4, 3)])
@pytest.mark.parametrize("value", [-3.75, 0])
def test_resampling_fill_value(data, affine_eye, value, force_resample):
    """Test resampling with a non-zero fill value.

    Check on 3D and 4D data.
    """
    angle = np.pi / 4
    rot = rotation(0, angle)

    if value:
        rot_img = resample_img(
            Nifti1Image(data, affine_eye),
            target_affine=rot,
            interpolation="nearest",
            fill_value=value,
            clip=False,
            force_resample=force_resample,
            copy_header=True,
        )
    else:
        rot_img = resample_img(
            Nifti1Image(data, affine_eye),
            target_affine=rot,
            interpolation="nearest",
            clip=False,
            force_resample=force_resample,
            copy_header=True,
        )

    assert get_data(rot_img).flatten()[0] == value

    rot_img2 = resample_to_img(
        Nifti1Image(data, affine_eye),
        rot_img,
        interpolation="nearest",
        fill_value=value,
        force_resample=force_resample,
        copy_header=True,
    )

    assert get_data(rot_img2).flatten()[0] == value


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("shape", [(1, 4, 4), (1, 4, 4, 3)])
@pytest.mark.parametrize("angle", ANGLES_TO_TEST)
def test_resampling_with_affine(data, affine_eye, angle, force_resample):
    """Test resampling with a given rotation part of the affine.

    Check on 3D and 4D data.
    """
    rot = rotation(0, angle)

    rot_img = resample_img(
        Nifti1Image(data, affine_eye),
        target_affine=rot,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    assert np.max(data) == np.max(get_data(rot_img))
    assert get_data(rot_img).dtype == data.dtype

    # We take the same rotation logic as above and test with nonnative endian
    # data as input
    img = Nifti1Image(data.astype(">f8"), affine_eye)
    rot = rotation(0, angle)

    rot_img = resample_img(
        img,
        target_affine=rot,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    assert np.max(data) == np.max(get_data(rot_img))


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("shape", [(1, 10, 10), (1, 10, 10, 3)])
@pytest.mark.parametrize("angle", (0, np.pi / 2.0, np.pi, 3 * np.pi / 2.0))
def test_resampling_continuous_with_affine(
    data, affine_eye, angle, force_resample
):
    rot = rotation(0, angle)
    img = Nifti1Image(data, affine_eye)

    rot_img = resample_img(
        img,
        target_affine=rot,
        interpolation="continuous",
        force_resample=force_resample,
        copy_header=True,
    )
    rot_img_back = resample_img(
        rot_img,
        target_affine=affine_eye,
        interpolation="continuous",
        force_resample=force_resample,
        copy_header=True,
    )

    center = slice(1, 9)
    # values on the edges are wrong for some reason
    mask = (0, center, center)
    assert_allclose(get_data(img)[mask], get_data(rot_img_back)[mask])

    assert get_data(rot_img).dtype == np.dtype(
        data.dtype.name.replace("int", "float")
    )


@pytest.mark.parametrize("force_resample", [False, True])
def test_resampling_error_checks(tmp_path, force_resample, data, affine_eye):
    img = Nifti1Image(data, affine_eye)
    target_shape = (5, 3, 2)

    # Correct parameters: no exception
    resample_img(
        img,
        target_shape=target_shape,
        target_affine=affine_eye,
        force_resample=force_resample,
        copy_header=True,
    )
    resample_img(
        img,
        target_affine=affine_eye,
        force_resample=force_resample,
        copy_header=True,
    )

    filename = testing.write_imgs_to_path(img, file_path=tmp_path)
    resample_img(
        filename,
        target_shape=target_shape,
        target_affine=affine_eye,
        force_resample=force_resample,
        copy_header=True,
    )

    # Missing parameter
    with pytest.raises(ValueError, match="target_affine should be specified"):
        resample_img(
            img,
            target_shape=target_shape,
            force_resample=force_resample,
            copy_header=True,
        )

    # Invalid shape
    with pytest.raises(ValueError, match="shape .* should be .* 3D grid"):
        resample_img(
            img,
            target_shape=(2, 3),
            target_affine=affine_eye,
            force_resample=force_resample,
            copy_header=True,
        )

    # Invalid interpolation
    with pytest.raises(ValueError, match="interpolation must be one of"):
        resample_img(
            img,
            target_shape=target_shape,
            target_affine=affine_eye,
            interpolation="an_invalid_interpolation",
            force_resample=force_resample,
            copy_header=True,
        )


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("target_shape", [None, (3, 2, 5)])
def test_resampling_copy_has_no_shared_memory(
    target_shape, force_resample, data, affine_eye
):
    """copy=true guarantees output array shares no memory with input array."""
    img = Nifti1Image(data, affine_eye)
    target_affine = None if target_shape is None else affine_eye

    img_r = resample_img(
        img,
        target_affine=target_affine,
        target_shape=target_shape,
        copy=False,
        force_resample=force_resample,
        copy_header=True,
    )

    assert img_r == img

    img_r = resample_img(
        img,
        target_affine=target_affine,
        target_shape=target_shape,
        copy=True,
        force_resample=force_resample,
        copy_header=True,
    )

    assert not np.may_share_memory(get_data(img_r), get_data(img))
    assert_almost_equal(get_data(img_r), get_data(img))
    assert_almost_equal(img_r.affine, img.affine)


@pytest.mark.parametrize(
    "force_resample",
    [False, True],
)
def test_resampling_warning_s_form(data, affine_eye, force_resample):
    img_no_sform = Nifti1Image(data, affine_eye)
    img_no_sform.set_sform(None)

    with pytest.warns(Warning, match="The provided image has no sform"):
        resample_img(
            img_no_sform,
            target_affine=affine_eye,
            force_resample=force_resample,
            copy_header=True,
        )


@pytest.mark.parametrize("force_resample", [False, True])
def test_resampling_warning_binary_image(affine_eye, rng, force_resample):
    # Resampling a binary image with continuous or
    # linear interpolation should raise a warning.
    data_binary = rng.integers(4, size=(1, 4, 4), dtype="int32")
    data_binary[data_binary > 0] = 1

    assert sorted(np.unique(data_binary)) == [0, 1]

    rot = rotation(0, np.pi / 4)
    img_binary = Nifti1Image(data_binary, affine_eye)

    assert _utils.niimg.is_binary_niimg(img_binary)

    with pytest.warns(Warning, match="Resampling binary images with"):
        resample_img(
            img_binary,
            target_affine=rot,
            interpolation="continuous",
            force_resample=force_resample,
            copy_header=True,
        )

    with pytest.warns(Warning, match="Resampling binary images with"):
        resample_img(
            img_binary,
            target_affine=rot,
            interpolation="linear",
            force_resample=force_resample,
            copy_header=True,
        )


@pytest.mark.parametrize("force_resample", [False, True])
def test_resample_img_copied_header(img_4d_mni_tr2, force_resample):
    # Test that the header is copied when resampling
    result = resample_img(
        img_4d_mni_tr2,
        target_affine=np.diag((6, 6, 6)),
        copy_header=True,
        force_resample=force_resample,
    )
    # pixdim[1:4] should change to [6, 6, 6]
    assert (result.header["pixdim"][1:4] == np.array([6, 6, 6])).all()
    # pixdim at other indices should remain the same
    assert (
        result.header["pixdim"][4:] == img_4d_mni_tr2.header["pixdim"][4:]
    ).all()
    assert result.header["pixdim"][0] == img_4d_mni_tr2.header["pixdim"][0]
    # dim, srow_* and min/max should also change
    match_headers_keys(
        img_4d_mni_tr2,
        result,
        except_keys=[
            "pixdim",
            "dim",
            "cal_max",
            "cal_min",
            "srow_x",
            "srow_y",
            "srow_z",
        ],
    )


@pytest.mark.parametrize("force_resample", [False, True])
def test_4d_affine_bounding_box_error(affine_eye, force_resample):
    bigger_data = np.zeros([10, 10, 10])
    bigger_img = Nifti1Image(bigger_data, affine_eye)

    small_data = np.ones([4, 4, 4])
    small_data_4D_affine = affine_eye
    small_data_4D_affine[:3, -1] = np.array([5, 4, 5])
    small_img = Nifti1Image(small_data, small_data_4D_affine)

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
        force_resample=force_resample,
        copy_header=True,
    )
    # resample using 3D affine and no target shape
    small_to_big_without_shape_3D_affine = resample_img(
        small_img,
        target_affine=bigger_img.affine[:3, :3],
        copy_header=True,
        force_resample=force_resample,
    )
    # resample using 4D affine and no target shape
    small_to_big_without_shape = resample_img(
        small_img,
        target_affine=bigger_img.affine,
        copy_header=True,
        force_resample=force_resample,
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


@pytest.mark.parametrize("force_resample", [False, True])
def test_raises_upon_3x3_affine_and_no_shape(affine_eye, force_resample):
    img = Nifti1Image(np.zeros([8, 9, 10]), affine=affine_eye)
    message = (
        "Given target shape without anchor "
        "vector: Affine shape should be \\(4, 4\\) and "
        "not \\(3, 3\\)"
    )
    with pytest.raises(ValueError, match=message):
        resample_img(
            img,
            target_affine=np.eye(3) * 2,
            target_shape=(10, 10, 10),
            force_resample=force_resample,
            copy_header=True,
        )


@pytest.mark.parametrize("force_resample", [False, True])
def test_3x3_affine_bbox(affine_eye, force_resample):
    """Test that the bounding-box is properly computed when \
    transforming with a negative affine component.

    This is specifically to test for a change in behavior between
    scipy < 0.18 and scipy >= 0.18, which is an interaction between
    offset and a diagonal affine
    """
    image = np.ones((20, 30))
    source_affine = affine_eye
    # Give the affine an offset
    source_affine[:2, 3] = np.array([96, 64])

    # We need to turn this data into a nibabel image
    img = Nifti1Image(image[:, :, np.newaxis], affine=source_affine)

    target_affine_3x3 = np.eye(3) * 2
    # One negative axes
    target_affine_3x3[1] *= -1

    img_3d_affine = resample_img(
        img,
        target_affine=target_affine_3x3,
        force_resample=force_resample,
        copy_header=True,
    )

    # If the bounding box is computed wrong, the image will be only zeros
    assert_allclose(get_data(img_3d_affine).max(), image.max())


@pytest.mark.parametrize("force_resample", [False, True])
def test_raises_bbox_error_if_data_outside_box(affine_eye, force_resample):
    """Make some cases which should raise exceptions."""
    # original image
    data = np.zeros([8, 9, 10])
    affine_offset = np.array([1, 1, 1])
    affine = affine_eye
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
            resample_img(
                img,
                target_affine=new_affine,
                force_resample=force_resample,
                copy_header=True,
            )


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize(
    "axis_permutation", [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]]
)
def test_resampling_result_axis_permutation(
    affine_eye, axis_permutation, force_resample
):
    """Transform real data using easily checkable transformations.

    For now: axis permutations
    create a cuboid full of deterministic data, padded with one
    voxel thickness of zeros
    """
    core_shape = (3, 5, 4)
    core_data = np.arange(np.prod(core_shape)).reshape(core_shape)
    full_data_shape = np.array(core_shape) + 2
    full_data = np.zeros(full_data_shape)
    full_data[tuple(slice(1, 1 + s) for s in core_shape)] = core_data

    source_img = Nifti1Image(full_data, affine_eye)

    # check 3x3 transformation matrix
    target_affine = np.eye(3)[axis_permutation]

    resampled_img = resample_img(
        source_img,
        target_affine=target_affine,
        force_resample=force_resample,
        copy_header=True,
    )

    resampled_data = get_data(resampled_img)
    expected_data = full_data.transpose(axis_permutation)
    assert_array_almost_equal(resampled_data, expected_data)

    # check 4x4 transformation matrix
    offset = np.array([-2, 1, -3])

    target_affine = affine_eye
    target_affine[:3, :3] = np.eye(3)[axis_permutation]
    target_affine[:3, 3] = offset

    resampled_img = resample_img(
        source_img,
        target_affine=target_affine,
        force_resample=force_resample,
        copy_header=True,
    )

    resampled_data = get_data(resampled_img)
    offset_cropping = (
        np.vstack([-offset[axis_permutation][np.newaxis, :], np.zeros([1, 3])])
        .T.ravel()
        .astype(int)
    )
    expected_data = _pad_array(
        full_data.transpose(axis_permutation), list(offset_cropping)
    )
    assert_array_almost_equal(resampled_data, expected_data)


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("core_shape", [(3, 5, 4), (3, 5, 4, 2)])
def test_resampling_nan(affine_eye, core_shape, force_resample):
    """Test that when the data has NaNs they do not propagate to the \
    whole image.
    """
    # create deterministic data, padded with one
    # voxel thickness of zeros
    core_data = (
        np.arange(np.prod(core_shape)).reshape(core_shape).astype(np.float64)
    )
    # Introduce a nan
    core_data[2, 2:4, 1] = np.nan
    full_data_shape = np.array(core_shape) + 2
    full_data = np.zeros(full_data_shape)
    full_data[tuple(slice(1, 1 + s) for s in core_shape)] = core_data

    source_img = Nifti1Image(full_data, affine_eye)

    # Transform real data using easily checkable transformations
    # For now: axis permutations
    axis_permutation = [0, 1, 2]

    # check 3x3 transformation matrix
    target_affine = np.eye(3)[axis_permutation]
    resampled_img = resample_img(
        source_img,
        target_affine=target_affine,
        force_resample=force_resample,
        copy_header=True,
    )

    resampled_data = get_data(resampled_img)
    if full_data.ndim == 4:
        axis_permutation.append(3)
    expected_data = full_data.transpose(axis_permutation)
    non_nan = np.isfinite(expected_data)

    # Check that the input data hasn't been modified:
    assert not np.all(non_nan)

    # Check that for finite value resampling works without problems
    assert_array_almost_equal(resampled_data[non_nan], expected_data[non_nan])

    # Check that what was not finite is still not finite
    assert not np.any(np.isfinite(resampled_data[np.logical_not(non_nan)]))


@pytest.mark.parametrize("force_resample", [False, True])
def test_resampling_nan_big(affine_eye, force_resample):
    """Test with an actual resampling, in the case of a bigish hole.

    This checks the extrapolation mechanism: if we don't do any
    extrapolation before resampling, the hole creates big
    artifacts
    """
    data = 10 * np.ones((10, 10, 10))
    data[4:6, 4:6, 4:6] = np.nan
    source_img = Nifti1Image(data, 2 * affine_eye)

    with pytest.warns(RuntimeWarning):
        resampled_img = resample_img(
            source_img,
            target_affine=affine_eye,
            force_resample=force_resample,
            copy_header=True,
        )

    resampled_data = get_data(resampled_img)
    assert_allclose(10, resampled_data[np.isfinite(resampled_data)])


@pytest.mark.parametrize("force_resample", [False, True])
def test_resample_to_img(data, affine_eye, force_resample):
    source_affine = affine_eye
    source_img = Nifti1Image(data, source_affine)

    target_affine = 2 * affine_eye
    target_img = Nifti1Image(data, target_affine)

    result_img = resample_to_img(
        source_img,
        target_img,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    assert_almost_equal(downsampled, get_data(result_img)[:x, :y, :z, ...])


def test_crop(affine_eye):
    # Testing that padding of arrays and cropping of images work symmetrically
    shape = (4, 6, 2)
    data = np.ones(shape)
    padded = _pad_array(data, [3, 2, 4, 4, 5, 7])
    padd_nii = Nifti1Image(padded, affine_eye)

    cropped = crop_img(padd_nii, pad=False, copy_header=True)

    assert_equal(get_data(cropped), data)


@pytest.mark.parametrize("force_resample", [False, True])
def test_resample_identify_affine_int_translation(
    affine_eye, rng, force_resample
):
    source_shape = (6, 4, 6)
    source_affine = affine_eye
    source_affine[:, 3] = np.append(rng.integers(0, 4, 3), 1)
    source_data = rng.random(source_shape)
    source_img = Nifti1Image(source_data, source_affine)

    target_shape = (11, 10, 9)
    target_data = np.zeros(target_shape)
    target_affine = source_affine
    target_affine[:3, 3] -= 3  # add an offset of 3 in x, y, z
    target_data[3:9, 3:7, 3:9] = (
        source_data  # put the data at the offset location
    )
    target_img = Nifti1Image(target_data, target_affine)

    result_img = resample_to_img(
        source_img,
        target_img,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )
    assert_almost_equal(get_data(target_img), get_data(result_img))

    result_img_2 = resample_to_img(
        result_img,
        source_img,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )
    assert_almost_equal(get_data(source_img), get_data(result_img_2))

    result_img_3 = resample_to_img(
        result_img,
        source_img,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )
    assert_almost_equal(get_data(result_img_2), get_data(result_img_3))

    result_img_4 = resample_to_img(
        source_img,
        target_img,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )
    assert_almost_equal(get_data(target_img), get_data(result_img_4))


@pytest.mark.parametrize("force_resample", [False, True])
def test_resample_clip(affine_eye, force_resample):
    # Resample and image and get larger and smaller
    # value than in the original. Use clip to get rid of these images

    shape = (6, 3, 6)
    data = np.zeros(shape=shape)
    data[1:-2, 1:-1, 1:-2] = 1

    source_affine = np.diag((2, 2, 2, 1))
    source_img = Nifti1Image(data, source_affine)

    no_clip_data = get_data(
        resample_img(
            source_img,
            target_affine=affine_eye,
            clip=False,
            force_resample=force_resample,
            copy_header=True,
        )
    )
    clip_data = get_data(
        resample_img(
            source_img,
            target_affine=affine_eye,
            clip=True,
            force_resample=force_resample,
            copy_header=True,
        )
    )

    not_clip = np.where(
        (no_clip_data > data.min()) & (no_clip_data < data.max())
    )

    assert np.any(no_clip_data > data.max())
    assert np.any(no_clip_data < data.min())
    assert np.all(clip_data <= data.max())
    assert np.all(clip_data >= data.min())
    assert_array_equal(no_clip_data[not_clip], clip_data[not_clip])


@pytest.mark.parametrize("force_resample", [False, True])
def test_reorder_img(affine_eye, rng, force_resample):
    # We need to test on a square array, as rotation does not change
    # shape, whereas reordering does.
    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)
    affine = affine_eye
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    ref_img = Nifti1Image(data, affine)

    # Test with purely positive matrices and compare to a rotation
    for theta, phi in rng.integers(4, size=(5, 2)):
        rot = rotation(theta * np.pi / 2, phi * np.pi / 2)
        rot[np.abs(rot) < 0.001] = 0
        rot[rot > 0.9] = 1
        rot[rot < -0.9] = 1
        b = 0.5 * np.array(shape[:3])
        new_affine = from_matrix_vector(rot, b)

        rot_img = resample_img(
            ref_img,
            target_affine=new_affine,
            force_resample=force_resample,
            copy_header=True,
        )

        assert_array_equal(rot_img.affine, new_affine)
        assert_array_equal(get_data(rot_img).shape, shape)

        reordered_img = reorder_img(rot_img, copy_header=True)

        assert_array_equal(reordered_img.affine[:3, :3], np.eye(3))
        assert_almost_equal(get_data(reordered_img), data)


@pytest.mark.parametrize("force_resample", [False, True])
def test_reorder_img_with_resample_arg(affine_eye, rng, force_resample):
    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)
    affine = affine_eye
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    ref_img = Nifti1Image(data, affine)

    interpolation = "nearest"

    reordered_img = reorder_img(
        ref_img, resample=interpolation, copy_header=True
    )

    resampled_img = resample_img(
        ref_img,
        target_affine=reordered_img.affine,
        interpolation=interpolation,
        force_resample=force_resample,
        copy_header=True,
    )
    assert_array_equal(get_data(reordered_img), get_data(resampled_img))


def test_reorder_img_error_reorder_axis(affine_eye, rng):
    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)

    # Create a non-diagonal affine,
    # and check that we raise a sensible exception
    non_diagonal_affine = affine_eye
    non_diagonal_affine[1, 0] = 0.1
    ref_img = Nifti1Image(data, non_diagonal_affine)

    # Test that no exception is raised when resample='continuous'
    reorder_img(ref_img, resample="continuous", copy_header=True)

    with pytest.raises(ValueError, match="Cannot reorder the axes"):
        reorder_img(ref_img, copy_header=True)


def test_reorder_img_flipping_axis(affine_eye, rng):
    shape = (5, 5, 5, 2, 2)

    data = rng.uniform(size=shape)

    for i in (0, 1, 2):
        # Make a diagonal affine with a negative axis, and check that
        # can be reordered, also vary the shape
        shape = (i + 1, i + 2, 3 - i)
        affine = affine_eye
        affine[i, i] *= -1

        img = Nifti1Image(data, affine)
        orig_img = copy.copy(img)

        img2 = reorder_img(img, copy_header=True)

        # Check that img has not been changed
        assert_array_equal(img.affine, orig_img.affine)
        assert_array_equal(get_data(img), get_data(orig_img))

        # Test that the affine is indeed diagonal:
        assert_array_equal(
            img2.affine[:3, :3], np.diag(np.diag(img2.affine[:3, :3]))
        )
        assert np.all(np.diag(img2.affine) >= 0)


def test_reorder_img_error_interpolation(affine_eye, rng):
    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)
    affine = affine_eye
    affine[1, 0] = 0.1
    ref_img = Nifti1Image(data, affine)

    with pytest.raises(ValueError, match="interpolation must be one of"):
        reorder_img(
            ref_img, resample="an_invalid_interpolation", copy_header=True
        )


@pytest.mark.parametrize("force_resample", [False, True])
def test_reorder_img_non_native_endianness(force_resample):
    def _get_resampled_img(dtype):
        data = np.ones((10, 10, 10), dtype=dtype)
        data[3:7, 3:7, 3:7] = 2

        theta = math.pi / 6.0
        c = math.cos(theta)
        s = math.sin(theta)

        affine = np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
        )

        img = Nifti1Image(data, affine)
        return resample_img(
            img,
            target_affine=affine,
            force_resample=force_resample,
            copy_header=True,
        )

    img_1 = _get_resampled_img("<f8")
    img_2 = _get_resampled_img(">f8")

    assert_equal(get_data(img_1), get_data(img_2))


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
    reordered = reorder_img(img, copy_header=True)
    assert_allclose(
        get_bounds(reordered.shape, reordered.affine),
        get_bounds(img.shape, img.affine),
    )


def test_reorder_img_copied_header(img_4d_mni_tr2):
    # Test that the header is copied when reordering
    result = reorder_img(img_4d_mni_tr2, copy_header=True)
    # all header fields should stay the same
    match_headers_keys(
        img_4d_mni_tr2,
        result,
        except_keys=[],
    )


@pytest.mark.parametrize(
    "func, input_img",
    [
        (resample_img, "img_4d_mni_tr2"),
        (reorder_img, "img_4d_mni_tr2"),
    ],
)
def test_warning_copy_header_false(request, func, input_img):
    # Use the request fixture to get the actual fixture value
    actual_input_img = request.getfixturevalue(input_img)
    with pytest.warns(FutureWarning, match="From release 0.13.0 onwards*"):
        func(actual_input_img, copy_header=False)


def test_coord_transform_trivial(affine_eye, rng):
    sform = affine_eye
    x = rng.random((10,))
    y = rng.random((10,))
    z = rng.random((10,))

    x_, y_, z_ = coord_transform(x, y, z, sform)
    assert_array_equal(x, x_)
    assert_array_equal(y, y_)
    assert_array_equal(z, z_)

    sform[:, -1] = 1
    x_, y_, z_ = coord_transform(x, y, z, sform)
    assert_array_equal(x + 1, x_)
    assert_array_equal(y + 1, y_)
    assert_array_equal(z + 1, z_)

    # Test the output in case of one item array
    x, y, z = x[:1], y[:1], z[:1]
    x_, y_, z_ = coord_transform(x, y, z, sform)
    assert_array_equal(x + 1, x_)
    assert_array_equal(y + 1, y_)
    assert_array_equal(z + 1, z_)

    # Test the output in case of simple items
    x, y, z = x[0], y[0], z[0]
    x_, y_, z_ = coord_transform(x, y, z, sform)
    assert_array_equal(x + 1, x_)
    assert_array_equal(y + 1, y_)
    assert_array_equal(z + 1, z_)

    # Test the outputs have the same shape as the inputs
    x = np.ones((3, 2, 4))
    y = np.ones((3, 2, 4))
    z = np.ones((3, 2, 4))
    x_, y_, z_ = coord_transform(x, y, z, sform)
    assert x.shape == x_.shape


#  TODO "This test does not run on ARM arch.",
@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.skipif(
    not testing.is_64bit(), reason="This test only runs on 64bits machines."
)
@pytest.mark.skipif(
    os.environ.get("APPVEYOR") == "True",
    reason="This test too slow (7-8 minutes) on AppVeyor",
)
def test_resample_img_segmentation_fault(force_resample):
    # see https://github.com/nilearn/nilearn/issues/346
    shape_in = (64, 64, 64)
    aff_in = np.diag([2.0, 2.0, 2.0, 1.0])
    aff_out = np.diag([3.0, 3.0, 3.0, 1.0])
    # fourth_dim = 1024 works fine but for 1025 creates a segmentation
    # fault with scipy < 0.14.1
    fourth_dim = 1025

    try:
        data = np.ones((*shape_in, fourth_dim), dtype=np.float64)
    except MemoryError:
        # This can happen on AppVeyor and for 32-bit Python on Windows
        pytest.skip("Not enough RAM to run this test")
    else:
        img_in = Nifti1Image(data, aff_in)

        resample_img(
            img_in,
            target_affine=aff_out,
            interpolation="nearest",
            force_resample=force_resample,
            copy_header=True,
        )


@pytest.mark.parametrize("force_resample", [False, True])
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
def test_resampling_with_int_types_no_crash(affine_eye, dtype, force_resample):
    data = np.zeros((2, 2, 2))
    img = Nifti1Image(data.astype(dtype), affine_eye)
    resample_img(
        img,
        target_affine=2.0 * affine_eye,
        force_resample=force_resample,
        copy_header=True,
    )


@pytest.mark.parametrize("force_resample", [False, True])
@pytest.mark.parametrize("dtype", ["int64", "uint64", "<i8", ">i8"])
@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_resampling_with_int64_types_no_crash(
    affine_eye, dtype, force_resample
):
    data = np.zeros((2, 2, 2))
    # Passing dtype or header is required when using int64
    # https://nipy.org/nibabel/changelog.html#api-changes-and-deprecations
    hdr = Nifti1Header()
    hdr.set_data_dtype(dtype)
    img = Nifti1Image(data.astype(dtype), affine_eye, header=hdr)
    resample_img(
        img,
        target_affine=2.0 * affine_eye,
        force_resample=force_resample,
        copy_header=True,
    )


@pytest.mark.parametrize("force_resample", [False, True])
def test_resample_input(affine_eye, shape, rng, tmp_path, force_resample):
    data = rng.integers(0, 10, shape, dtype="int32")
    affine = affine_eye
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    img = Nifti1Image(data, affine)

    filename = testing.write_imgs_to_path(
        img, file_path=tmp_path, create_files=True
    )
    filename = Path(filename)
    resample_img(
        filename,
        target_affine=affine,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )


@pytest.mark.parametrize("force_resample", [False, True])
def test_smoke_resampling_non_nifti(affine_eye, shape, rng, force_resample):
    target_affine = 2 * affine_eye
    data = rng.integers(0, 10, shape, dtype="int32")
    img = MGHImage(data, affine_eye)

    resample_img(
        img,
        target_affine=target_affine,
        interpolation="nearest",
        force_resample=force_resample,
        copy_header=True,
    )


@pytest.mark.parametrize("shape", [(1, 4, 4)])
def test_get_mask_bounds(data, affine_eye):
    img = Nifti1Image(data, affine_eye)
    assert_allclose((0.0, 0.0, 0.0, 3.0, 0.0, 3.0), get_mask_bounds(img))


def test_get_mask_bounds_error(data, affine_eye):
    with pytest.raises(TypeError, match="Data given cannot be loaded because"):
        get_mask_bounds(None)

    with pytest.raises(
        DimensionError, match="Expected dimension is 3D and you provided a 4D"
    ):
        img = Nifti1Image(data, affine_eye)
        get_mask_bounds(img)
