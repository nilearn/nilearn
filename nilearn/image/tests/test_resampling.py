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
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

AFFINE_EYE = np.eye(4)

ANGLES_TO_TEST = (0, np.pi, np.pi / 2.0, np.pi / 4.0, np.pi / 3.0)

SHAPE = (3, 2, 5, 2)


def _make_resampling_test_data():
    rng = np.random.RandomState(42)
    shape = SHAPE
    affine = AFFINE_EYE
    data = rng.randint(0, 10, shape, dtype="int32")
    img = Nifti1Image(data, affine)
    return img, affine, data


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


@pytest.fixture
def shape():
    return SHAPE


@pytest.fixture
def affine():
    return AFFINE_EYE


def test_identity_resample(shape, affine):
    """Test resampling with an identity affine."""
    rng = np.random.RandomState(42)
    data = rng.randint(0, 10, shape, dtype="int32")
    affine[:3, -1] = 0.5 * np.array(shape[:3])

    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=affine,
        interpolation="nearest",
    )

    assert_almost_equal(data, get_data(rot_img))

    # Test with a 3x3 affine
    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=affine[:3, :3],
        interpolation="nearest",
    )

    assert_almost_equal(data, get_data(rot_img))

    # Smoke-test with a list affine
    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=affine.tolist(),
        interpolation="nearest",
    )


@pytest.mark.parametrize("endian_type", [">f8", "<f8"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear", "continuous"])
def test_identity_resample_non_native_endians(
    shape, affine, endian_type, interpolation
):
    """Test resampling with an identity affine with non native endians

    with big endian data ('>f8')
    with little endian data ('<f8')
    """
    rng = np.random.RandomState(42)
    data = rng.randint(0, 10, shape, dtype="int32")
    affine[:3, -1] = 0.5 * np.array(shape[:3])

    rot_img = resample_img(
        Nifti1Image(data.astype(endian_type), affine),
        target_affine=affine.tolist(),
        interpolation=interpolation,
    )

    assert_almost_equal(data, get_data(rot_img))


def test_downsample(shape, affine):
    """Test resampling with a 1/2 down-sampling affine."""
    rng = np.random.RandomState(42)
    data = rng.random_sample(shape)

    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=2 * affine,
        interpolation="nearest",
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]

    assert_almost_equal(downsampled, get_data(rot_img)[:x, :y, :z, ...])

    rot_img_2 = resample_img(
        Nifti1Image(data, affine),
        target_affine=2 * affine,
        interpolation="nearest",
        force_resample=True,
    )

    assert_almost_equal(get_data(rot_img_2), get_data(rot_img))


@pytest.mark.parametrize("endian_type", [">f8", "<f8"])
@pytest.mark.parametrize("copy_data", [True, False])
def test_downsample_non_native_endian_data(
    shape, affine, endian_type, copy_data
):
    """Test resampling with a 1/2 down-sampling affine with non native endians.

    Test to check that if giving non native endian data as input should
    work as normal and expected to return the same output as above tests.

    Big endian data ">f8"
    Little endian data "<f8"
    """
    rng = np.random.RandomState(42)
    data = rng.random_sample(shape)

    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=2 * affine,
        interpolation="nearest",
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]

    assert_almost_equal(downsampled, get_data(rot_img)[:x, :y, :z, ...])

    rot_img = resample_img(
        Nifti1Image(data.astype(endian_type), affine),
        target_affine=2 * affine,
        interpolation="nearest",
        copy=copy_data,
    )

    assert_almost_equal(downsampled, get_data(rot_img)[:x, :y, :z, ...])


@pytest.mark.parametrize("shape", [(1, 4, 4), (1, 4, 4, 3)])
@pytest.mark.parametrize("value", [-3.75, 0])
def test_resampling_fill_value(affine, shape, value):
    """Test resampling with a non-zero fill value

    Check on 3D and 4D data.
    """
    rng = np.random.RandomState(42)

    data = rng.uniform(size=shape)

    angle = np.pi / 4
    rot = rotation(0, angle)

    if value:
        rot_img = resample_img(
            Nifti1Image(data, affine),
            target_affine=rot,
            interpolation="nearest",
            fill_value=value,
            clip=False,
        )
    else:
        rot_img = resample_img(
            Nifti1Image(data, affine),
            target_affine=rot,
            interpolation="nearest",
            clip=False,
        )

    assert get_data(rot_img).flatten()[0] == value

    rot_img2 = resample_to_img(
        Nifti1Image(data, affine),
        rot_img,
        interpolation="nearest",
        fill_value=value,
    )

    assert get_data(rot_img2).flatten()[0] == value


@pytest.mark.parametrize("shape", [(1, 4, 4), (1, 4, 4, 3)])
@pytest.mark.parametrize("angle", ANGLES_TO_TEST)
def test_resampling_with_affine(affine, shape, angle):
    """Test resampling with a given rotation part of the affine.

    Check on 3D and 4D data.
    """
    rng = np.random.RandomState(42)

    data = rng.randint(4, size=shape, dtype="int32")
    rot = rotation(0, angle)

    rot_img = resample_img(
        Nifti1Image(data, affine),
        target_affine=rot,
        interpolation="nearest",
    )

    assert np.max(data) == np.max(get_data(rot_img))
    assert get_data(rot_img).dtype == data.dtype

    # We take the same rotation logic as above and test with nonnative endian
    # data as input
    img = Nifti1Image(data.astype(">f8"), affine)
    rot = rotation(0, angle)

    rot_img = resample_img(img, target_affine=rot, interpolation="nearest")

    assert np.max(data) == np.max(get_data(rot_img))


@pytest.mark.parametrize("shape", [(1, 10, 10), (1, 10, 10, 3)])
@pytest.mark.parametrize("angle", (0, np.pi / 2.0, np.pi, 3 * np.pi / 2.0))
def test_resampling_continuous_with_affine(affine, shape, angle):
    rng = np.random.RandomState(42)

    data = rng.randint(1, 4, size=shape, dtype="int32")
    rot = rotation(0, angle)
    img = Nifti1Image(data, affine)

    rot_img = resample_img(img, target_affine=rot, interpolation="continuous")
    rot_img_back = resample_img(
        rot_img, target_affine=affine, interpolation="continuous"
    )

    center = slice(1, 9)
    # values on the edges are wrong for some reason
    mask = (0, center, center)
    assert_allclose(get_data(img)[mask], get_data(rot_img_back)[mask])

    assert get_data(rot_img).dtype == np.dtype(
        data.dtype.name.replace("int", "float")
    )


def test_resampling_error_checks():
    img, affine, _ = _make_resampling_test_data()
    target_shape = (5, 3, 2)

    # Correct parameters: no exception
    resample_img(img, target_shape=target_shape, target_affine=affine)
    resample_img(img, target_affine=affine)

    with testing.write_tmp_imgs(img) as filename:
        resample_img(filename, target_shape=target_shape, target_affine=affine)

    # Missing parameter
    with pytest.raises(ValueError, match="target_affine should be specified"):
        resample_img(img, target_shape=target_shape)

    # Invalid shape
    with pytest.raises(ValueError, match="shape .* should be .* 3D grid"):
        resample_img(
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


@pytest.mark.parametrize("target_shape", [None, (3, 2, 5)])
def test_resampling_copy_has_no_shared_memory(target_shape):
    """copy=true guarantees output array shares no memory with input array."""
    img, affine, _ = _make_resampling_test_data()
    target_affine = None if target_shape is None else affine

    img_r = resample_img(
        img, target_affine=target_affine, target_shape=target_shape, copy=False
    )

    assert img_r == img

    img_r = resample_img(
        img, target_affine=target_affine, target_shape=target_shape, copy=True
    )

    assert not np.may_share_memory(get_data(img_r), get_data(img))
    assert_almost_equal(get_data(img_r), get_data(img))
    assert_almost_equal(img_r.affine, img.affine)


def test_resampling_warning_s_form(shape):
    rng = np.random.RandomState(42)

    affine = np.eye(4)

    data = rng.randint(0, 10, shape, dtype="int32")
    img_no_sform = Nifti1Image(data, affine)
    img_no_sform.set_sform(None)

    with pytest.warns(Warning, match="The provided image has no sform"):
        resample_img(img_no_sform, target_affine=affine)


def test_resampling_warning_binary_image(affine):
    rng = np.random.RandomState(42)

    # Resampling a binary image with continuous or
    # linear interpolation should raise a warning.
    data_binary = rng.randint(4, size=(1, 4, 4), dtype="int32")
    data_binary[data_binary > 0] = 1

    assert sorted(list(np.unique(data_binary))) == [0, 1]

    rot = rotation(0, np.pi / 4)
    img_binary = Nifti1Image(data_binary, affine)

    assert _utils.niimg._is_binary_niimg(img_binary)

    with pytest.warns(Warning, match="Resampling binary images with"):
        resample_img(img_binary, target_affine=rot, interpolation="continuous")

    with pytest.warns(Warning, match="Resampling binary images with"):
        resample_img(img_binary, target_affine=rot, interpolation="linear")


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


def test_raises_upon_3x3_affine_and_no_shape(affine):
    img = Nifti1Image(np.zeros([8, 9, 10]), affine=affine)
    message = (
        "Given target shape without anchor "
        "vector: Affine shape should be \\(4, 4\\) and "
        "not \\(3, 3\\)"
    )
    with pytest.raises(ValueError, match=message):
        resample_img(
            img, target_affine=np.eye(3) * 2, target_shape=(10, 10, 10)
        )


def test_3x3_affine_bbox(affine):
    """Test that the bounding-box is properly computed when \
    transforming with a negative affine component.

    This is specifically to test for a change in behavior between
    scipy < 0.18 and scipy >= 0.18, which is an interaction between
    offset and a diagonal affine
    """
    image = np.ones((20, 30))
    source_affine = affine
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
    assert_allclose(get_data(img_3d_affine).max(), image.max())


def test_raises_bbox_error_if_data_outside_box(affine):
    """Make some cases which should raise exceptions"""
    # original image
    data = np.zeros([8, 9, 10])
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


@pytest.mark.parametrize(
    "axis_permutation", [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]]
)
def test_resampling_result_axis_permutation(axis_permutation):
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

    source_img = Nifti1Image(full_data, np.eye(4))

    # check 3x3 transformation matrix
    target_affine = np.eye(3)[axis_permutation]

    resampled_img = resample_img(source_img, target_affine=target_affine)

    resampled_data = get_data(resampled_img)
    expected_data = full_data.transpose(axis_permutation)
    assert_array_almost_equal(resampled_data, expected_data)

    # check 4x4 transformation matrix
    offset = np.array([-2, 1, -3])

    target_affine = np.eye(4)
    target_affine[:3, :3] = np.eye(3)[axis_permutation]
    target_affine[:3, 3] = offset

    resampled_img = resample_img(source_img, target_affine=target_affine)

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


@pytest.mark.parametrize("core_shape", [(3, 5, 4), (3, 5, 4, 2)])
def test_resampling_nan(affine, core_shape):
    """Test that when the data has NaNs they do not propagate to the \
    whole image."""
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

    source_img = Nifti1Image(full_data, affine)

    # Transform real data using easily checkable transformations
    # For now: axis permutations
    axis_permutation = [0, 1, 2]

    # check 3x3 transformation matrix
    target_affine = np.eye(3)[axis_permutation]
    with pytest.warns(Warning, match=r"(\bnan\b|invalid value)"):
        resampled_img = resample_img(source_img, target_affine=target_affine)

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


def test_resampling_nan_big(affine):
    """Test with an actual resampling, in the case of a bigish hole.

    This checks the extrapolation mechanism: if we don't do any
    extrapolation before resampling, the hole creates big
    artefacts
    """
    data = 10 * np.ones((10, 10, 10))
    data[4:6, 4:6, 4:6] = np.nan
    source_img = Nifti1Image(data, 2 * affine)

    with pytest.warns(RuntimeWarning):
        resampled_img = resample_img(source_img, target_affine=affine)

    resampled_data = get_data(resampled_img)
    assert_allclose(10, resampled_data[np.isfinite(resampled_data)])


def test_resample_to_img(affine, shape):
    rng = np.random.RandomState(42)

    data = rng.random_sample(shape)

    source_affine = affine
    source_img = Nifti1Image(data, source_affine)

    target_affine = 2 * source_affine
    target_img = Nifti1Image(data, target_affine)

    result_img = resample_to_img(
        source_img, target_img, interpolation="nearest"
    )

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    assert_almost_equal(downsampled, get_data(result_img)[:x, :y, :z, ...])


def test_crop(affine):
    # Testing that padding of arrays and cropping of images work symmetrically
    shape = (4, 6, 2)
    data = np.ones(shape)
    padded = _pad_array(data, [3, 2, 4, 4, 5, 7])
    padd_nii = Nifti1Image(padded, affine)

    cropped = crop_img(padd_nii, pad=False)

    assert_equal(get_data(cropped), data)


def test_resample_identify_affine_int_translation(affine):
    rng = np.random.RandomState(42)

    source_shape = (6, 4, 6)
    source_affine = affine
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
    assert_almost_equal(get_data(target_img), get_data(result_img))

    result_img_2 = resample_to_img(
        result_img, source_img, interpolation="nearest"
    )
    assert_almost_equal(get_data(source_img), get_data(result_img_2))

    result_img_3 = resample_to_img(
        result_img, source_img, interpolation="nearest", force_resample=True
    )
    assert_almost_equal(get_data(result_img_2), get_data(result_img_3))

    result_img_4 = resample_to_img(
        source_img, target_img, interpolation="nearest", force_resample=True
    )
    assert_almost_equal(get_data(target_img), get_data(result_img_4))


def test_resample_clip(affine):
    # Resample and image and get larger and smaller
    # value than in the original. Use clip to get rid of these images

    shape = (6, 3, 6)
    data = np.zeros(shape=shape)
    data[1:-2, 1:-1, 1:-2] = 1

    source_affine = np.diag((2, 2, 2, 1))
    source_img = Nifti1Image(data, source_affine)

    no_clip_data = get_data(
        resample_img(source_img, target_affine=affine, clip=False)
    )
    clip_data = get_data(
        resample_img(source_img, target_affine=affine, clip=True)
    )

    not_clip = np.where(
        (no_clip_data > data.min()) & (no_clip_data < data.max())
    )

    assert np.any(no_clip_data > data.max())
    assert np.any(no_clip_data < data.min())
    assert np.all(clip_data <= data.max())
    assert np.all(clip_data >= data.min())
    assert_array_equal(no_clip_data[not_clip], clip_data[not_clip])


def test_reorder_img(affine):
    rng = np.random.RandomState(42)

    # We need to test on a square array, as rotation does not change
    # shape, whereas reordering does.
    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)
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

        assert_array_equal(rot_img.affine, new_affine)
        assert_array_equal(get_data(rot_img).shape, shape)

        reordered_img = reorder_img(rot_img)

        assert_array_equal(reordered_img.affine[:3, :3], np.eye(3))
        assert_almost_equal(get_data(reordered_img), data)


def test_reorder_img_with_resample_arg(affine):
    rng = np.random.RandomState(42)

    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)
    affine = affine
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    ref_img = Nifti1Image(data, affine)

    interpolation = "nearest"

    reordered_img = reorder_img(ref_img, resample=interpolation)

    resampled_img = resample_img(
        ref_img,
        target_affine=reordered_img.affine,
        interpolation=interpolation,
    )
    assert_array_equal(get_data(reordered_img), get_data(resampled_img))


def test_reorder_img_error_reorder_axis(affine):
    rng = np.random.RandomState(42)

    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)

    # Create a non-diagonal affine, and check that we raise a sensible
    # exception
    affine[1, 0] = 0.1
    ref_img = Nifti1Image(data, affine)
    with pytest.raises(ValueError, match="Cannot reorder the axes"):
        reorder_img(ref_img)

    # Test that no exception is raised when resample='continuous'
    reorder_img(ref_img, resample="continuous")


def test_reorder_img_flipping_axis():
    rng = np.random.RandomState(42)

    shape = (5, 5, 5, 2, 2)

    data = rng.uniform(size=shape)

    for i in (0, 1, 2):
        # Make a diagonal affine with a negative axis, and check that
        # can be reordered, also vary the shape
        shape = (i + 1, i + 2, 3 - i)
        affine = np.eye(4)
        affine[i, i] *= -1

        img = Nifti1Image(data, affine)
        orig_img = copy.copy(img)

        img2 = reorder_img(img)

        # Check that img has not been changed
        assert_array_equal(img.affine, orig_img.affine)
        assert_array_equal(get_data(img), get_data(orig_img))

        # Test that the affine is indeed diagonal:
        assert_array_equal(
            img2.affine[:3, :3], np.diag(np.diag(img2.affine[:3, :3]))
        )
        assert np.all(np.diag(img2.affine) >= 0)


def test_reorder_img_error_interpolation():
    rng = np.random.RandomState(42)

    shape = (5, 5, 5, 2, 2)
    data = rng.uniform(size=shape)
    affine = np.eye(4)
    affine[1, 0] = 0.1
    ref_img = Nifti1Image(data, affine)

    with pytest.raises(ValueError, match="interpolation must be one of"):
        reorder_img(ref_img, resample="an_invalid_interpolation")


def test_reorder_img_non_native_endianness():
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
        return resample_img(img, target_affine=affine)

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
    reordered = reorder_img(img)
    assert_allclose(
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
def test_resampling_with_int_types_no_crash(affine, dtype):
    data = np.zeros((2, 2, 2))
    img = Nifti1Image(data.astype(dtype), affine)
    resample_img(img, target_affine=2.0 * affine)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "<i8", ">i8"])
@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_resampling_with_int64_types_no_crash(affine, dtype):
    data = np.zeros((2, 2, 2))
    # Passing dtype or header is required when using int64
    # https://nipy.org/nibabel/changelog.html#api-changes-and-deprecations
    hdr = Nifti1Header()
    hdr.set_data_dtype(dtype)
    img = Nifti1Image(data.astype(dtype), affine, header=hdr)
    resample_img(img, target_affine=2.0 * affine)


def test_resample_input(affine, shape):
    rng = np.random.RandomState(42)

    data = rng.randint(0, 10, shape, dtype="int32")
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    img = Nifti1Image(data, affine)

    with testing.write_tmp_imgs(img, create_files=True) as filename:
        filename = Path(filename)
        resample_img(filename, target_affine=affine, interpolation="nearest")


def test_smoke_resampling_non_nifti(affine, shape):
    rng = np.random.RandomState(42)

    target_affine = 2 * affine
    data = rng.randint(0, 10, shape, dtype="int32")
    img = MGHImage(data, affine)

    resample_img(img, target_affine=target_affine, interpolation="nearest")
