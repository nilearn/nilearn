"""Test nilearn.maskers.nifti_maps_masker.

Functions in this file only test features added by the NiftiMapsMasker class,
rather than the underlying functions (clean(), img_to_signals_labels(), etc.).

See test_masking.py and test_signal.py for details.
"""

import warnings
from contextlib import nullcontext

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn import clone

from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_maps,
    generate_random_img,
)
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import _shape_3d_default
from nilearn.image import get_data
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker

from .conftest import check_signal_and_inverse_transform


def test_data_atlas_different_shape(length, affine_eye, img_maps):
    """Test with data and atlas of different shape.

    The atlas should be resampled to the data.
    """
    shape2 = (12, 10, 14)

    shape22 = (5, 5, 6)

    affine2 = 2 * affine_eye
    affine2[-1, -1] = 1

    _, mask21_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )
    fmri22_img, _ = generate_fake_fmri(shape22, affine=affine2, length=length)

    masker = NiftiMapsMasker(img_maps, mask_img=mask21_img)

    with warnings.catch_warnings(record=True) as warning_list:
        masker.fit(fmri22_img)
        assert all(
            "consider using nearest interpolation instead"
            not in str(x.message)
            for x in warning_list
        )

    assert_array_equal(masker.maps_img_.affine, affine2)


@pytest.mark.parametrize("n_regions", [1, 3])
def test_fit(n_regions, img_maps):
    """Check fitted attributes."""
    assert img_maps.shape[3] == n_regions

    masker = NiftiMapsMasker(img_maps, resampling_target=None)

    masker.fit()

    # Check attributes defined at fit
    assert masker.n_elements_ == n_regions


def test_error():
    """Raise error when fitting with no map image."""
    masker = NiftiMapsMasker()
    with pytest.raises(TypeError, match="input should be a NiftiLike object"):
        masker.fit()


def test_empty_maps_dropped_warning(img_maps):
    """Warn and drop empty maps at fit time."""
    maps_data = get_data(img_maps).copy()
    n_maps = maps_data.shape[-1]
    maps_data[..., 1] = 0
    maps_img = Nifti1Image(maps_data, img_maps.affine)

    masker = NiftiMapsMasker(maps_img)
    with pytest.warns(
        UserWarning,
        match=(
            r"The following maps are empty and were removed: \{1\}. "
            f"Out of {n_maps} maps"
        ),
    ):
        masker.fit()

    assert masker.n_elements_ == n_maps - 1
    assert masker.maps_img_.shape[-1] == n_maps - 1


def test_empty_img_map_error(img_3d_zeros_eye):
    """Raise error when image maps is empty."""
    masker = NiftiMapsMasker(img_3d_zeros_eye)
    with pytest.raises(ValueError, match="maps_img contains no map"):
        masker.fit()


@pytest.mark.parametrize("resampling_target", ["mask", "maps"])
def test_mask_img_masks_all_maps_error(
    affine_eye, shape_3d_default, img_4d_rand_eye, resampling_target
):
    """Raise error if mask_img excludes all voxels with map value.

    For resampling target is
    - "maps" or "mask": raise error at fit time whether img is passed or not
    - "data": raise error at fit time,
      as the mask is applied to the maps
      (on the data grid if data is passed at fit time,
      on the maps grid otherwise)
    """
    mask = np.zeros(shape_3d_default)
    mask[-1][-1][-1] = 1
    mask_img = Nifti1Image(mask, affine_eye)

    # generate label image with slightly different field of view
    # to force resampling
    maps = np.zeros((*tuple(x + 1 for x in shape_3d_default), 1))
    maps[0][0][0][0] = 0.5
    maps_img = Nifti1Image(maps, affine_eye)

    masker = NiftiMapsMasker(
        maps_img,
        mask_img=mask_img,
        resampling_target=resampling_target,
    )

    with pytest.raises(
        ValueError,
        match="No map left after applying mask to the maps image",
    ):
        masker.fit()

    with pytest.raises(
        ValueError,
        match="No map left after applying mask to the maps image",
    ):
        masker.fit(img_4d_rand_eye)

    # by default we resample to data
    # if no img is passed at fit time,
    # the mask is applied to the maps on the maps grid

    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)

    with pytest.raises(
        ValueError, match="No map left after applying mask to the maps image"
    ):
        masker.fit(img_4d_rand_eye)

    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)
    with pytest.raises(
        ValueError, match="No map left after applying mask to the maps image"
    ):
        masker.fit()


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("create_files", (True, False))
def test_errors_field_of_view(
    tmp_path, length, affine_eye, shape_3d_default, create_files, img_maps
):
    """Test all kinds of mismatches between shapes and between affines
    with resampling_target=None.
    """
    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    fmri12_img, mask12_img = generate_fake_fmri(
        shape_3d_default, affine=affine2, length=length
    )
    fmri21_img, mask21_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    error_msg = "Following field of view errors were detected"

    masker = NiftiMapsMasker(
        img_maps, mask_img=mask21_img, resampling_target=None
    )

    with pytest.raises(ValueError, match=error_msg):
        masker.fit()

    images = write_imgs_to_path(
        img_maps,
        mask12_img,
        file_path=tmp_path,
        create_files=create_files,
    )
    maps, mask12 = images

    masker = NiftiMapsMasker(maps, resampling_target=None)

    with pytest.raises(ValueError, match=error_msg):
        masker.fit(fmri12_img)

    with pytest.raises(ValueError, match=error_msg):
        masker.fit(fmri21_img)

    masker = NiftiMapsMasker(maps, mask_img=mask12, resampling_target=None)
    with pytest.raises(ValueError, match=error_msg):
        masker.fit()


def test_resampling_errors(img_maps):
    """Test resampling errors."""
    masker = NiftiMapsMasker(img_maps, resampling_target="mask")

    with pytest.raises(
        ValueError,
        match=(
            r"resampling_target has been set to 'mask' "
            r"but no mask has been provided."
        ),
    ):
        masker.fit()

    masker = NiftiMapsMasker(img_maps, resampling_target="invalid")
    with pytest.raises(
        ValueError,
        match="'resampling_target' must be one of",
    ):
        masker.fit()


def test_no_map_after_resampling_error(
    img_maps, affine_mni, shape_3d_large, rng
):
    """Ensure error is raised when resampling leads to no map left."""
    input_img = Nifti1Image(rng.random(shape_3d_large), affine_mni)

    estimator = NiftiMapsMasker(maps_img=img_maps)
    estimator.fit()
    with (
        pytest.warns(UserWarning, match="Resampling maps at transform time"),
        pytest.raises(ValueError, match="No map left after resampling"),
    ):
        estimator.transform(input_img)

    estimator = NiftiMapsMasker(maps_img=img_maps)
    with pytest.raises(ValueError, match="No map left after resampling"):
        estimator.fit_transform(input_img)


def test_with_nans_and_infs(length, n_regions, affine_eye):
    """Apply a NiftiMapsMasker containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    without raising a warning.
    """
    fmri_img, mask_img = generate_random_img(
        (13, 11, 12, length), affine=affine_eye
    )
    maps_img, _ = generate_maps((13, 11, 12), n_regions, affine=affine_eye)

    # Add NaNs and infs to atlas
    maps_data = get_data(maps_img).astype(np.float32)
    mask_data = get_data(mask_img).astype(np.float32)
    maps_data = maps_data * mask_data[..., None]

    # Choose a good voxel from the first label
    vox_idx = np.where(maps_data[..., 0] > 0)
    i1, j1, k1 = vox_idx[0][0], vox_idx[1][0], vox_idx[2][0]
    i2, j2, k2 = vox_idx[0][1], vox_idx[1][1], vox_idx[2][1]

    maps_data[:, :, :, 0] = np.nan
    maps_data[i2, j2, k2, 0] = np.inf
    maps_data[i1, j1, k1, 0] = 1

    maps_img = Nifti1Image(maps_data, affine_eye)

    # No warning, because maps_img is run through clean_img
    # *before* safe_get_data.
    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)

    signals = masker.fit_transform(fmri_img)

    assert np.all(np.isfinite(signals))

    check_signal_and_inverse_transform(masker, signals, expected_length=length)


def test_with_nans_and_infs_in_data(length, n_regions, affine_eye):
    """Apply a NiftiMapsMasker to 4D data containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    fmri_img, mask_img = generate_random_img(
        (13, 11, 12, length), affine=affine_eye
    )
    maps_img, _ = generate_maps((13, 11, 12), n_regions, affine=affine_eye)

    # Add NaNs and infs to data
    fmri_data = get_data(fmri_img)

    fmri_data[:, 9, 9, :] = np.nan
    fmri_data[:, 5, 5, :] = np.inf

    fmri_img = Nifti1Image(fmri_data, affine_eye)

    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        signals = masker.fit_transform(fmri_img)

    assert np.all(np.isfinite(signals))

    check_signal_and_inverse_transform(masker, signals, expected_length=length)


@pytest.mark.parametrize("estimator", [NiftiMapsMasker, MultiNiftiMapsMasker])
def test_resampling_to_mask(
    estimator,
    length,
    n_regions,
    affine_eye,
    shape_mask,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to mask in (Multi)NiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, length=length, affine=affine_eye
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = estimator(
        maps33_img, mask_img=mask22_img, resampling_target="mask"
    )

    input_img = [img_fmri, img_fmri]
    if isinstance(masker, NiftiMapsMasker):
        input_img = img_fmri

    # maps emptied by resampling or excluded by the mask
    # are dropped at fit time,
    # and each step only reports the maps it removed
    with (
        pytest.warns(
            UserWarning,
            match="the resampled map image only contains 4 non-empty maps",
        ),
        pytest.warns(
            UserWarning,
            match="the masked map image only contains 2 maps",
        ),
    ):
        masker.fit()
    assert masker.n_elements_ == 2
    with (
        warnings.catch_warnings(record=True) as warning_list,
    ):
        signals = masker.fit_transform(input_img)
        assert all(
            "consider using nearest interpolation instead" not in str(x)
            for x in warning_list
        )
        assert any(
            "Out of 4 maps, the masked map image only contains 2 maps"
            in str(x)
            for x in warning_list
        )
    n_regions_left = 2
    assert masker.n_elements_ == n_regions_left

    assert_almost_equal(masker.mask_img_.affine, mask22_img.affine)
    assert masker.mask_img_.shape == mask22_img.shape

    assert_almost_equal(masker.maps_img_.affine, masker.mask_img_.affine)
    assert masker.maps_img_.shape[:3] == masker.mask_img_.shape

    check_signal_and_inverse_transform(masker, signals, expected_length=length)


@pytest.mark.parametrize("estimator", [NiftiMapsMasker, MultiNiftiMapsMasker])
def test_resampling_to_maps(
    estimator,
    length,
    n_regions,
    affine_eye,
    shape_mask,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to maps in NiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, length=length, affine=affine_eye
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = estimator(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    input_img = [img_fmri, img_fmri]
    if isinstance(masker, NiftiMapsMasker):
        input_img = img_fmri

    # maps excluded by the mask are dropped at fit time
    with pytest.warns(
        UserWarning,
        match="the masked map image only contains 2 maps",
    ):
        masker.fit()
    n_regions_left = 2
    assert masker.n_elements_ == n_regions_left
    with pytest.warns(
        UserWarning, match="Resampling images at transform time"
    ):
        signals = masker.transform(input_img)
    assert masker.n_elements_ == n_regions_left

    assert_array_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == (*maps33_img.shape[:3], n_regions_left)

    assert_array_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    check_signal_and_inverse_transform(masker, signals, expected_length=length)


@pytest.mark.parametrize("estimator", [NiftiMapsMasker, MultiNiftiMapsMasker])
def test_clipped_mask(estimator, affine_eye, length, n_regions, img_fmri):
    """Test with clipped maps: mask does not contain all maps."""
    # Shapes do matter in that case
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps
    affine2 = np.diag((2, 2, 2, 1))  # just for mask

    _, mask22_img = generate_fake_fmri(shape2, length=1, affine=affine2)
    maps33_img, _ = generate_maps(shape3, n_regions, affine=affine_eye)

    masker = estimator(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    input_img = [img_fmri, img_fmri]
    if isinstance(masker, NiftiMapsMasker):
        input_img = img_fmri

    # clipping is done at fit time
    with pytest.warns(
        UserWarning,
        match="the masked map image only contains 4 maps",
    ):
        masker.fit()

    n_regions_left = 4
    assert masker.n_elements_ == n_regions_left

    with pytest.warns(
        UserWarning, match="Resampling images at transform time"
    ):
        signals = masker.transform(input_img)

    assert masker.n_elements_ == n_regions_left

    assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == (*maps33_img.shape[:3], n_regions_left)

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    check_signal_and_inverse_transform(masker, signals, expected_length=length)

    if isinstance(masker, NiftiMapsMasker):
        signals = [signals]
    for t in signals:
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions


def non_overlapping_maps():
    """Generate maps with non-overlapping regions.

    All voxels belong to only 1 region.
    """
    non_overlapping_data = np.zeros((*_shape_3d_default(), 2))
    non_overlapping_data[:2, :, :, 0] = 1.0
    non_overlapping_data[2:, :, :, 1] = 1.0
    return Nifti1Image(
        non_overlapping_data,
        np.eye(4),
    )


def overlapping_maps():
    """Generate maps with overlapping regions.

    Same voxel has non null value for 2 different regions.
    """
    overlapping_data = np.zeros((*_shape_3d_default(), 2))
    overlapping_data[:3, :, :, 0] = 1.0
    overlapping_data[2:, :, :, 1] = 1.0
    return Nifti1Image(overlapping_data, np.eye(4))


@pytest.mark.parametrize(
    "maps_img_fn", [overlapping_maps, non_overlapping_maps]
)
@pytest.mark.parametrize("allow_overlap", [True, False])
def test_overlap(maps_img_fn, allow_overlap, img_fmri):
    """Test overlap in NiftiMapsMasker."""
    masker = NiftiMapsMasker(maps_img_fn(), allow_overlap=allow_overlap)

    if allow_overlap is False and maps_img_fn.__name__ == "overlapping_maps":
        with pytest.raises(ValueError, match="Overlap detected"):
            masker.fit_transform(img_fmri)
    else:
        signal = masker.fit_transform(img_fmri)
        check_signal_and_inverse_transform(masker, signal)


@pytest.mark.parametrize("with_mask", [False, True])
def test_maps_emptied_by_resampling_are_dropped(affine_eye, with_mask):
    """Check maps emptied by resampling at fit time are dropped.

    This should happen whether a mask is used or not.
    """
    fmri_img, _ = generate_random_img((7, 8, 9, 10), affine=affine_eye * 2)
    _, mask_img = generate_random_img((8, 9, 10, 1), affine=affine_eye * 3)
    maps_img, _ = generate_maps((16, 18, 20), 9, affine=affine_eye * 3)

    masker = NiftiMapsMasker(
        maps_img, mask_img=mask_img if with_mask else None
    )

    with pytest.warns(
        UserWarning,
        match="the resampled map image only contains 2 non-empty maps",
    ):
        signals = masker.fit_transform(fmri_img)

    assert masker.n_elements_ == 2
    assert masker.maps_img_.shape[-1] == 2

    check_signal_and_inverse_transform(masker, signals, expected_length=10)


@pytest.mark.parametrize("with_mask", [False, True])
def test_masking_inverse_transform(n_regions, length, affine_eye, with_mask):
    """Check for regression for https://github.com/nilearn/nilearn/issues/6658.

    Checks consistency of behavior for inverse transform whether doing:
    - using different 'methods'
        - fit_transform
        - fit then transform passing the same data to both
        - only passing data at transform time
    - with and without a mask that will drop some maps
    """
    # Generate data
    shape1 = (7, 8, 9, length)
    fmri_img, _ = generate_random_img(shape1, affine=affine_eye * 2)

    # Generate mask that will mask some maps
    # at fit time and some more at transform time
    # when maps and mask are resampled to the data
    shape2 = (8, 9, 10, 1)
    _, mask_img = generate_random_img(shape2, affine=affine_eye * 3)

    # Generate label image
    shape3 = (16, 18, 20)
    maps_img, _ = generate_maps(shape3, n_regions, affine=affine_eye * 3)

    masker = NiftiMapsMasker(
        maps_img, mask_img=mask_img if with_mask else None
    )

    # fit_transform
    masker = clone(masker)
    with pytest.warns(UserWarning, match="After resampling the maps image"):
        signals = masker.fit_transform(fmri_img)
    check_signal_and_inverse_transform(masker, signals)

    # same data at fit and transform time
    masker = clone(masker)
    with pytest.warns(UserWarning, match="After resampling the maps image"):
        masker = masker.fit(fmri_img)
    n_elements_post_fit = masker.n_elements_
    signals = masker.transform(fmri_img)
    assert masker.n_elements_ == n_elements_post_fit
    check_signal_and_inverse_transform(masker, signals)

    # resampling to data at transform time
    masker = clone(masker)
    with (
        pytest.warns(
            UserWarning, match="After applying mask to the maps image"
        )
        if with_mask
        else nullcontext()
    ):
        masker = masker.fit()

    n_elements_post_fit = masker.n_elements_

    with (
        pytest.warns(UserWarning, match="Resampling .* at transform time"),
        pytest.warns(
            UserWarning,
            match=(
                "extracted data will not have the shape "
                "expected by 'inverse_transform"
            ),
        )
        if with_mask
        else nullcontext(),
    ):
        signals = masker.transform(fmri_img)

    assert masker.n_elements_ == n_elements_post_fit

    if not with_mask:
        # in this case signal shape
        # does not match the expected n_elements_
        # because some maps were dropped at transform time
        check_signal_and_inverse_transform(masker, signals)
