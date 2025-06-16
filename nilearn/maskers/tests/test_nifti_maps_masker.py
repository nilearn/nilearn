"""Test nilearn.maskers.nifti_maps_masker.

Functions in this file only test features added by the NiftiMapsMasker class,
rather than the underlying functions (clean(), img_to_signals_labels(), etc.).

See test_masking.py and test_signal.py for details.
"""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_maps,
    generate_random_img,
)
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import _img_maps, _shape_3d_default
from nilearn.image import get_data
from nilearn.maskers import NiftiMapsMasker

ESTIMATORS_TO_CHECK = [NiftiMapsMasker()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[  # pass less than the default number of regions
            # to speed up the tests
            NiftiMapsMasker(maps_img=_img_maps(n_regions=2))
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_nifti_maps_masker_data_atlas_different_shape(
    length, affine_eye, img_maps
):
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

    masker.fit(fmri22_img)

    assert_array_equal(masker.maps_img_.affine, affine2)


def test_nifti_maps_masker_fit(n_regions, img_maps):
    """Check fitted attributes."""
    masker = NiftiMapsMasker(img_maps, resampling_target=None)

    masker.fit()

    # Check attributes defined at fit
    assert masker.n_elements_ == n_regions


def test_nifti_maps_masker_errors():
    """Check fitting errors."""
    masker = NiftiMapsMasker()
    with pytest.raises(TypeError, match="input should be a NiftiLike object"):
        masker.fit()


@pytest.mark.parametrize("create_files", (True, False))
def test_nifti_maps_masker_errors_field_of_view(
    tmp_path, length, affine_eye, shape_3d_default, create_files, img_maps
):
    """Check field of view errors."""
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

    # Test all kinds of mismatches between shapes and between affines
    images = write_imgs_to_path(
        img_maps,
        mask12_img,
        file_path=tmp_path,
        create_files=create_files,
    )
    labels11, mask12 = images

    masker = NiftiMapsMasker(labels11, resampling_target=None)

    with pytest.raises(ValueError, match=error_msg):
        masker.fit_transform(fmri12_img)

    with pytest.raises(ValueError, match=error_msg):
        masker.fit_transform(fmri21_img)

    masker = NiftiMapsMasker(labels11, mask_img=mask12, resampling_target=None)
    with pytest.raises(ValueError, match=error_msg):
        masker.fit()


def test_nifti_maps_masker_resampling_errors(
    n_regions, affine_eye, shape_3d_large
):
    """Test resampling errors."""
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = NiftiMapsMasker(maps33_img, resampling_target="mask")

    with pytest.raises(
        ValueError,
        match=(
            "resampling_target has been set to 'mask' "
            "but no mask has been provided."
        ),
    ):
        masker.fit()

    masker = NiftiMapsMasker(maps33_img, resampling_target="invalid")
    with pytest.raises(
        ValueError,
        match="invalid value for 'resampling_target' parameter: invalid",
    ):
        masker.fit()


def test_nifti_maps_masker_with_nans_and_infs(length, n_regions, affine_eye):
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

    assert signals.shape == (length, n_regions)
    assert np.all(np.isfinite(signals))


def test_nifti_maps_masker_with_nans_and_infs_in_data(
    length, n_regions, affine_eye
):
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

    assert signals.shape == (length, n_regions)
    assert np.all(np.isfinite(signals))


def test_nifti_maps_masker_resampling_to_mask(
    length,
    n_regions,
    affine_eye,
    shape_mask,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to_mask in NiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, length=length, affine=affine_eye
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    # Target: mask
    masker = NiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="mask"
    )

    signals = masker.fit_transform(img_fmri)

    assert_almost_equal(masker.mask_img_.affine, mask22_img.affine)
    assert masker.mask_img_.shape == mask22_img.shape

    assert_almost_equal(masker.maps_img_.affine, masker.mask_img_.affine)
    assert masker.maps_img_.shape[:3] == masker.mask_img_.shape

    assert signals.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(signals)

    assert_almost_equal(fmri11_img_r.affine, masker.mask_img_.affine)
    assert fmri11_img_r.shape == (masker.mask_img_.shape[:3] + (length,))


def test_nifti_maps_masker_resampling_to_maps(
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

    masker = NiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    signals = masker.fit_transform(img_fmri)

    assert_array_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    assert_array_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    assert signals.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(signals)

    assert_array_equal(fmri11_img_r.affine, masker.maps_img_.affine)
    assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


def test_nifti_maps_masker_clipped_mask(n_regions, affine_eye):
    """Test with clipped maps: mask does not contain all maps."""
    # Shapes do matter in that case
    length = 21
    shape1 = (10, 11, 12, length)
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps
    affine2 = np.diag((2, 2, 2, 1))  # just for mask

    fmri11_img, _ = generate_random_img(shape1, affine=affine_eye)
    _, mask22_img = generate_fake_fmri(shape2, length=1, affine=affine2)
    # Target: maps
    maps33_img, _ = generate_maps(shape3, n_regions, affine=affine_eye)

    masker = NiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    signals = masker.fit_transform(fmri11_img)

    assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    assert signals.shape == (length, n_regions)
    # Some regions have been clipped. Resulting signal must be zero
    assert (signals.var(axis=0) == 0).sum() < n_regions

    fmri11_img_r = masker.inverse_transform(signals)

    assert_almost_equal(fmri11_img_r.affine, masker.maps_img_.affine)
    assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


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
def test_nifti_maps_masker_overlap(maps_img_fn, allow_overlap, img_fmri):
    """Test resampling in NiftiMapsMasker."""
    masker = NiftiMapsMasker(maps_img_fn(), allow_overlap=allow_overlap)

    if allow_overlap is False and maps_img_fn.__name__ == "overlapping_maps":
        with pytest.raises(ValueError, match="Overlap detected"):
            masker.fit_transform(img_fmri)
    else:
        masker.fit_transform(img_fmri)


def test_standardization(rng, affine_eye, shape_3d_default):
    """Check output properly standardized with 'standardize' parameter."""
    length = 500

    signals = rng.standard_normal(size=(np.prod(shape_3d_default), length))
    means = (
        rng.standard_normal(size=(np.prod(shape_3d_default), 1)) * 50 + 1000
    )
    signals += means
    img = Nifti1Image(signals.reshape((*shape_3d_default, length)), affine_eye)

    maps, _ = generate_maps((9, 9, 5), 10)

    # Unstandarized
    masker = NiftiMapsMasker(maps, standardize=False)
    unstandarized_label_signals = masker.fit_transform(img)

    # z-score
    masker = NiftiMapsMasker(maps, standardize="zscore_sample")
    trans_signals = masker.fit_transform(img)

    assert_almost_equal(trans_signals.mean(0), 0)
    assert_almost_equal(trans_signals.std(0), 1, decimal=3)

    # psc
    masker = NiftiMapsMasker(maps, standardize="psc")
    trans_signals = masker.fit_transform(img)

    assert_almost_equal(trans_signals.mean(0), 0)
    assert_almost_equal(
        trans_signals,
        (
            unstandarized_label_signals
            / unstandarized_label_signals.mean(0)
            * 100
            - 100
        ),
    )
