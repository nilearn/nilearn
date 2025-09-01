"""Test the multi_nifti_maps_masker module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri, generate_maps
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import _img_maps
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker

ESTIMATORS_TO_CHECK = [MultiNiftiMapsMasker()]

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
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            valid=False,
        ),
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
        estimators=[
            # pass less than the default number of regions
            # to speed up the tests
            MultiNiftiMapsMasker(_img_maps(n_regions=2)),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.timeout(0)
def test_multi_nifti_maps_masker(
    affine_eye, length, n_regions, shape_3d_default, img_maps
):
    """Check basic functions of MultiNiftiMapsMasker.

    - fit, transform, fit_transform, inverse_transform.
    - 4D and list[4D] inputs
    """
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(
        img_maps, mask_img=mask11_img, resampling_target=None
    )

    signals11 = masker.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    MultiNiftiMapsMasker(img_maps).fit_transform(fmri11_img)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]

    signals11_list = masker.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # Call inverse transform
    for signals in signals11_list:
        fmri11_img_r = masker.inverse_transform(signals)

        assert fmri11_img_r.shape == fmri11_img.shape
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)

    # Now try on a masker that has never seen the call to "transform"
    masker = MultiNiftiMapsMasker(img_maps, resampling_target=None)
    masker.fit()
    masker.inverse_transform(signals)


def test_multi_nifti_maps_masker_data_atlas_different_shape(
    affine_eye, length, img_maps
):
    """Test with data and atlas of different shape.

    The atlas should be resampled to the data.
    """
    # Check working of shape/affine checks
    shape2 = (12, 10, 14)
    shape22 = (5, 5, 6)
    affine2 = np.diag((1, 2, 3, 1))
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    _, mask21_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )
    fmri22_img, _ = generate_fake_fmri(shape22, affine=affine2, length=length)

    masker = MultiNiftiMapsMasker(img_maps, mask_img=mask21_img)

    masker.fit_transform(fmri22_img)

    assert_array_equal(masker.maps_img_.affine, affine2)


def test_multi_nifti_maps_masker_errors(
    affine_eye, length, shape_3d_default, img_maps
):
    """Check errors raised by MultiNiftiMapsMasker."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(
        img_maps, mask_img=mask11_img, resampling_target=None
    )

    signals_input = [fmri11_img, fmri11_img]

    # NiftiMapsMasker should not work with 4D + 1D input
    masker = NiftiMapsMasker(img_maps, resampling_target=None)
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker.fit_transform(signals_input)


@pytest.mark.parametrize("create_files", [True, False])
def test_multi_nifti_maps_masker_errors_field_of_view(
    tmp_path,
    affine_eye,
    length,
    create_files,
    shape_3d_default,
    img_maps,
):
    """Test all kinds of mismatches between shapes and between affines."""
    # Check working of shape/affine checks
    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    fmri12_img, mask12_img = generate_fake_fmri(
        shape_3d_default, affine=affine2, length=length
    )
    fmri21_img, mask21_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    error_msg = "Following field of view errors were detected"

    masker = MultiNiftiMapsMasker(
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
    labels11, mask12 = images
    masker = MultiNiftiMapsMasker(labels11, resampling_target=None)
    masker.fit()

    with pytest.raises(ValueError, match=error_msg):
        masker.transform(fmri12_img)

    with pytest.raises(ValueError, match=error_msg):
        masker.transform(fmri21_img)

    masker = MultiNiftiMapsMasker(
        labels11, mask_img=mask12, resampling_target=None
    )
    with pytest.raises(ValueError, match=error_msg):
        masker.fit()


def test_multi_nifti_maps_masker_resampling_error(
    affine_eye, n_regions, shape_3d_large
):
    """Test MultiNiftiMapsMasker when using resampling."""
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    # Test error checking
    masker = MultiNiftiMapsMasker(maps33_img, resampling_target="mask")
    with pytest.raises(
        ValueError,
        match=(
            "resampling_target has been set to 'mask' "
            "but no mask has been provided"
        ),
    ):
        masker.fit()

    masker = MultiNiftiMapsMasker(maps33_img, resampling_target="invalid")
    with pytest.raises(
        ValueError, match="invalid value for 'resampling_target' parameter:"
    ):
        masker.fit()


@pytest.mark.timeout(0)
def test_multi_nifti_maps_masker_resampling_to_mask(
    shape_mask,
    affine_eye,
    length,
    n_regions,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to mask in MultiNiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, affine=affine_eye, length=length
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="mask"
    )

    signals = masker.fit_transform([img_fmri, img_fmri])

    assert_almost_equal(masker.mask_img_.affine, mask22_img.affine)
    assert masker.mask_img_.shape == mask22_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    for t in signals:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, masker.maps_img_.affine)
        assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


def test_multi_nifti_maps_masker_resampling_to_maps(
    shape_mask,
    affine_eye,
    length,
    n_regions,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to maps in MultiNiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, affine=affine_eye, length=length
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    signals = masker.fit_transform([img_fmri, img_fmri])

    assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    for t in signals:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, masker.maps_img_.affine)
        assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


def test_multi_nifti_maps_masker_resampling_clipped_mask(
    affine_eye, length, n_regions, img_fmri
):
    """Test with clipped maps: mask does not contain all maps."""
    # Shapes do matter in that case
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps
    affine2 = np.diag((2, 2, 2, 1))  # just for mask

    _, mask22_img = generate_fake_fmri(shape2, length=1, affine=affine2)
    maps33_img, _ = generate_maps(shape3, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    signals = masker.fit_transform([img_fmri, img_fmri])

    assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    for t in signals:
        assert t.shape == (length, n_regions)
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, masker.maps_img_.affine)
        assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))
