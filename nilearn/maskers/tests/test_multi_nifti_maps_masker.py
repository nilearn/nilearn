"""Test the multi_nifti_maps_masker module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri, generate_maps
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.conftest import _img_maps
from nilearn.exceptions import DimensionError
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker

ESTIMATORS_TO_CHECK = [MultiNiftiMapsMasker(standardize=None)]

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


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            # pass less than the default number of regions
            # to speed up the tests
            MultiNiftiMapsMasker(_img_maps(n_regions=2), standardize=None),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.slow
def test_multi_nifti_maps_masker(
    affine_eye, length, n_regions, shape_3d_default, img_maps
):
    """Check basic functions of MultiNiftiMapsMasker.

    - fit, transform, fit_transform, inverse_transform.
    - 4D and list[4D] inputs
    """
    fmri11_img, _ = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    # with default resampling_target="data"
    masker = MultiNiftiMapsMasker(img_maps, standardize=None)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]

    signals11_list = masker.fit_transform(signals_input)

    assert masker.n_elements_ == n_regions

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # Call inverse transform
    for signals in signals11_list:
        fmri11_img_r = masker.inverse_transform(signals)

        assert fmri11_img_r.shape == fmri11_img.shape
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)

    # Now try on a masker that has never seen the call to "transform"
    masker = MultiNiftiMapsMasker(
        img_maps, resampling_target=None, standardize=None
    )
    masker.fit(fmri11_img)
    masker.inverse_transform(signals)


def test_fit_errors(affine_eye, length, shape_3d_default, img_maps):
    """Check errors raised by MultiNiftiMapsMasker."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(
        img_maps, mask_img=mask11_img, resampling_target=None, standardize=None
    )

    signals_input = [fmri11_img, fmri11_img]

    # NiftiMapsMasker should not work with 4D + 1D input
    masker = NiftiMapsMasker(
        img_maps, resampling_target=None, standardize=None
    )
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker.fit_transform(signals_input)


@pytest.mark.slow
def test_resampling_to_mask(
    shape_mask,
    affine_eye,
    length,
    n_regions,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to mask in MultiNiftiMapsMasker.

    Mostly check output of fit_transform on 5D images.

    More systematic checks performed in tests for NiftiMapsMasker.
    """
    _, mask22_img = generate_fake_fmri(
        shape_mask, affine=affine_eye, length=length
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img,
        mask_img=mask22_img,
        resampling_target="mask",
        standardize=None,
    )

    signals = masker.fit_transform([img_fmri, img_fmri])

    # We are losing a few regions due to masking
    n_regions_expected = n_regions - 7

    expected_affine = mask22_img.affine
    expected_shape = mask22_img.shape

    for t in signals:
        assert t.shape == (length, n_regions_expected)

        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, expected_affine)
        assert fmri11_img_r.shape == (*expected_shape, length)


@pytest.mark.slow
def test_resampling_to_maps(
    shape_mask,
    affine_eye,
    length,
    n_regions,
    shape_3d_large,
    img_fmri,
):
    """Test resampling to maps in MultiNiftiMapsMasker.

    Mostly check output of fit_transform on 5D images.

    More systematic checks performed in tests for NiftiMapsMasker.
    """
    _, mask22_img = generate_fake_fmri(
        shape_mask, affine=affine_eye, length=length
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img,
        mask_img=mask22_img,
        resampling_target="maps",
        standardize=None,
    )

    signals = masker.fit_transform([img_fmri, img_fmri])

    # We have lost some regions due to masking
    expected_n_regions = n_regions - 7

    expected_affine = maps33_img.affine
    expected_shape = maps33_img.shape

    for t in signals:
        assert t.shape == (length, expected_n_regions)

        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, expected_affine)
        assert fmri11_img_r.shape == (expected_shape[:3] + (length,))


@pytest.mark.slow
def test_resampling_clipped_mask(affine_eye, length, n_regions, img_fmri):
    """Test with clipped maps: mask does not contain all maps.

    Mostly check output of fit_transform on 5D images.

    More systematic checks performed in tests for NiftiMapsMasker.
    """
    # Shapes do matter in that case
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps
    affine2 = np.diag((2, 2, 2, 1))  # just for mask

    _, mask22_img = generate_fake_fmri(shape2, length=1, affine=affine2)
    maps33_img, _ = generate_maps(shape3, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img,
        mask_img=mask22_img,
        resampling_target="maps",
        standardize=None,
    )

    signals = masker.fit_transform([img_fmri, img_fmri])

    # We are losing a few regions due to clipping
    n_expected_regions = n_regions - 5

    assert masker.n_elements_ == n_expected_regions

    for t in signals:
        assert t.shape == (length, n_expected_regions)

        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        masker.inverse_transform(t)
