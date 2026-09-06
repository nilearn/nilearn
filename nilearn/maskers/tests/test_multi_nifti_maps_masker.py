"""Test the multi_nifti_maps_masker module."""

import pytest
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.estimator_checks import (
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn.conftest import _img_maps
from nilearn.exceptions import DimensionError
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker

ESTIMATORS_TO_CHECK = [MultiNiftiMapsMasker()]


@parametrize_with_checks(
    estimators=ESTIMATORS_TO_CHECK,
    expected_failed_checks=return_expected_failed_checks,
)
def test_check_estimator_sklearn(estimator, check):
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            # pass less than the default number of regions
            # to speed up the tests
            MultiNiftiMapsMasker(_img_maps(n_regions=2), standardize=None),
            MultiNiftiMapsMasker(_img_maps(n_regions=1), standardize=None),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.parametrize("n_regions", [1, 3])
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
        img_maps,
        mask_img=mask11_img,
        resampling_target=None,
        keep_masked_maps=True,
        standardize=None,
    )

    with pytest.warns(
        FutureWarning,
        match=r'"keep_masked_maps" parameter will be removed in version 0\.15',
    ):
        signals11 = masker.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    MultiNiftiMapsMasker(img_maps, standardize=None).fit_transform(fmri11_img)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    with pytest.warns(
        FutureWarning, match=('"keep_masked_maps" parameter will be removed')
    ):
        signals11_list = masker.fit_transform(signals_input)

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
    masker.fit()
    masker.inverse_transform(signals)


def test_errors(affine_eye, length, shape_3d_default, img_maps):
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
