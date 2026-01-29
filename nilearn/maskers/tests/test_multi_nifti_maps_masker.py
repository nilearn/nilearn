"""Test MultiNiftiMapsMasker."""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.conftest import _img_maps
from nilearn.maskers import MultiNiftiMapsMasker
from nilearn.maskers.tests.conftest import (
    check_nifti_maps_masker_post_fit,
    check_nifti_maps_masker_post_transform,
)

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
@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_multi_nifti_maps_masker(
    affine_eye, length, n_regions, shape_3d_default, img_maps, keep_masked_maps
):
    """Check basic functions of MultiNiftiMapsMasker.

    - fit, transform, fit_transform, inverse_transform.
    - 4D and list[4D] inputs
    """
    fmri_img, mask_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    # Should work with 4D + 1D input too
    input = [fmri_img, fmri_img]

    masker = MultiNiftiMapsMasker(
        img_maps,
        mask_img=mask_img,
        resampling_target=None,
        keep_masked_maps=keep_masked_maps,
        standardize=None,
    )

    masker.fit()

    check_nifti_maps_masker_post_fit(
        masker,
        n_regions,
        ref_affine=affine_eye,
        ref_shape=shape_3d_default,
    )

    if keep_masked_maps:
        # TODO (nilearn >=0.15)
        # only keep else block
        with pytest.warns(
            FutureWarning, match='"keep_masked_maps" parameter will be removed'
        ):
            signals11_list = masker.fit_transform(input)
    else:
        signals11_list = masker.fit_transform(input)

    expected_n_regions = n_regions
    if not keep_masked_maps:
        expected_n_regions = n_regions - 3

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals11_list,
        length,
        ref_affine=affine_eye,
        ref_shape=shape_3d_default,
    )
