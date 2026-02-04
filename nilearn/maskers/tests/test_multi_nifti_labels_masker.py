"""Test the multi_nifti_labels_masker module."""

import pytest
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import (
    generate_fake_fmri,
)
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.conftest import _img_labels
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.maskers.tests.conftest import (
    check_nifti_labels_masker_post_transform,
)

ESTIMATORS_TO_CHECK = [MultiNiftiLabelsMasker(standardize=None)]

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


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            MultiNiftiLabelsMasker(labels_img=_img_labels(), standardize=None)
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@ignore_warnings(category=RuntimeWarning)
@pytest.mark.slow
@pytest.mark.parametrize("keep_masked_labels", [True, False])
def test_multi_nifti_labels_masker(
    affine_eye,
    n_regions,
    shape_3d_default,
    length,
    img_labels,
    keep_masked_labels,
):
    """Check working of shape/affine checks."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker11 = MultiNiftiLabelsMasker(
        img_labels,
        mask_img=mask11_img,
        resampling_target=None,
        keep_masked_labels=keep_masked_labels,
        standardize=None,
    )

    expected_n_regions = n_regions
    if not keep_masked_labels:
        expected_n_regions = n_regions - 5

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]

    if keep_masked_labels:
        # TODO (nilearn >=0.15)
        # only keep else block
        with pytest.warns(
            FutureWarning,
            match='"keep_masked_labels" parameter will be removed',
        ):
            signals11_list = masker11.fit_transform(signals_input)
    else:
        with pytest.warns(
            UserWarning,
            match=(
                "After applying mask to the labels image, "
                "the following labels were removed"
            ),
        ):
            signals11_list = masker11.fit_transform(signals_input)

    check_nifti_labels_masker_post_transform(
        masker11,
        expected_n_regions,
        signals11_list,
        length,
        ref_affine=affine_eye,
        ref_shape=shape_3d_default,
    )
