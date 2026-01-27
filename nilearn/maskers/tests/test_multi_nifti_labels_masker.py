"""Test the multi_nifti_labels_masker module."""

import pytest
from numpy.testing import assert_almost_equal
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


@pytest.mark.slow
def test_multi_nifti_labels_masker(
    affine_eye, n_regions, shape_3d_default, length, img_labels
):
    """Check working of shape/affine checks."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiLabelsMasker(
        img_labels,
        mask_img=mask11_img,
        resampling_target=None,
        standardize=None,
    )

    # We are losing a few regions due to masking
    n_regions_expected = n_regions - 5

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions_expected)

        # ensure we can inverse transform even after losing a few regions
        fmri11_img_r = masker.inverse_transform(signals)

    # same with no mask
    masker = MultiNiftiLabelsMasker(
        img_labels, resampling_target=None, standardize=None
    )
    signals11_list = masker.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker.inverse_transform(signals)

        assert fmri11_img_r.shape == fmri11_img.shape
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)


@pytest.mark.slow
def test_resampling(affine_eye, n_regions, length, img_labels):
    """Test resampling in MultiNiftiLabelsMasker."""
    shape1 = (10, 11, 12)

    # mask
    shape2 = (16, 17, 18)

    # With data of the same affine
    fmri11_img, _ = generate_fake_fmri(
        shape1, affine=affine_eye, length=length
    )
    _, mask22_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    masker = MultiNiftiLabelsMasker(
        img_labels,
        mask_img=mask22_img,
        resampling_target="labels",
        standardize=None,
    )

    fmri11_img = [fmri11_img, fmri11_img]

    signals = masker.fit_transform(fmri11_img)

    # We are losing a few regions due to masking
    n_regions_expected = n_regions - 7

    for t in signals:
        assert t.shape == (length, n_regions_expected)

        # ensure we can inverse transform even after losing a few regions
        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


@pytest.mark.slow
def test_resampling_to_clipped_labels(
    affine_eye, n_regions, length, img_labels, img_fmri
):
    """Test with clipped labels.

    Mask does not contain all labels.
    Shapes do matter in that case,
    because there is some resampling taking place.
    """
    shape2 = (8, 9, 10)  # mask

    _, mask22_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    # Multi-subject example
    fmri11_img = [img_fmri, img_fmri]

    masker = MultiNiftiLabelsMasker(
        img_labels,
        mask_img=mask22_img,
        resampling_target="labels",
        standardize=None,
    )

    signals = masker.fit_transform(fmri11_img)

    # We are losing a few regions due to masking
    n_regions_expected = n_regions - 4

    for t in signals:
        # We are losing a few regions due to clipping
        assert t.shape == (length, n_regions_expected)

        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        # ensure we can inverse transform even after losing a few regions
        masker.inverse_transform(t)
