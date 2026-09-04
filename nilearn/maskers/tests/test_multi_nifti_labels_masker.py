"""Test the multi_nifti_labels_masker module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_labeled_regions,
)
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.conftest import _img_labels
from nilearn.image import get_data
from nilearn.maskers import MultiNiftiLabelsMasker

ESTIMATORS_TO_CHECK = [MultiNiftiLabelsMasker()]

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


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            MultiNiftiLabelsMasker(labels_img=_img_labels(), standardize=None),
            MultiNiftiLabelsMasker(
                labels_img=_img_labels(n_regions=1), standardize=None
            ),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


def test_multi_nifti_labels_masker(
    affine_eye, n_regions, shape_3d_default, length, img_labels
):
    """Check working of shape/affine checks."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker11 = MultiNiftiLabelsMasker(
        img_labels, resampling_target=None, standardize=None
    )

    # No exception raised here
    signals11 = masker11.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    # No exception should be raised either
    masker11 = MultiNiftiLabelsMasker(
        img_labels, resampling_target=None, standardize=None
    )
    masker11.fit()
    masker11.inverse_transform(signals11)

    masker11 = MultiNiftiLabelsMasker(
        img_labels,
        mask_img=mask11_img,
        resampling_target=None,
        keep_masked_labels=True,
        standardize=None,
    )
    with pytest.warns(
        FutureWarning, match='"keep_masked_labels" parameter will be removed'
    ):
        signals11 = masker11.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    with pytest.warns(
        FutureWarning, match='"keep_masked_labels" parameter will be removed'
    ):
        signals11_list = masker11.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    masker11 = MultiNiftiLabelsMasker(
        img_labels, resampling_target=None, standardize=None
    )
    signals11_list = masker11.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker11.inverse_transform(signals)

        assert fmri11_img_r.shape == fmri11_img.shape
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)


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

    # Target: labels
    masker = MultiNiftiLabelsMasker(
        img_labels,
        mask_img=mask22_img,
        resampling_target="labels",
        keep_masked_labels=True,
        standardize=None,
    )

    fmri11_img = [fmri11_img, fmri11_img]

    with pytest.warns(
        FutureWarning, match='"keep_masked_labels" parameter will be removed'
    ):
        signals = masker.fit_transform(fmri11_img)

    assert_almost_equal(masker.labels_img_.affine, img_labels.affine)
    assert masker.labels_img_.shape == img_labels.shape

    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    for t in signals:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)
        assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
        assert fmri11_img_r.shape == ((*masker.labels_img_.shape[:3], length))


def test_resampling_target():
    """Test labels masker with resampling target in 'data', 'labels'.

    Must return resampled labels having number of labels
    equal with transformed shape of 2nd dimension.

    This tests are added based on issue #1673 in Nilearn.
    """
    shape = (13, 11, 12)
    affine = np.eye(4) * 2

    fmri_img, _ = generate_fake_fmri(shape, affine=affine, length=21)
    labels_img = generate_labeled_regions(
        (9, 8, 6), affine=np.eye(4), n_regions=10
    )
    for resampling_target in ["data", "labels"]:
        masker = MultiNiftiLabelsMasker(
            labels_img=labels_img,
            resampling_target=resampling_target,
            keep_masked_labels=True,
            standardize=None,
        )
        if resampling_target == "data":
            with (
                pytest.warns(
                    UserWarning,
                    match=(
                        "After resampling the label image "
                        "to the data image, the following "
                        "labels were removed"
                    ),
                ),
                pytest.warns(
                    FutureWarning,
                    match=(
                        r"In version 0.15.0, "
                        '"keep_masked_labels" parameter will be removed'
                    ),
                ),
            ):
                signals = masker.fit_transform(fmri_img)
        else:
            with pytest.warns(
                FutureWarning,
                match=(
                    r"In version 0.15.0, "
                    '"keep_masked_labels" parameter will be removed'
                ),
            ):
                signals = masker.fit_transform(fmri_img)

        resampled_labels_img = masker.labels_img_
        n_resampled_labels = len(np.unique(get_data(resampled_labels_img)))
        assert n_resampled_labels - 1 == signals.shape[1]

        # inverse transform
        compressed_img = masker.inverse_transform(signals)

        # Test that compressing the image a second time should yield an image
        # with the same data as compressed_img.
        with pytest.warns(
            FutureWarning,
            match='"keep_masked_labels" parameter will be removed',
        ):
            signals2 = masker.fit_transform(fmri_img)

        # inverse transform again
        compressed_img2 = masker.inverse_transform(signals2)

        assert_array_equal(get_data(compressed_img), get_data(compressed_img2))
