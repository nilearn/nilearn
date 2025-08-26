"""Test the multi_nifti_labels_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
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
from nilearn._utils.tags import SKLEARN_LT_1_6
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


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[MultiNiftiLabelsMasker(labels_img=_img_labels())]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.timeout(0)
def test_multi_nifti_labels_masker(
    affine_eye, n_regions, shape_3d_default, length, img_labels
):
    """Check working of shape/affine checks."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)

    # No exception raised here
    signals11 = masker11.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    # No exception should be raised either
    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)
    masker11.fit()
    masker11.inverse_transform(signals11)

    masker11 = MultiNiftiLabelsMasker(
        img_labels, mask_img=mask11_img, resampling_target=None
    )
    signals11 = masker11.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker11.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)
    signals11_list = masker11.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker11.inverse_transform(signals)

        assert fmri11_img_r.shape == fmri11_img.shape
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)


def test_multi_nifti_labels_masker_errors(
    affine_eye, shape_3d_default, length, img_labels
):
    """Test errors in MultiNiftiLabelsMasker."""
    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    fmri12_img, mask12_img = generate_fake_fmri(
        shape_3d_default, affine=affine2, length=length
    )
    fmri21_img, mask21_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    # Test all kinds of mismatch between shapes and between affines
    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)
    masker11.fit()

    with pytest.raises(
        ValueError, match="Images have different affine matrices."
    ):
        masker11.transform(fmri12_img)

    with pytest.raises(ValueError, match="Images have incompatible shapes."):
        masker11.transform(fmri21_img)

    masker11 = MultiNiftiLabelsMasker(
        img_labels, mask_img=mask12_img, resampling_target=None
    )

    with pytest.raises(
        ValueError, match="Following field of view errors were detected"
    ):
        masker11.fit()

    masker11 = MultiNiftiLabelsMasker(
        img_labels, mask_img=mask21_img, resampling_target=None
    )

    with pytest.raises(
        ValueError, match="Following field of view errors were detected"
    ):
        masker11.fit()


def test_multi_nifti_labels_masker_errors_strategy(img_labels):
    """Test strategy errors."""
    masker = MultiNiftiLabelsMasker(img_labels, strategy="TESTRAISE")
    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        masker.fit()


@pytest.mark.parametrize("resampling_target", ["mask", "invalid"])
def test_multi_nifti_labels_masker_errors_resampling(
    img_labels, resampling_target
):
    """Test error checking resampling_target."""
    masker = MultiNiftiLabelsMasker(
        img_labels,
        resampling_target=resampling_target,
    )
    with pytest.raises(
        ValueError, match="invalid value for 'resampling_target' parameter"
    ):
        masker.fit()


@pytest.mark.parametrize("test_values", [[-2.0, -1.0, 0.0, 1.0, 2]])
@pytest.mark.parametrize(
    "strategy, fn",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("sum", np.sum),
        ("minimum", np.min),
        ("maximum", np.max),
        ("standard_deviation", np.std),
        ("variance", np.var),
    ],
)
def test_multi_nifti_labels_masker_reduction_strategies(
    affine_eye, test_values, strategy, fn
):
    """Tests strategies of MultiNiftiLabelsMasker.

    - whether the usage of different reduction strategies work
    - whether the default option is backwards compatible (calls "mean")
    """
    img_data = np.array([[test_values, test_values]])

    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    img = Nifti1Image(img_data, affine_eye)
    labels = Nifti1Image(labels_data, affine_eye)

    masker = MultiNiftiLabelsMasker(labels, strategy=strategy)
    # Here passing [img, img] within a list because it is multiple subjects
    # with a 3D object.
    results = masker.fit_transform([img, img])

    # What MultiNiftiLabelsMasker should return for each reduction strategy?
    expected_result = fn(test_values)

    for r in results:
        assert r.squeeze() == expected_result

    default_masker = MultiNiftiLabelsMasker(labels)
    assert default_masker.strategy == "mean"


def test_multi_nifti_labels_masker_resampling(
    affine_eye, n_regions, length, img_labels
):
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
        img_labels, mask_img=mask22_img, resampling_target="labels"
    )

    fmri11_img = [fmri11_img, fmri11_img]

    signals = masker.fit_transform(fmri11_img)

    assert_almost_equal(masker.labels_img_.affine, img_labels.affine)
    assert masker.labels_img_.shape == img_labels.shape

    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    for t in signals:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)
        assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_multi_nifti_labels_masker_resampling_clipped_labels(
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
        img_labels, mask_img=mask22_img, resampling_target="labels"
    )

    signals = masker.fit_transform(fmri11_img)

    assert_almost_equal(masker.labels_img_.affine, img_labels.affine)
    assert masker.labels_img_.shape == img_labels.shape
    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]
    uniq_labels = np.unique(get_data(masker.labels_img_))
    assert uniq_labels[0] == 0
    assert len(uniq_labels) - 1 == n_regions

    for t in signals:
        assert t.shape == (length, n_regions)
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        fmri11_img_r = masker.inverse_transform(t)

        assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_multi_nifti_labels_masker_atlas_data_different_fov(
    affine_eye, img_labels, length
):
    """Test with data and atlas of different shape.

    The atlas should be resampled to the data.
    """
    shape2 = (8, 9, 10)  # mask
    shape22 = (5, 5, 6)
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    _, mask22_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    fmri22_img, _ = generate_fake_fmri(shape22, affine=affine2, length=length)
    masker = MultiNiftiLabelsMasker(img_labels, mask_img=mask22_img)

    masker.fit_transform(fmri22_img)

    assert_array_equal(masker.labels_img_.affine, affine2)


def test_multi_nifti_labels_masker_resampling_target():
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
            labels_img=labels_img, resampling_target=resampling_target
        )
        if resampling_target == "data":
            with pytest.warns(
                UserWarning,
                match=(
                    "After resampling the label image "
                    "to the data image, the following "
                    "labels were removed"
                ),
            ):
                signals = masker.fit_transform(fmri_img)
        else:
            signals = masker.fit_transform(fmri_img)

        resampled_labels_img = masker.labels_img_
        n_resampled_labels = len(np.unique(get_data(resampled_labels_img)))
        assert n_resampled_labels - 1 == signals.shape[1]

        # inverse transform
        compressed_img = masker.inverse_transform(signals)

        # Test that compressing the image a second time should yield an image
        # with the same data as compressed_img.
        signals2 = masker.fit_transform(fmri_img)

        # inverse transform again
        compressed_img2 = masker.inverse_transform(signals2)

        assert_array_equal(get_data(compressed_img), get_data(compressed_img2))
