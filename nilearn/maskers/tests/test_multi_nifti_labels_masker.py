"""Test the multi_nifti_labels_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils import data_gen, testing
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.exceptions import DimensionError
from nilearn.conftest import _affine_eye, _shape_3d_default
from nilearn.image import get_data
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker

extra_valid_checks = [
    "check_estimators_unfitted",
    "check_get_params_invariance",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[
            MultiNiftiLabelsMasker(
                data_gen.generate_labeled_regions(
                    _shape_3d_default(), affine=_affine_eye(), n_regions=9
                )
            ),
            NiftiLabelsMasker(
                data_gen.generate_labeled_regions(
                    _shape_3d_default(), affine=_affine_eye(), n_regions=9
                )
            ),
        ],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[
            MultiNiftiLabelsMasker(
                data_gen.generate_labeled_regions(
                    _shape_3d_default(), affine=_affine_eye(), n_regions=9
                )
            ),
            NiftiLabelsMasker(
                data_gen.generate_labeled_regions(
                    _shape_3d_default(), affine=_affine_eye(), n_regions=9
                )
            ),
        ],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_multi_nifti_labels_masker():
    # Check working of shape/affine checks
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)

    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    n_regions = 9
    length = 3

    fmri11_img, mask11_img = data_gen.generate_fake_fmri(
        shape1, affine=affine1, length=length
    )
    fmri12_img, mask12_img = data_gen.generate_fake_fmri(
        shape1, affine=affine2, length=length
    )
    fmri21_img, mask21_img = data_gen.generate_fake_fmri(
        shape2, affine=affine1, length=length
    )

    labels11_img = data_gen.generate_labeled_regions(
        shape1, affine=affine1, n_regions=n_regions
    )

    mask_img_4d = Nifti1Image(
        np.ones((2, 2, 2, 2), dtype=np.int8), affine=np.diag((4, 4, 4, 1))
    )

    # verify that 4D mask arguments are refused
    masker = MultiNiftiLabelsMasker(labels11_img, mask_img=mask_img_4d)
    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided "
        "a 4D image.",
    ):
        masker.fit()

    # check exception when transform() called without prior fit()
    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    with pytest.raises(ValueError, match="has not been fitted. "):
        masker11.transform(fmri11_img)

    # No exception raised here
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    # No exception should be raised either
    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    masker11.fit()
    masker11.inverse_transform(signals11)

    masker11 = MultiNiftiLabelsMasker(
        labels11_img, mask_img=mask11_img, resampling_target=None
    )
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker11.fit_transform(signals_input)
    assert len(signals11_list) == len(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # NiftiLabelsMasker should not work with 4D + 1D input
    signals_input = [fmri11_img, fmri11_img]
    masker11 = NiftiLabelsMasker(labels11_img, resampling_target=None)
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker11.fit_transform(signals_input)

    # Test all kinds of mismatch between shapes and between affines
    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    masker11.fit()
    with pytest.raises(ValueError):
        masker11.transform(fmri12_img)
    with pytest.raises(ValueError):
        masker11.transform(fmri21_img)

    masker11 = MultiNiftiLabelsMasker(
        labels11_img, mask_img=mask12_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker11.fit()

    masker11 = MultiNiftiLabelsMasker(
        labels11_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker11.fit()

    # Transform, with smoothing (smoke test)
    masker11 = MultiNiftiLabelsMasker(
        labels11_img, smoothing_fwhm=3, resampling_target=None
    )
    signals11_list = masker11.fit().transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    masker11 = MultiNiftiLabelsMasker(
        labels11_img, smoothing_fwhm=3, resampling_target=None
    )
    signals11_list = masker11.fit_transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

        with pytest.raises(ValueError, match="has not been fitted. "):
            MultiNiftiLabelsMasker(labels11_img).inverse_transform(signals)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker11.inverse_transform(signals)
        assert fmri11_img_r.shape == fmri11_img.shape
        np.testing.assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)


def test_multi_nifti_labels_masker_reduction_strategies():
    """Tests strategies of MultiNiftiLabelsMasker.

    1. whether the usage of different reduction strategies work
    2. whether unrecognized strategies raise a ValueError
    3. whether the default option is backwards compatible (calls "mean")
    """
    test_values = [-2.0, -1.0, 0.0, 1.0, 2]

    img_data = np.array([[test_values, test_values]])

    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    affine = np.eye(4)
    img = Nifti1Image(img_data, affine)
    labels = Nifti1Image(labels_data, affine)

    # What MultiNiftiLabelsMasker should return for each reduction strategy?
    expected_results = {
        "mean": np.mean(test_values),
        "median": np.median(test_values),
        "sum": np.sum(test_values),
        "minimum": np.min(test_values),
        "maximum": np.max(test_values),
        "standard_deviation": np.std(test_values),
        "variance": np.var(test_values),
    }

    for strategy, expected_result in expected_results.items():
        masker = MultiNiftiLabelsMasker(labels, strategy=strategy)
        # Here passing [img, img] within a list because it is multiple subjects
        # with a 3D object.
        results = masker.fit_transform([img, img])
        for result in results:
            assert result.squeeze() == expected_result

    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        MultiNiftiLabelsMasker(labels, strategy="TESTRAISE")

    default_masker = MultiNiftiLabelsMasker(labels)
    assert default_masker.strategy == "mean"


def test_multi_nifti_labels_masker_resampling(tmp_path):
    # Test resampling in MultiNiftiLabelsMasker
    shape1 = (10, 11, 12)
    affine = np.eye(4)

    # mask
    shape2 = (16, 17, 18)

    # labels
    shape3 = (13, 14, 15)

    n_regions = 9
    length = 3

    # With data of the same affine
    fmri11_img, _ = data_gen.generate_fake_fmri(
        shape1, affine=affine, length=length
    )
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine, length=length
    )

    labels33_img = data_gen.generate_labeled_regions(
        shape3, n_regions, affine=affine
    )

    # Multi-subject example
    fmri11_img = [fmri11_img, fmri11_img]

    # Test error checking
    with pytest.raises(ValueError):
        MultiNiftiLabelsMasker(
            labels33_img,
            resampling_target="mask",
        )
    with pytest.raises(ValueError):
        MultiNiftiLabelsMasker(
            labels33_img,
            resampling_target="invalid",
        )

    # Target: labels
    masker = MultiNiftiLabelsMasker(
        labels33_img, mask_img=mask22_img, resampling_target="labels"
    )

    masker.fit()
    np.testing.assert_almost_equal(
        masker.labels_img_.affine, labels33_img.affine
    )
    assert masker.labels_img_.shape == labels33_img.shape

    np.testing.assert_almost_equal(
        masker.mask_img_.affine, masker.labels_img_.affine
    )
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    for t in transformed:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)
        np.testing.assert_almost_equal(
            fmri11_img_r.affine, masker.labels_img_.affine
        )
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))

    # Test with clipped labels: mask does not contain all labels.
    # Shapes do matter in that case, because there is some resampling
    # taking place.
    shape1 = (10, 11, 12)  # fmri
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps

    n_regions = 9
    length = 21

    fmri11_img, _ = data_gen.generate_fake_fmri(
        shape1, affine=affine, length=length
    )
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine, length=length
    )

    # Multi-subject example
    fmri11_img = [fmri11_img, fmri11_img]

    # Target: labels
    labels33_img = data_gen.generate_labeled_regions(
        shape3, n_regions, affine=affine
    )

    masker = MultiNiftiLabelsMasker(
        labels33_img, mask_img=mask22_img, resampling_target="labels"
    )

    masker.fit()
    np.testing.assert_almost_equal(
        masker.labels_img_.affine, labels33_img.affine
    )
    assert masker.labels_img_.shape == labels33_img.shape

    np.testing.assert_almost_equal(
        masker.mask_img_.affine, masker.labels_img_.affine
    )
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    uniq_labels = np.unique(get_data(masker.labels_img_))
    assert uniq_labels[0] == 0
    assert len(uniq_labels) - 1 == n_regions

    transformed = masker.transform(fmri11_img)
    for t in transformed:
        assert t.shape == (length, n_regions)
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        fmri11_img_r = masker.inverse_transform(t)
        np.testing.assert_almost_equal(
            fmri11_img_r.affine, masker.labels_img_.affine
        )
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))

    # Test with data and atlas of different shape: the atlas should be
    # resampled to the data
    shape22 = (5, 5, 6)
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    fmri22_img, _ = data_gen.generate_fake_fmri(
        shape22, affine=affine2, length=length
    )
    masker = MultiNiftiLabelsMasker(labels33_img, mask_img=mask22_img)

    masker.fit_transform(fmri22_img)
    np.testing.assert_array_equal(
        masker._resampled_labels_img_.affine, affine2
    )

    # Test with filenames
    filename = testing.write_imgs_to_path(fmri22_img, file_path=tmp_path)
    masker = MultiNiftiLabelsMasker(labels33_img, resampling_target="data")
    masker.fit_transform(filename)

    # test labels masker with resampling target in 'data', 'labels' to return
    # resampled labels having number of labels equal with transformed shape of
    # 2nd dimension. This tests are added based on issue #1673 in Nilearn
    shape = (13, 11, 12)
    affine = np.eye(4) * 2

    fmri_img, _ = data_gen.generate_fake_fmri(shape, affine=affine, length=21)
    labels_img = data_gen.generate_labeled_regions(
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
                transformed = masker.fit_transform(fmri_img)
        else:
            transformed = masker.fit_transform(fmri_img)
        resampled_labels_img = masker._resampled_labels_img_
        n_resampled_labels = len(np.unique(get_data(resampled_labels_img)))
        assert n_resampled_labels - 1 == transformed.shape[1]
        # inverse transform
        compressed_img = masker.inverse_transform(transformed)

        # Test that compressing the image a second time should yield an image
        # with the same data as compressed_img.
        transformed2 = masker.fit_transform(fmri_img)
        # inverse transform again
        compressed_img2 = masker.inverse_transform(transformed2)
        np.testing.assert_array_equal(
            get_data(compressed_img), get_data(compressed_img2)
        )


def test_multi_nifti_labels_masker_list_of_sample_mask():
    """Tests MultiNiftiLabelsMasker.fit_transform with a list of "sample_mask".

    "sample_mask" was directly sent as input to the parallel calls of
    "transform_single_imgs" instead of sending iterations.
    See https://github.com/nilearn/nilearn/issues/3967 for more details.
    """
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)

    n_regions = 9
    length = 6
    n_scrub1 = 3
    n_scrub2 = 2

    fmri11_img, mask11_img = data_gen.generate_fake_fmri(
        shape1, affine=affine1, length=length
    )

    labels11_img = data_gen.generate_labeled_regions(
        shape1, affine=affine1, n_regions=n_regions
    )
    sample_mask1 = np.arange(length - n_scrub1)
    sample_mask2 = np.arange(length - n_scrub2)

    masker = MultiNiftiLabelsMasker(labels11_img)
    ts_list = masker.fit_transform(
        [fmri11_img, fmri11_img], sample_mask=[sample_mask1, sample_mask2]
    )

    assert len(ts_list) == 2
    for ts, n_scrub in zip(ts_list, [n_scrub1, n_scrub2]):
        assert ts.shape == (length - n_scrub, n_regions)
