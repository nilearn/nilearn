"""Test the multi_nifti_labels_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal

from nilearn._utils import data_gen, testing
from nilearn._utils.exceptions import DimensionError
from nilearn.image import get_data
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker


@pytest.fixture()
def n_regions():
    return 9


@pytest.fixture()
def length():
    return 3


def test_multi_nifti_labels_masker_errors(
    shape_3d_default, affine_eye, img_4d_ones_eye, n_regions, length
):
    labels_img = data_gen.generate_labeled_regions(
        shape=shape_3d_default, affine=affine_eye, n_regions=n_regions
    )

    # verify that 4D mask arguments are refused
    masker = MultiNiftiLabelsMasker(labels_img, mask_img=img_4d_ones_eye)

    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided a 4D image.",
    ):
        masker.fit()

    # check exception when transform() called without prior fit()
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiLabelsMasker(labels_img, resampling_target=None)

    with pytest.raises(ValueError, match="has not been fitted. "):
        masker.transform(fmri_img)

    # No exception raised here
    signals = masker.fit().transform(fmri_img)
    assert signals.shape == (length, n_regions)

    # No exception should be raised either
    masker = MultiNiftiLabelsMasker(labels_img, resampling_target=None)
    masker.fit()
    masker.inverse_transform(signals)

    # NiftiLabelsMasker should not work with 4D + 1D input
    labels_img = data_gen.generate_labeled_regions(
        shape=shape_3d_default, affine=affine_eye, n_regions=n_regions
    )
    signals_input = [fmri_img, fmri_img]
    masker = NiftiLabelsMasker(labels_img, resampling_target=None)

    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker.fit_transform(signals_input)


def test_multi_nifti_labels_masker_errors_shape_affine(
    shape_3d_default, affine_eye, n_regions, length
):
    """Test all kinds of mismatch between shapes and between affines."""
    labels_img = data_gen.generate_labeled_regions(
        shape=shape_3d_default, affine=affine_eye, n_regions=n_regions
    )

    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    fmri12_img, mask12_img = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine2, length=length
    )
    fmri21_img, mask21_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    masker = MultiNiftiLabelsMasker(labels_img, resampling_target=None)
    masker.fit()
    with pytest.raises(ValueError):
        masker.transform(fmri12_img)
    with pytest.raises(ValueError):
        masker.transform(fmri21_img)

    masker = MultiNiftiLabelsMasker(
        labels_img, mask_img=mask12_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker.fit()

    masker = MultiNiftiLabelsMasker(
        labels_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker.fit()


def test_multi_nifti_labels_masker(
    shape_3d_default, affine_eye, n_regions, length
):
    # Check working of shape/affine checks
    fmri11_img, mask11_img = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )

    labels11_img = data_gen.generate_labeled_regions(
        shape=shape_3d_default, affine=affine_eye, n_regions=n_regions
    )

    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

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
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)


@pytest.mark.parametrize(
    "strategy, fun",
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
    strategy, fun, affine_eye
):
    """Test whether the usage of different reduction strategies work."""
    test_values = [-2.0, -1.0, 0.0, 1.0, 2]

    img_data = np.array([[test_values, test_values]])

    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    img = Nifti1Image(img_data, affine_eye)
    labels = Nifti1Image(labels_data, affine_eye)

    # What MultiNiftiLabelsMasker should return for each reduction strategy
    expected_result = fun(test_values)

    masker = MultiNiftiLabelsMasker(labels, strategy=strategy)
    # Here passing [img, img] within a list because it is multiple subjects
    # with a 3D object.
    results = masker.fit_transform([img, img])
    for result in results:
        assert result.squeeze() == expected_result


def test_multi_nifti_labels_masker_defaults(affine_eye):
    """Tests:
    1. whether unrecognised strategies raise a ValueError
    2. whether the default option is backwards compatible (calls "mean")
    """
    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    labels = Nifti1Image(labels_data, affine_eye)

    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        MultiNiftiLabelsMasker(labels, strategy="TESTRAISE")

    default_masker = MultiNiftiLabelsMasker(labels)
    assert default_masker.strategy == "mean"


def test_multi_nifti_labels_masker_resampling(
    shape_3d_default, affine_eye, n_regions, length
):
    # Test resampling in MultiNiftiLabelsMasker
    # With data of the same affine
    # mask
    shape2 = (16, 17, 18)
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    # labels
    shape3 = (13, 14, 15)
    labels33_img = data_gen.generate_labeled_regions(
        shape3, n_regions, affine=affine_eye
    )

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
    assert_almost_equal(masker.labels_img_.affine, labels33_img.affine)
    assert masker.labels_img_.shape == labels33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    # Multi-subject example
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )

    fmri_img = [fmri_img, fmri_img]

    transformed = masker.transform(fmri_img)
    for t in transformed:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)
        assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_multi_nifti_labels_masker_resampling_clipped(
    shape_3d_default, affine_eye, n_regions, length
):
    # Test with clipped labels: mask does not contain all labels.
    # Shapes do matter in that case, because there is some resampling
    # taking place.
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps

    fmri11_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    # Multi-subject example
    fmri11_img = [fmri11_img, fmri11_img]

    # Target: labels
    labels33_img = data_gen.generate_labeled_regions(
        shape3, n_regions, affine=affine_eye
    )

    masker = MultiNiftiLabelsMasker(
        labels33_img, mask_img=mask22_img, resampling_target="labels"
    )

    masker.fit()
    assert_almost_equal(masker.labels_img_.affine, labels33_img.affine)
    assert masker.labels_img_.shape == labels33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
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
        assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
        assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_multi_nifti_labels_masker_data_atlas_different_shape(
    affine_eye, n_regions, length
):
    # mask
    shape2 = (8, 9, 10)
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    shape3 = (16, 18, 20)  # maps
    # Target: labels
    labels33_img = data_gen.generate_labeled_regions(
        shape3, n_regions, affine=affine_eye
    )

    # Test with data and atlas of different shape: the atlas should be
    # resampled to the data
    shape22 = (5, 5, 6)
    affine2 = 2 * affine_eye
    affine2[-1, -1] = 1

    fmri22_img, _ = data_gen.generate_fake_fmri(
        shape22, affine=affine2, length=length
    )
    masker = MultiNiftiLabelsMasker(labels33_img, mask_img=mask22_img)

    masker.fit_transform(fmri22_img)
    assert_array_equal(masker._resampled_labels_img_.affine, affine2)

    # Test with filenames
    with testing.write_tmp_imgs(fmri22_img) as filename:
        masker = MultiNiftiLabelsMasker(labels33_img, resampling_target="data")
        masker.fit_transform(filename)


@pytest.mark.parametrize("resampling_target", ["data", "labels"])
def test_multi_nifti_labels_masker_resampling_target(
    resampling_target, shape_3d_default, affine_eye
):
    """Test labels masker with resampling target in 'data', 'labels' to return
    resampled labels having number of labels equal
    with transformed shape of 2nd dimension.

    These tests are added based on issue #1673 in Nilearn.
    """
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye * 2, length=21
    )

    labels_img = data_gen.generate_labeled_regions(
        (9, 8, 6), affine=affine_eye, n_regions=10
    )

    masker = MultiNiftiLabelsMasker(
        labels_img=labels_img, resampling_target=resampling_target
    )
    if resampling_target == "data":
        with pytest.warns(
            UserWarning,
            match=(
                "After resampling the label image to the data image, "
                "the following labels were removed"
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
    assert_array_equal(get_data(compressed_img), get_data(compressed_img2))
