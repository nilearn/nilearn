"""Test the nifti_region module

Functions in this file only test features added by the NiftiLabelsMasker class,
not the underlying functions (clean(), img_to_signals_labels(), etc.). See
test_masking.py and test_signal.py for details.
"""

import warnings

import nibabel
import numpy as np
import pytest

from nilearn._utils import data_gen, testing
from nilearn._utils.exceptions import DimensionError
from nilearn.image import get_data
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker


def test_nifti_labels_masker():
    """Check working of shape/affine checks."""
    length = 3
    shape1 = (13, 11, 12, length)
    affine1 = np.eye(4)

    shape2 = (12, 10, 14, length)
    affine2 = np.diag((1, 2, 3, 1))

    n_regions = 9

    fmri11_img, mask11_img = data_gen.generate_random_img(
        shape1,
        affine=affine1,
    )
    fmri12_img, mask12_img = data_gen.generate_random_img(
        shape1,
        affine=affine2,
    )
    fmri21_img, mask21_img = data_gen.generate_random_img(
        shape2,
        affine=affine1,
    )

    labels11_img = data_gen.generate_labeled_regions(
        shape1[:3],
        affine=affine1,
        n_regions=n_regions,
    )

    mask_img_4d = nibabel.Nifti1Image(
        np.ones((2, 2, 2, 2), dtype=np.int8),
        affine=np.diag((4, 4, 4, 1)),
    )

    # verify that 4D mask arguments are refused
    masker = NiftiLabelsMasker(labels11_img, mask_img=mask_img_4d)
    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided "
        "a 4D image.",
    ):
        masker.fit()

    # check exception when transform() called without prior fit()
    masker11 = NiftiLabelsMasker(labels11_img, resampling_target=None)
    with pytest.raises(ValueError, match="has not been fitted. "):
        masker11.transform(fmri11_img)

    # No exception raised here
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    # No exception should be raised either
    masker11 = NiftiLabelsMasker(labels11_img, resampling_target=None)

    # Check attributes defined at fit
    assert not hasattr(masker11, "mask_img_")
    assert not hasattr(masker11, "labels_img_")
    assert not hasattr(masker11, "n_elements_")

    masker11.fit()

    # Check attributes defined at fit
    assert hasattr(masker11, "mask_img_")
    assert hasattr(masker11, "labels_img_")
    assert hasattr(masker11, "n_elements_")
    assert masker11.n_elements_ == n_regions

    masker11.inverse_transform(signals11)

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask11_img, resampling_target=None
    )
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    # Test all kinds of mismatch between shapes and between affines
    masker11 = NiftiLabelsMasker(labels11_img, resampling_target=None)
    masker11.fit()
    with pytest.raises(ValueError):
        masker11.transform(fmri12_img)
    with pytest.raises(ValueError):
        masker11.transform(fmri21_img)

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask12_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker11.fit()

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker11.fit()

    # Transform, with smoothing (smoke test)
    masker11 = NiftiLabelsMasker(
        labels11_img, smoothing_fwhm=3, resampling_target=None
    )
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    masker11 = NiftiLabelsMasker(
        labels11_img, smoothing_fwhm=3, resampling_target=None
    )
    signals11 = masker11.fit_transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    with pytest.raises(ValueError, match="has not been fitted. "):
        NiftiLabelsMasker(labels11_img).inverse_transform(signals11)

    # Call inverse transform (smoke test)
    fmri11_img_r = masker11.inverse_transform(signals11)
    assert fmri11_img_r.shape == fmri11_img.shape
    np.testing.assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)


def test_nifti_labels_masker_io_shapes(rng):
    """Ensure that NiftiLabelsMasker handles 1D/2D/3D/4D data appropriately.

    transform(4D image) --> 2D output, no warning
    transform(3D image) --> 2D output, DeprecationWarning
    inverse_transform(2D array) --> 4D image, no warning
    inverse_transform(1D array) --> 3D image, no warning
    inverse_transform(2D array with wrong shape) --> IndexError
    """
    n_regions, n_volumes = 9, 5
    shape_3d = (10, 11, 12)
    shape_4d = (10, 11, 12, n_volumes)
    data_1d = rng.random(n_regions)
    data_2d = rng.random((n_volumes, n_regions))
    affine = np.eye(4)

    img_4d, mask_img = data_gen.generate_random_img(
        shape_4d,
        affine=affine,
    )
    img_3d, _ = data_gen.generate_random_img(shape_3d, affine=affine)
    labels_img = data_gen.generate_labeled_regions(
        shape_3d,
        affine=affine,
        n_regions=n_regions,
    )
    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)
    masker.fit()

    # DeprecationWarning *should* be raised for 3D inputs
    with pytest.warns(DeprecationWarning, match="Starting in version 0.12"):
        test_data = masker.transform(img_3d)
        assert test_data.shape == (1, n_regions)

    # DeprecationWarning should *not* be raised for 4D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_data = masker.transform(img_4d)
        assert test_data.shape == (n_volumes, n_regions)

    # DeprecationWarning should *not* be raised for 1D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_1d)
        assert test_img.shape == shape_3d

    # DeprecationWarning should *not* be raised for 2D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_2d)
        assert test_img.shape == shape_4d

    with pytest.raises(IndexError):
        masker.inverse_transform(data_2d.T)


def test_nifti_labels_masker_with_nans_and_infs():
    """Apply a NiftiLabelsMasker containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    length = 3
    n_regions = 9
    fmri_img, mask_img = data_gen.generate_random_img(
        (13, 11, 12, length),
        affine=np.eye(4),
    )
    labels_img = data_gen.generate_labeled_regions(
        (13, 11, 12),
        affine=np.eye(4),
        n_regions=n_regions,
    )
    # Introduce nans with data type float
    # See issue: https://github.com/nilearn/nilearn/issues/2580
    labels_data = get_data(labels_img).astype(np.float32)
    labels_data[:, :, 7] = np.nan
    labels_data[:, :, 4] = np.inf
    labels_img = nibabel.Nifti1Image(labels_data, np.eye(4))

    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_labels_masker_with_nans_and_infs_in_mask():
    """Apply a NiftiLabelsMasker with a mask containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    length = 3
    n_regions = 9
    fmri_img, mask_img = data_gen.generate_random_img(
        (13, 11, 12, length),
        affine=np.eye(4),
    )
    labels_img = data_gen.generate_labeled_regions(
        (13, 11, 12),
        affine=np.eye(4),
        n_regions=n_regions,
    )
    # Introduce nans with data type float
    # See issue: https://github.com/nilearn/nilearn/issues/2580
    mask_data = get_data(mask_img).astype(np.float32)
    mask_data[:, :, 7] = np.nan
    mask_data[:, :, 4] = np.inf
    mask_img = nibabel.Nifti1Image(mask_data, np.eye(4))

    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_labels_masker_with_nans_and_infs_in_data():
    """Apply a NiftiLabelsMasker to 4D data containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    length = 3
    n_regions = 9
    fmri_img, mask_img = data_gen.generate_random_img(
        (13, 11, 12, length),
        affine=np.eye(4),
    )
    labels_img = data_gen.generate_labeled_regions(
        (13, 11, 12),
        affine=np.eye(4),
        n_regions=n_regions,
    )
    # Introduce nans with data type float
    # See issues:
    # - https://github.com/nilearn/nilearn/issues/2580 (why floats)
    # - https://github.com/nilearn/nilearn/issues/2711 (why test)
    fmri_data = get_data(fmri_img).astype(np.float32)
    fmri_data[:, :, 7, :] = np.nan
    fmri_data[:, :, 4, 0] = np.inf
    fmri_img = nibabel.Nifti1Image(fmri_data, np.eye(4))

    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_labels_masker_reduction_strategies():
    """Tests:
    1. whether the usage of different reduction strategies work.
    2. whether unrecognised strategies raise a ValueError
    3. whether the default option is backwards compatible (calls "mean")
    """
    test_values = [-2.0, -1.0, 0.0, 1.0, 2]

    img_data = np.array([[test_values, test_values]])

    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    affine = np.eye(4)
    img = nibabel.Nifti1Image(img_data, affine)
    labels = nibabel.Nifti1Image(labels_data, affine)

    # What NiftiLabelsMasker should return for each reduction strategy?
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
        masker = NiftiLabelsMasker(labels, strategy=strategy)
        # Here passing [img] within a list because it's a 3D object.
        result = masker.fit_transform([img]).squeeze()
        assert result == expected_result

    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        NiftiLabelsMasker(labels, strategy="TESTRAISE")

    default_masker = NiftiLabelsMasker(labels)
    assert default_masker.strategy == "mean"


def test_nifti_labels_masker_resampling(tmp_path):
    """Test resampling in NiftiLabelsMasker."""
    n_regions = 9
    length = 3

    shape1 = (10, 11, 12, length)
    affine = np.eye(4)

    # mask
    shape2 = (16, 17, 18, length)

    # labels
    shape3 = (13, 14, 15)

    # With data of the same affine
    fmri11_img, _ = data_gen.generate_random_img(
        shape1,
        affine=affine,
    )
    _, mask22_img = data_gen.generate_random_img(
        shape2,
        affine=affine,
    )

    labels33_img = data_gen.generate_labeled_regions(
        shape3,
        n_regions,
        affine=affine,
    )

    # Test error checking
    with pytest.raises(ValueError):
        NiftiLabelsMasker(labels33_img, resampling_target="mask")
    with pytest.raises(ValueError):
        NiftiLabelsMasker(
            labels33_img,
            resampling_target="invalid",
        )

    # Target: labels
    masker = NiftiLabelsMasker(
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
    assert transformed.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(
        fmri11_img_r.affine, masker.labels_img_.affine
    )
    assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))

    # Test with clipped labels: mask does not contain all labels.
    # Shapes do matter in that case, because there is some resampling
    # taking place.
    n_regions = 9
    length = 21

    shape1 = (10, 11, 12, length)  # fmri
    shape2 = (8, 9, 10, length)  # mask
    shape3 = (16, 18, 20)  # maps

    fmri11_img, _ = data_gen.generate_random_img(
        shape1,
        affine=affine,
    )
    _, mask22_img = data_gen.generate_random_img(
        shape2,
        affine=affine,
    )

    # Target: labels
    labels33_img = data_gen.generate_labeled_regions(
        shape3, n_regions, affine=affine
    )

    masker = NiftiLabelsMasker(
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
    assert transformed.shape == (length, n_regions)
    # Some regions have been clipped. Resulting signal must be zero
    assert (transformed.var(axis=0) == 0).sum() < n_regions

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(
        fmri11_img_r.affine, masker.labels_img_.affine
    )
    assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))

    # Test with data and atlas of different shape: the atlas should be
    # resampled to the data
    shape22 = (5, 5, 6, length)
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    fmri22_img, _ = data_gen.generate_random_img(
        shape22,
        affine=affine2,
    )
    masker = NiftiLabelsMasker(labels33_img, mask_img=mask22_img)

    masker.fit_transform(fmri22_img)
    np.testing.assert_array_equal(
        masker._resampled_labels_img_.affine, affine2
    )

    # Test with filenames
    filename = testing.write_imgs_to_path(fmri22_img, file_path=tmp_path)
    masker = NiftiLabelsMasker(labels33_img, resampling_target="data")
    masker.fit_transform(filename)

    # test labels masker with resampling target in 'data', 'labels' to return
    # resampled labels having number of labels equal with transformed shape of
    # 2nd dimension. This tests are added based on issue #1673 in Nilearn
    shape = (13, 11, 12, 21)
    affine = np.eye(4) * 2

    fmri_img, _ = data_gen.generate_random_img(shape, affine=affine)
    labels_img = data_gen.generate_labeled_regions(
        (9, 8, 6), affine=np.eye(4), n_regions=10
    )
    for resampling_target in ["data", "labels"]:
        masker = NiftiLabelsMasker(
            labels_img=labels_img, resampling_target=resampling_target
        )
        if resampling_target == "data":
            with pytest.warns(
                UserWarning,
                match="After resampling the label image "
                "to the data image, the following "
                "labels were removed",
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


def test_standardization(rng):
    data_shape = (9, 9, 5)
    n_samples = 500

    signals = rng.standard_normal(size=(np.prod(data_shape), n_samples))
    means = rng.standard_normal(size=(np.prod(data_shape), 1)) * 50 + 1000
    signals += means
    img = nibabel.Nifti1Image(
        signals.reshape(data_shape + (n_samples,)),
        np.eye(4),
    )

    labels = data_gen.generate_labeled_regions((9, 9, 5), 10)

    # Unstandarized
    masker = NiftiLabelsMasker(labels, standardize=False)
    unstandarized_label_signals = masker.fit_transform(img)

    # z-score
    masker = NiftiLabelsMasker(labels, standardize="zscore_sample")
    trans_signals = masker.fit_transform(img)

    np.testing.assert_almost_equal(trans_signals.mean(0), 0)
    np.testing.assert_almost_equal(trans_signals.std(0), 1, decimal=3)

    # psc
    masker = NiftiLabelsMasker(labels, standardize="psc")
    trans_signals = masker.fit_transform(img)

    np.testing.assert_almost_equal(trans_signals.mean(0), 0)
    np.testing.assert_almost_equal(
        trans_signals,
        (
            unstandarized_label_signals
            / unstandarized_label_signals.mean(0)
            * 100
            - 100
        ),
    )


def test_nifti_labels_masker_with_mask():
    """Test NiftiLabelsMasker with a separate mask_img parameter."""
    shape = (13, 11, 12, 3)
    affine = np.eye(4)
    fmri_img, mask_img = data_gen.generate_random_img(shape, affine=affine)
    labels_img = data_gen.generate_labeled_regions(
        shape[:3],
        affine=affine,
        n_regions=7,
    )
    masker = NiftiLabelsMasker(
        labels_img, resampling_target=None, mask_img=mask_img
    )
    signals = masker.fit().transform(fmri_img)
    bg_masker = NiftiMasker(mask_img).fit()
    masked_labels = bg_masker.inverse_transform(
        bg_masker.transform(labels_img),
    )
    masked_masker = NiftiLabelsMasker(
        masked_labels, resampling_target=None, mask_img=mask_img
    )
    masked_signals = masked_masker.fit().transform(fmri_img)
    assert np.allclose(signals, masked_signals)

    #  masker.region_atlas_ should be the same as the masked_labels
    # masked_labels is a 4D image with shape (13,11,12,1)
    masked_labels_data = get_data(masked_labels)[:, :, :, 0]
    assert np.allclose(get_data(masker.region_atlas_), masked_labels_data)


def test_region_names():
    """Test region_names_ attribute in NiftiLabelsMasker"""
    shape = (13, 11, 12, 3)
    affine = np.eye(4)
    fmri_img, _ = data_gen.generate_random_img(shape, affine=affine)
    labels_img = data_gen.generate_labeled_regions(
        shape[:3],
        affine=affine,
        n_regions=7,
    )

    # define region_names
    region_names = ["background"] + ["region_" + str(i + 1) for i in range(7)]

    masker = NiftiLabelsMasker(
        labels_img,
        labels=region_names,
        resampling_target=None,
    )
    _ = masker.fit().transform(fmri_img)

    region_names_after_fit = [
        masker.region_names_[i] for i in masker.region_names_
    ]
    region_names_after_fit.sort()
    region_names.sort()
    region_names.pop(region_names.index("background"))
    assert region_names_after_fit == region_names


def test_missing_labels_due_to_resampling(affine_eye):
    """Test region_names_ matches signals after resampling drops labels."""
    shape = (13, 11, 12, 3)
    affine_low = np.diag((4, 4, 4, 4))
    fmri_img, _ = data_gen.generate_random_img(shape, affine=affine_low)
    labels_img = data_gen.generate_labeled_regions(
        shape[:3],
        affine=affine_eye,
        n_regions=7,
    )

    # define region_names
    region_names = ["background"] + ["region_" + str(i + 1) for i in range(7)]

    masker = NiftiLabelsMasker(
        labels_img,
        labels=region_names,
        resampling_target="data",
    )
    signals = masker.fit().transform(fmri_img)

    assert len(masker.region_names_) == signals.shape[1]
    assert len(list(masker.region_ids_.items())[1:]) == signals.shape[1]


def test_3d_images():
    # Test that the NiftiLabelsMasker works with 3D images
    affine = np.eye(4)
    n_regions = 3
    shape3 = (2, 2, 2)

    labels33_img = data_gen.generate_labeled_regions(shape3, n_regions)
    mask_img = nibabel.Nifti1Image(
        np.ones(shape3, dtype=np.int8),
        affine=affine,
    )
    epi_img1 = nibabel.Nifti1Image(np.ones(shape3), affine=affine)
    epi_img2 = nibabel.Nifti1Image(np.ones(shape3), affine=affine)
    masker = NiftiLabelsMasker(labels33_img, mask_img=mask_img)

    epis = masker.fit_transform(epi_img1)
    assert epis.shape == (1, 3)
    epis = masker.fit_transform([epi_img1, epi_img2])
    assert epis.shape == (2, 3)
