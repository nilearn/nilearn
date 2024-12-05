"""Test the nifti_region module.

Functions in this file only test features added by the NiftiLabelsMasker class,
not the underlying functions (clean(), img_to_signals_labels(), etc.). See
test_masking.py and test_signal.py for details.
"""

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal

from nilearn._utils.data_gen import (
    generate_labeled_regions,
    generate_random_img,
)
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.testing import write_imgs_to_path
from nilearn.image import get_data
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker


@pytest.fixture
def n_regions():
    return 9


@pytest.fixture
def length():
    return 93


def test_nifti_labels_masker(affine_eye, shape_3d_default, n_regions, length):
    """Check working of shape/affine checks."""
    shape1 = (*shape_3d_default, length)

    fmri_img, mask11_img = generate_random_img(
        shape1,
        affine=affine_eye,
    )

    labels_img = generate_labeled_regions(
        shape1[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )

    # No exception raised here
    masker = NiftiLabelsMasker(labels_img, resampling_target=None)
    signals = masker.fit().transform(fmri_img)

    assert signals.shape == (length, n_regions)

    # No exception should be raised either
    masker = NiftiLabelsMasker(labels_img, resampling_target=None)

    # Check attributes defined at fit
    assert not hasattr(masker, "mask_img_")
    assert not hasattr(masker, "labels_img_")
    assert not hasattr(masker, "n_elements_")

    masker.fit()

    # Check attributes defined at fit
    assert hasattr(masker, "mask_img_")
    assert hasattr(masker, "labels_img_")
    assert hasattr(masker, "n_elements_")
    assert masker.n_elements_ == n_regions

    masker.inverse_transform(signals)

    # now with several mask_img
    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask11_img, resampling_target=None
    )
    signals = masker.fit().transform(fmri_img)

    assert signals.shape == (length, n_regions)

    shape2 = (12, 10, 14, length)
    _, mask21_img = generate_random_img(
        shape2,
        affine=affine_eye,
    )

    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask21_img, resampling_target=None
    )

    # Transform, with smoothing (smoke test)
    masker = NiftiLabelsMasker(
        labels_img, smoothing_fwhm=3, resampling_target=None
    )
    signals = masker.fit().transform(fmri_img)

    assert signals.shape == (length, n_regions)

    # Call inverse transform (smoke test)
    fmri_img_r = masker.inverse_transform(signals)

    assert fmri_img_r.shape == fmri_img.shape
    assert_almost_equal(fmri_img_r.affine, fmri_img.affine)


def test_nifti_labels_masker_errors(
    affine_eye, shape_3d_default, n_regions, length
):
    """Check working of shape/affine checks."""
    shape1 = (*shape_3d_default, length)

    shape2 = (12, 10, 14, length)
    affine2 = np.diag((1, 2, 3, 1))

    fmri11_img, _ = generate_random_img(
        shape1,
        affine=affine_eye,
    )
    fmri12_img, mask12_img = generate_random_img(
        shape1,
        affine=affine2,
    )
    fmri21_img, mask21_img = generate_random_img(
        shape2,
        affine=affine_eye,
    )

    labels11_img = generate_labeled_regions(
        shape1[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )

    mask_img_4d = Nifti1Image(
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

    # Test all kinds of mismatch between shapes and between affines
    masker11.fit()
    with pytest.raises(
        ValueError, match="Images have different affine matrices."
    ):
        masker11.transform(fmri12_img)
    with pytest.raises(ValueError, match="Images have incompatible shapes."):
        masker11.transform(fmri21_img)

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask12_img, resampling_target=None
    )
    with pytest.raises(
        ValueError, match="Regions and mask do not have the same affine."
    ):
        masker11.fit()

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(
        ValueError, match="Regions and mask do not have the same shape"
    ):
        masker11.fit()

    masker11 = NiftiLabelsMasker(
        labels11_img, smoothing_fwhm=3, resampling_target=None
    )
    signals11 = masker11.fit_transform(fmri11_img)

    with pytest.raises(ValueError, match="has not been fitted. "):
        NiftiLabelsMasker(labels11_img).inverse_transform(signals11)


def test_nifti_labels_masker_io_shapes(
    rng, affine_eye, shape_3d_default, n_regions
):
    """Ensure that NiftiLabelsMasker handles 1D/2D/3D/4D data appropriately.

    transform(4D image) --> 2D output, no warning
    transform(3D image) --> 2D output, DeprecationWarning
    inverse_transform(2D array) --> 4D image, no warning
    inverse_transform(1D array) --> 3D image, no warning
    inverse_transform(2D array with wrong shape) --> IndexError
    """
    length = 5
    shape_4d = (*shape_3d_default, length)
    data_1d = rng.random(n_regions)
    data_2d = rng.random((length, n_regions))

    img_4d, mask_img = generate_random_img(
        shape_4d,
        affine=affine_eye,
    )
    img_3d, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    labels_img = generate_labeled_regions(
        shape_3d_default,
        affine=affine_eye,
        n_regions=n_regions,
    )
    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)
    masker.fit()

    # DeprecationWarning *should* be raised for 3D inputs
    with pytest.deprecated_call(match="Starting in version 0.12"):
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

        assert test_data.shape == (length, n_regions)

    # DeprecationWarning should *not* be raised for 1D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_1d)

        assert test_img.shape == shape_3d_default

    # DeprecationWarning should *not* be raised for 2D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_2d)

        assert test_img.shape == shape_4d

    with pytest.raises(
        IndexError, match="index 6 is out of bounds for axis 1 with size 5"
    ):
        masker.inverse_transform(data_2d.T)


@pytest.mark.parametrize("nans_in", ["mask", "labels"])
def test_nifti_labels_masker_with_nans_and_infs(
    affine_eye, shape_3d_default, nans_in, n_regions, length
):
    """Deal with NaNs and infs in label image or mask.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    fmri_img, mask_img = generate_random_img(
        (*shape_3d_default, length),
        affine=affine_eye,
    )
    labels_img = generate_labeled_regions(
        shape_3d_default,
        affine=affine_eye,
        n_regions=n_regions,
    )

    # Introduce nans with data type float
    # See issue: https://github.com/nilearn/nilearn/issues/2580
    def add_nans_and_infs(img, affine):
        data = get_data(img).astype(np.float32)
        data[:, :, 7] = np.nan
        data[:, :, 4] = np.inf
        return Nifti1Image(data, affine)

    if nans_in == "labels":
        labels_img = add_nans_and_infs(labels_img, affine_eye)
    elif nans_in == "mask":
        mask_img = add_nans_and_infs(mask_img, affine_eye)

    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_labels_masker_with_nans_and_infs_in_data(
    affine_eye, shape_3d_default, n_regions, length
):
    """Apply a NiftiLabelsMasker to 4D data containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    fmri_img, mask_img = generate_random_img(
        (*shape_3d_default, length),
        affine=affine_eye,
    )
    labels_img = generate_labeled_regions(
        shape_3d_default,
        affine=affine_eye,
        n_regions=n_regions,
    )
    # Introduce nans with data type float
    # See issues:
    # - https://github.com/nilearn/nilearn/issues/2580 (why floats)
    # - https://github.com/nilearn/nilearn/issues/2711 (why test)
    fmri_data = get_data(fmri_img).astype(np.float32)
    fmri_data[:, :, 7, :] = np.nan
    fmri_data[:, :, 4, 0] = np.inf
    fmri_img = Nifti1Image(fmri_data, affine_eye)

    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


@pytest.mark.parametrize(
    "strategy, function",
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
def test_nifti_labels_masker_reduction_strategies(
    affine_eye, strategy, function
):
    """Tests NiftiLabelsMasker strategies.

    1. whether the usage of different reduction strategies work.
    2. whether unrecognized strategies raise a ValueError
    3. whether the default option is backwards compatible (calls "mean")
    """
    test_values = [-2.0, -1.0, 0.0, 1.0, 2]

    img_data = np.array([[test_values, test_values]])

    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    img = Nifti1Image(img_data, affine_eye)
    labels = Nifti1Image(labels_data, affine_eye)

    # What NiftiLabelsMasker should return for each reduction strategy?
    expected_result = function(test_values)

    masker = NiftiLabelsMasker(labels, strategy=strategy)
    # Here passing [img] within a list because it's a 3D object.
    result = masker.fit_transform([img]).squeeze()

    assert result == expected_result

    default_masker = NiftiLabelsMasker(labels)

    assert default_masker.strategy == "mean"


def test_nifti_labels_masker_reduction_strategies_error(affine_eye):
    """Tests NiftiLabelsMasker invalid strategy."""
    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    labels = Nifti1Image(labels_data, affine_eye)

    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        NiftiLabelsMasker(labels, strategy="TESTRAISE")


def test_nifti_labels_masker_resampling_errors(
    affine_eye, shape_3d_default, n_regions
):
    """Test errors of resampling in NiftiLabelsMasker."""
    labels_img = generate_labeled_regions(
        shape=shape_3d_default,
        n_regions=n_regions,
        affine=affine_eye,
    )

    with pytest.raises(
        ValueError,
        match="invalid value for 'resampling_target' parameter: mask",
    ):
        NiftiLabelsMasker(labels_img, resampling_target="mask")

    with pytest.raises(
        ValueError,
        match="invalid value for 'resampling_target' parameter: invalid",
    ):
        NiftiLabelsMasker(
            labels_img,
            resampling_target="invalid",
        )


def test_nifti_labels_masker_resampling_to_data(
    tmp_path, affine_eye, n_regions, length
):
    """Test resampling to data in NiftiLabelsMasker."""
    # mask
    shape2 = (8, 9, 10, length)
    # maps
    shape3 = (16, 18, 20)

    _, mask_img = generate_random_img(
        shape2,
        affine=affine_eye,
    )

    labels_img = generate_labeled_regions(shape3, n_regions, affine=affine_eye)

    # Test with data and atlas of different shape:
    # the atlas should be resampled to the data
    shape22 = (5, 5, 6, length)
    affine2 = 2 * affine_eye
    affine2[-1, -1] = 1

    fmri_img, _ = generate_random_img(
        shape22,
        affine=affine2,
    )

    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask_img, resampling_target="data"
    )
    masker.fit_transform(fmri_img)

    assert_array_equal(masker._resampled_labels_img_.affine, affine2)

    # Test with filenames
    filename = write_imgs_to_path(fmri_img, file_path=tmp_path)
    masker = NiftiLabelsMasker(labels_img, resampling_target="data")
    masker.fit_transform(filename)


@pytest.mark.parametrize("resampling_target", ["data", "labels"])
def test_nifti_labels_masker_resampling(
    affine_eye, shape_3d_default, resampling_target, n_regions, length
):
    """Test to return resampled labels having number of labels \
       equal with transformed shape of 2nd dimension.

    See https://github.com/nilearn/nilearn/issues/1673
    """
    shape = (*shape_3d_default, length)
    affine = 2 * affine_eye

    fmri_img, _ = generate_random_img(shape, affine=affine)
    labels_img = generate_labeled_regions(
        shape_3d_default, affine=affine_eye, n_regions=n_regions
    )

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

    assert_array_equal(get_data(compressed_img), get_data(compressed_img2))


def test_nifti_labels_masker_resampling_to_labels(
    affine_eye, shape_3d_default, n_regions, length
):
    """Test resampling to labels in NiftiLabelsMasker."""
    # fmri
    shape1 = (*shape_3d_default, length)
    # mask
    shape2 = (16, 17, 18, length)
    # labels
    shape3 = (13, 14, 15)

    # With data of the same affine
    fmri_img, _ = generate_random_img(
        shape1,
        affine=affine_eye,
    )
    _, mask_img = generate_random_img(
        shape2,
        affine=affine_eye,
    )

    labels_img = generate_labeled_regions(
        shape3,
        n_regions,
        affine=affine_eye,
    )

    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask_img, resampling_target="labels"
    )

    masker.fit()

    assert_almost_equal(masker.labels_img_.affine, labels_img.affine)
    assert masker.labels_img_.shape == labels_img.shape
    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    transformed = masker.transform(fmri_img)

    assert transformed.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(transformed)

    assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
    assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_nifti_labels_masker_resampling_to_clipped_labels(
    affine_eye, shape_3d_default, n_regions, length
):
    """Test with clipped labels.

    Mask does not contain all labels.

    Shapes do matter in that case,
    because there is some resampling taking place.
    """
    # fmri
    shape1 = (*shape_3d_default, length)
    # mask
    shape2 = (8, 9, 10, length)
    # maps
    shape3 = (16, 18, 20)

    fmri11_img, _ = generate_random_img(
        shape1,
        affine=affine_eye,
    )
    _, mask22_img = generate_random_img(
        shape2,
        affine=affine_eye,
    )

    labels33_img = generate_labeled_regions(
        shape3, n_regions, affine=affine_eye
    )

    masker = NiftiLabelsMasker(
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

    assert transformed.shape == (length, n_regions)
    # Some regions have been clipped. Resulting signal must be zero
    assert (transformed.var(axis=0) == 0).sum() < n_regions

    fmri11_img_r = masker.inverse_transform(transformed)

    assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
    assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_nifti_labels_masker_resampling_to_none(
    affine_eye, n_regions, length, shape_3d_default
):
    """Test resampling to None in NiftiLabelsMasker.

    All inputs must have same affine to avoid errors.
    """
    fmri_img, mask_img = generate_random_img(
        shape=(*shape_3d_default, length),
        affine=affine_eye,
    )
    labels_img = generate_labeled_regions(
        shape_3d_default, n_regions, affine=affine_eye
    )

    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask_img, resampling_target=None
    )
    masker.fit_transform(fmri_img)

    fmri_img, _ = generate_random_img(
        (*shape_3d_default, length),
        affine=affine_eye * 2,
    )
    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask_img, resampling_target=None
    )
    with pytest.raises(
        ValueError, match="Images have different affine matrices."
    ):
        masker.fit_transform(fmri_img)


def test_standardization(rng, affine_eye, shape_3d_default, n_regions):
    n_samples = 400

    signals = rng.standard_normal(size=(np.prod(shape_3d_default), n_samples))
    means = (
        rng.standard_normal(size=(np.prod(shape_3d_default), 1)) * 50 + 1000
    )
    signals += means
    img = Nifti1Image(
        signals.reshape((*shape_3d_default, n_samples)),
        affine_eye,
    )

    labels = generate_labeled_regions(shape_3d_default, n_regions)

    # Unstandarized
    masker = NiftiLabelsMasker(labels, standardize=False)
    unstandarized_label_signals = masker.fit_transform(img)

    # z-score
    masker = NiftiLabelsMasker(labels, standardize="zscore_sample")
    trans_signals = masker.fit_transform(img)

    assert_almost_equal(trans_signals.mean(0), 0)
    assert_almost_equal(trans_signals.std(0), 1, decimal=3)

    # psc
    masker = NiftiLabelsMasker(labels, standardize="psc")
    trans_signals = masker.fit_transform(img)

    assert_almost_equal(trans_signals.mean(0), 0)
    assert_almost_equal(
        trans_signals,
        (
            unstandarized_label_signals
            / unstandarized_label_signals.mean(0)
            * 100
            - 100
        ),
    )


def test_nifti_labels_masker_with_mask(
    shape_3d_default, affine_eye, n_regions, length
):
    """Test NiftiLabelsMasker with a separate mask_img parameter."""
    shape = (*shape_3d_default, length)
    fmri_img, mask_img = generate_random_img(shape, affine=affine_eye)
    labels_img = generate_labeled_regions(
        shape[:3],
        affine=affine_eye,
        n_regions=n_regions,
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
    # masked_labels is a 4D image with shape (10,10,10,1)
    masked_labels_data = get_data(masked_labels)[:, :, :, 0]
    assert np.allclose(get_data(masker.region_atlas_), masked_labels_data)


@pytest.mark.parametrize(
    "background",
    [
        None,
        "background",
        "Background",
    ],
)
def test_warning_n_labels_not_equal_n_regions(
    shape_3d_default, affine_eye, background, n_regions
):
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )
    region_names = generate_labels(n_regions + 2, background=background)
    with pytest.warns(
        UserWarning, match="Mismatch between the number of provided labels"
    ):
        NiftiLabelsMasker(
            labels_img,
            labels=region_names,
        )


def test_sanitize_labels_warnings(shape_3d_default, affine_eye, n_regions):
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )
    with pytest.warns(UserWarning, match="'labels' must be a list."):
        NiftiLabelsMasker(
            labels_img,
            labels="foo",
        )
    with pytest.warns(
        UserWarning, match="All elements of 'labels' must be a string"
    ):
        NiftiLabelsMasker(
            labels_img,
            labels=[1, 2, 3],
        )


@pytest.mark.parametrize(
    "background",
    [
        None,
        "background",
        "Background",
    ],  # In case the list of labels includes one for background
)
@pytest.mark.parametrize(
    "dtype",
    ["int32", "float32"],  # In case regions are labeled with floats
)
@pytest.mark.parametrize(
    "affine_data",
    [
        None,  # no resampling
        np.diag(
            (4, 4, 4, 4)  # with resampling
        ),  # region_names_ matches signals after resampling drops labels
    ],
)
def test_region_names(
    shape_3d_default, affine_eye, background, affine_data, dtype, n_regions
):
    """Test region_names_ attribute in NiftiLabelsMasker."""
    resampling = True
    if affine_data is None:
        resampling = False
        affine_data = affine_eye
    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_data)
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=n_regions,
        dtype=dtype,
    )

    region_names = generate_labels(n_regions, background=background)

    masker = NiftiLabelsMasker(
        labels_img,
        labels=region_names,
        resampling_target="data",
    )
    signals = masker.fit().transform(fmri_img)

    check_region_names_after_fit(
        masker, signals, region_names, background, resampling
    )


def check_region_names_after_fit(
    masker, signals, region_names, background, resampling=False
):
    """Perform several checks on the expected attributes of the masker.

    - region_names_ does not include background
      should have same length as signals
    - region_ids_ does include background
    - region_names_ should be the same as the region names
      passed to the masker minus that for "background"
    """
    assert len(masker.region_names_) == signals.shape[1]
    assert len(list(masker.region_ids_.items())) == signals.shape[1] + 1

    # resampling may drop some labels so we do not check the region names
    # in this case
    if not resampling:
        region_names_after_fit = [
            masker.region_names_[i] for i in masker.region_names_
        ]
        region_names_after_fit.sort()
        region_names.sort()
        if background:
            region_names.pop(region_names.index(background))
        assert region_names_after_fit == region_names


@pytest.mark.parametrize(
    "background",
    [
        None,
        "background",
        "Background",
    ],
)
@pytest.mark.parametrize(
    "affine_data",
    [
        None,  # no resampling
        np.diag(
            (4, 4, 4, 4)  # with resampling
        ),  # region_names_ matches signals after resampling drops labels
    ],
)
@pytest.mark.parametrize(
    "masking",
    [
        False,  # no masking
        True,  # with masking
    ],
)
@pytest.mark.parametrize(
    "keep_masked_labels",
    [
        False,
        True,
    ],
)
def test_region_names_ids_match_after_fit(
    shape_3d_default,
    affine_eye,
    background,
    affine_data,
    n_regions,
    masking,
    keep_masked_labels,
):
    """Test that the same region names and ids correspond after fit."""
    if affine_data is None:
        # no resampling
        affine_data = affine_eye
    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_data)
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )

    region_names = generate_labels(n_regions, background=background)
    region_ids = list(np.unique(get_data(labels_img)))

    if masking:
        # create a mask_img with 3 regions
        labels_data = get_data(labels_img)
        mask_data = (
            (labels_data == 1) + (labels_data == 2) + (labels_data == 5)
        )
        mask_img = Nifti1Image(mask_data.astype(np.int8), labels_img.affine)
    else:
        mask_img = None

    masker = NiftiLabelsMasker(
        labels_img,
        labels=region_names,
        resampling_target="data",
        mask_img=mask_img,
        keep_masked_labels=keep_masked_labels,
    )

    _ = masker.fit().transform(fmri_img)

    check_region_names_ids_match_after_fit(
        masker, region_names, region_ids, background
    )


def check_region_names_ids_match_after_fit(
    masker, region_names, region_ids, background
):
    """Check the region names and ids correspondence.

    Check that the same region names and ids correspond to each other
    after fit by comparing with before fit.
    """
    # region_ids includes background, so we make
    # sure that the region_names also include it
    if not background:
        region_names.insert(0, "background")
    # if they don't have the same length, we can't compare them
    if len(region_names) == len(region_ids):
        region_id_names = {
            region_id: region_names[i]
            for i, region_id in enumerate(region_ids)
        }
        for key, region_name in masker.region_names_.items():
            assert region_id_names[masker.region_ids_[key]] == region_name


def generate_labels(n_regions, background=True):
    labels = []
    if background:
        labels.append(background)
    labels.extend([f"region_{i + 1!s}" for i in range(n_regions)])
    return labels


@pytest.mark.parametrize("background", [None, "background", "Background"])
def test_region_names_with_non_sequential_labels(
    shape_3d_default, affine_eye, background
):
    """Test for atlases with region id that are not consecutive.

    See the AAL atlas for an example of this.
    """
    labels = [2001, 2002, 2101, 2102, 9170]
    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=len(labels),
        labels=labels,
    )

    region_names = generate_labels(len(labels), background=background)

    masker = NiftiLabelsMasker(
        labels_img,
        labels=region_names,
        resampling_target=None,
    )
    signals = masker.fit().transform(fmri_img)

    check_region_names_after_fit(masker, signals, region_names, background)


@pytest.mark.parametrize("background", [None, "background", "Background"])
def test_more_labels_than_actual_region_in_atlas(
    shape_3d_default, affine_eye, background, n_regions
):
    """Test region_names_ attribute in NiftiLabelsMasker.

    See fetch_atlas_destrieux_2009 for example.
    Some labels have no associated voxels.
    """
    n_regions_in_labels = n_regions + 5

    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )

    region_names = generate_labels(n_regions_in_labels, background=background)

    masker = NiftiLabelsMasker(
        labels_img,
        labels=region_names,
        resampling_target="data",
    )

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    with pytest.warns(
        UserWarning, match="Mismatch between the number of provided labels"
    ):
        masker.fit().transform(fmri_img)


def test_3d_images(affine_eye, shape_3d_default, n_regions):
    """Test that the NiftiLabelsMasker works with 3D images."""
    mask_img = Nifti1Image(
        np.ones(shape_3d_default, dtype=np.int8),
        affine=affine_eye,
    )
    labels_img = generate_labeled_regions(shape_3d_default, n_regions)

    masker = NiftiLabelsMasker(labels_img, mask_img=mask_img)

    epi_img1 = Nifti1Image(np.ones(shape_3d_default), affine=affine_eye)
    epis = masker.fit_transform(epi_img1)

    assert epis.shape == (1, n_regions)

    epi_img2 = Nifti1Image(np.ones(shape_3d_default), affine=affine_eye)
    epis = masker.fit_transform([epi_img1, epi_img2])

    assert epis.shape == (2, n_regions)


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_nifti_labels_masker_reporting_mpl_warning(
    shape_3d_default, n_regions, length, affine_eye
):
    """Raise warning after exception if matplotlib is not installed."""
    shape1 = (*shape_3d_default, length)
    labels_img = generate_labeled_regions(
        shape1[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )

    with warnings.catch_warnings(record=True) as warning_list:
        result = NiftiLabelsMasker(labels_img).generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
    assert result == [None]
