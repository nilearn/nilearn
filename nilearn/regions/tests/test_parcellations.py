"""Test the parcellations tools module"""
import nibabel
import numpy as np
import pandas as pd
import pytest
from nilearn.regions.parcellations import (
    Parcellations,
    _check_parameters_transform,
)

METHODS = [
    "kmeans",
    "ward",
    "complete",
    "average",
    "rena",
    "hierarchical_kmeans",
]


@pytest.fixture
def test_image():
    data = np.zeros((10, 11, 12, 5))
    data[9, 10, 2] = 1
    data[4, 9, 3] = 2
    return nibabel.Nifti1Image(data, affine=np.eye(4))


@pytest.fixture
def test_image_2():
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    return nibabel.Nifti1Image(data, affine=np.eye(4))


@pytest.fixture
def test_empty_image():
    data = np.zeros((10, 11, 12, 5))
    return nibabel.Nifti1Image(data, affine=np.eye(4))


def test_error_parcellation_method_none(test_image):
    with pytest.raises(
        ValueError, match="Parcellation method is specified as None. "
    ):
        Parcellations(method=None).fit(test_image)


@pytest.mark.parametrize("method", ["kmens", "avg", "complte"])
def test_errors_raised_in_check_parameters_fit(method, test_image):
    """Test whether an error is raised or not given
    a false method type.
    """
    with pytest.raises(
        ValueError,
        match=(f"The method you have selected is not implemented '{method}'"),
    ):
        Parcellations(method=method).fit(test_image)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5, 10, 15])
def test_parcellations_fit_on_single_nifti_image(method, n_parcel, test_image):
    """Test return attributes for each method."""
    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit(test_image)
    # Test that object returns attribute labels_img_
    labels_img = parcellator.labels_img_
    assert labels_img is not None
    # After inverse_transform, shape must match with
    # original input data
    assert labels_img.shape == test_image.shape[:3]
    # Test object returns attribute masker_
    assert parcellator.masker_ is not None
    assert parcellator.mask_img_ is not None
    if method not in ["kmeans", "rena", "hierarchical_kmeans"]:
        # Test that object returns attribute connectivity_
        # only for AgglomerativeClustering methods
        assert parcellator.connectivity_ is not None


def test_parcellations_warnings(test_empty_image):
    parcellator = Parcellations(method="kmeans", n_parcels=7)
    with pytest.warns(UserWarning):
        parcellator.fit(test_empty_image)


def test_parcellations_no_warnings(test_empty_image):
    parcellator = Parcellations(method="kmeans", n_parcels=1)
    with pytest.warns(None) as record:
        parcellator.fit(test_empty_image)
    assert all(r.category is not UserWarning for r in record.list)


@pytest.mark.parametrize("method", METHODS)
def test_parcellations_fit_on_multi_nifti_images(method, test_image):
    fmri_imgs = [test_image] * 3
    parcellator = Parcellations(method=method, n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert parcellator.labels_img_ is not None

    parcellator = Parcellations(method="rena", n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert parcellator.labels_img_ is not None

    parcellator = Parcellations(method="hierarchical_kmeans", n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert parcellator.labels_img_ is not None

    # Smoke test with explicit mask image
    mask_img = np.ones((10, 11, 12))
    mask_img = nibabel.Nifti1Image(mask_img, np.eye(4))
    parcellator = Parcellations(method=method, n_parcels=5, mask=mask_img)
    parcellator.fit(fmri_imgs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_parcellations_transform_single_nifti_image(
    method, n_parcel, test_image_2
):
    """Test with NiftiLabelsMasker extraction of timeseries data
    after building a parcellations image.

    """
    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit(test_image_2)
    # transform to signals
    signals = parcellator.transform(test_image_2)
    # Test if the signals extracted are of same shape as inputs
    # Here, we simply return numpy array for single subject input
    assert signals.shape == (test_image_2.shape[3], n_parcel)
    # Test for single subject but in a list.
    signals = parcellator.transform([test_image_2])
    assert signals.shape == (test_image_2.shape[3], n_parcel)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_parcellations_transform_multi_nifti_images(
    method, n_parcel, test_image_2
):
    fmri_imgs = [test_image_2] * 3
    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit(fmri_imgs)
    # transform multi images to signals.
    # In return, we have length equal to the number of images
    signals = parcellator.transform(fmri_imgs)
    assert signals[0].shape == (test_image_2.shape[3], n_parcel)
    assert signals[1].shape == (test_image_2.shape[3], n_parcel)
    assert signals[2].shape == (test_image_2.shape[3], n_parcel)
    assert len(signals) == len(fmri_imgs)


def test_check_parameters_transform(test_image_2):
    rng = np.random.RandomState(42)
    # single confound
    confounds = rng.standard_normal(size=(10, 3))
    # Tests to check whether imgs, confounds returned are
    # list or not. Pre-check in parameters to work for list
    # of multi images and multi confounds
    imgs, confounds, single_subject = _check_parameters_transform(
        test_image_2, confounds
    )
    assert isinstance(imgs, (list, tuple))
    assert isinstance(confounds, (list, tuple))
    assert single_subject, True
    # confounds as pandas DataFrame
    imgs, confounds, single_subject = _check_parameters_transform(
        test_image_2, pd.DataFrame(np.array(confounds)[0])
    )
    assert isinstance(confounds, (list, tuple))
    # multi images
    fmri_imgs = [test_image_2] * 3
    confounds_list = [confounds] * 3
    imgs, confounds, _ = _check_parameters_transform(fmri_imgs, confounds_list)
    assert imgs == fmri_imgs
    assert confounds_list == confounds
    # Test the error when length of images and confounds are not same
    msg = (
        "Number of confounds given does not match with the "
        "given number of images"
    )

    not_match_confounds_list = [confounds] * 2
    with pytest.raises(ValueError, match=msg):
        _check_parameters_transform(fmri_imgs, not_match_confounds_list)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_parcellations_transform_with_multi_confounds_multi_images(
    method, n_parcel, test_image_2
):
    rng = np.random.RandomState(42)
    fmri_imgs = [test_image_2] * 3
    confounds = rng.standard_normal(size=(10, 3))
    confounds_list = [confounds] * 3
    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit(fmri_imgs)
    signals = parcellator.transform(fmri_imgs, confounds=confounds_list)
    assert isinstance(signals, list)
    # n_parcels=5, length of data=10
    assert signals[0].shape == (10, n_parcel)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_fit_transform(method, n_parcel, test_image_2):
    rng = np.random.RandomState(42)
    fmri_imgs = [test_image_2] * 3
    confounds = rng.standard_normal(size=(10, 3))
    confounds_list = [confounds] * 3
    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    signals = parcellator.fit_transform(fmri_imgs)
    assert parcellator.labels_img_ is not None
    if method not in ["kmeans", "rena", "hierarchical_kmeans"]:
        assert parcellator.connectivity_ is not None
    assert parcellator.masker_ is not None
    # fit_transform with confounds
    signals = parcellator.fit_transform(fmri_imgs, confounds=confounds_list)
    assert isinstance(signals, list)
    assert signals[0].shape == (10, n_parcel)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_inverse_transform_single_nifti_image(method, n_parcel, test_image_2):
    parcellate = Parcellations(method=method, n_parcels=n_parcel)
    parcellate.fit(test_image_2)
    assert parcellate.labels_img_ is not None
    fmri_reduced = parcellate.transform(test_image_2)
    assert isinstance(fmri_reduced, np.ndarray)
    # Shape matching with (scans, regions)
    assert fmri_reduced.shape == (10, n_parcel)
    fmri_compressed = parcellate.inverse_transform(fmri_reduced)
    # A single Nifti image for single subject input
    assert isinstance(fmri_compressed, nibabel.Nifti1Image)
    # returns shape of fmri_img
    assert fmri_compressed.shape == test_image_2.shape
    # fmri_reduced in a list
    fmri_compressed = parcellate.inverse_transform([fmri_reduced])
    # A single Nifti image for single subject input
    assert isinstance(fmri_compressed, nibabel.Nifti1Image)
    # returns shape of fmri_img
    assert fmri_compressed.shape == test_image_2.shape


def test_transform_3d_input_images():
    # test list of 3D images
    data = np.ones((10, 11, 12))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    img = nibabel.Nifti1Image(data, affine=np.eye(4))
    imgs = [img] * 3
    parcellate = Parcellations(method="ward", n_parcels=20)
    X = parcellate.fit_transform(imgs)
    assert isinstance(X, list)
    # (number of samples, number of features)
    assert np.concatenate(X).shape == (3, 20)
    # inverse transform
    imgs_ = parcellate.inverse_transform(X)
    assert isinstance(imgs_, list)
    # test single 3D image
    X = parcellate.fit_transform(imgs[0])
    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 20)
