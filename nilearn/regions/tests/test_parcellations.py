"""Test the parcellations tools module."""

import warnings

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _affine_eye
from nilearn.regions.parcellations import (
    Parcellations,
    _check_parameters_transform,
)
from nilearn.surface import SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

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
    return Nifti1Image(data, affine=_affine_eye())


@pytest.fixture
def test_image_2():
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    return Nifti1Image(data, affine=_affine_eye())


def test_error_parcellation_method_none(test_image):
    with pytest.raises(
        ValueError, match="Parcellation method is specified as None. "
    ):
        Parcellations(method=None, verbose=0).fit(test_image)


@pytest.mark.parametrize("method", ["kmens", "avg", "completed"])
def test_errors_raised_in_check_parameters_fit(method, test_image):
    """Test whether an error is raised or not given a false method type."""
    with pytest.raises(
        ValueError,
        match=(f"The method you have selected is not implemented '{method}'"),
    ):
        Parcellations(method=method, verbose=0).fit(test_image)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5, 10, 15])
def test_parcellations_fit_on_single_nifti_image(method, n_parcel, test_image):
    """Test return attributes for each method."""
    parcellator = Parcellations(method=method, n_parcels=n_parcel, verbose=0)
    parcellator.fit(test_image)

    labels_img = parcellator.labels_img_
    # Test that object returns attribute labels_img_
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


def test_parcellations_warnings(img_4d_zeros_eye):
    parcellator = Parcellations(method="kmeans", n_parcels=7, verbose=0)

    with pytest.warns(UserWarning):
        parcellator.fit(img_4d_zeros_eye)


def test_parcellations_no_warnings(img_4d_zeros_eye):
    parcellator = Parcellations(method="kmeans", n_parcels=1, verbose=0)
    with warnings.catch_warnings(record=True) as record:
        parcellator.fit(img_4d_zeros_eye)
    assert all(r.category is not UserWarning for r in record)


def test_parcellations_no_int64_warnings(img_4d_zeros_eye):
    parcellator = Parcellations(method="kmeans", n_parcels=1, verbose=0)
    with warnings.catch_warnings(record=True) as record:
        parcellator.fit(img_4d_zeros_eye)
    for r in record:
        if issubclass(r.category, UserWarning):
            assert "image contains 64-bit ints" not in str(r.message)


@pytest.mark.parametrize("method", METHODS)
def test_parcellations_fit_on_multi_nifti_images(
    method, test_image, affine_eye
):
    fmri_imgs = [test_image] * 3

    parcellator = Parcellations(method=method, n_parcels=5, verbose=0)
    parcellator.fit(fmri_imgs)

    assert parcellator.labels_img_ is not None

    # Smoke test with explicit mask image
    mask_img = np.ones((10, 11, 12))
    mask_img = Nifti1Image(mask_img, affine_eye)
    parcellator = Parcellations(
        method=method, n_parcels=5, mask=mask_img, verbose=0
    )
    parcellator.fit(fmri_imgs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_parcellations_transform_single_nifti_image(
    method, n_parcel, test_image_2
):
    """Test with NiftiLabelsMasker extraction of timeseries data \
       after building a parcellations image.
    """
    parcellator = Parcellations(method=method, n_parcels=n_parcel, verbose=0)
    parcellator.fit(test_image_2)
    # transform to signals
    signals = parcellator.transform(test_image_2)

    # Test if the signals extracted are of same shape as inputs
    # Here, we simply return numpy array for single subject input
    assert signals.shape == (test_image_2.shape[3], n_parcel)
    # Test for single subject but in a list.
    signals = parcellator.transform([test_image_2])
    assert signals.shape == (test_image_2.shape[3], n_parcel)


@pytest.mark.parametrize("verbose", [True, False, -1, 0, 1, 2])
def test_parcellations_transform_verbose(test_image_2, verbose):
    """Test verbose mostly for coverage purpose."""
    parcellator = Parcellations(method="kmeans", n_parcels=5, verbose=verbose)
    parcellator.fit(test_image_2)
    parcellator.transform(test_image_2)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_parcellations_transform_multi_nifti_images(
    method, n_parcel, test_image_2
):
    fmri_imgs = [test_image_2] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel, verbose=0)
    parcellator.fit(fmri_imgs)

    # transform multi images to signals.
    # In return, we have length equal to the number of images
    signals = parcellator.transform(fmri_imgs)

    assert signals[0].shape == (test_image_2.shape[3], n_parcel)
    assert signals[1].shape == (test_image_2.shape[3], n_parcel)
    assert signals[2].shape == (test_image_2.shape[3], n_parcel)
    assert len(signals) == len(fmri_imgs)


def test_check_parameters_transform(test_image_2, rng):
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
    assert single_subject

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
    method, n_parcel, test_image_2, rng
):
    fmri_imgs = [test_image_2] * 3
    confounds = rng.standard_normal(size=(10, 3))
    confounds_list = [confounds] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel, verbose=0)

    parcellator.fit(fmri_imgs)
    signals = parcellator.transform(fmri_imgs, confounds=confounds_list)

    assert isinstance(signals, list)
    # n_parcels=5, length of data=10
    assert signals[0].shape == (10, n_parcel)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_fit_transform(method, n_parcel, test_image_2):
    fmri_imgs = [test_image_2] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel, verbose=0)
    parcellator.fit_transform(fmri_imgs)

    assert parcellator.labels_img_ is not None
    if method not in ["kmeans", "rena", "hierarchical_kmeans"]:
        assert parcellator.connectivity_ is not None
    assert parcellator.masker_ is not None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_fit_transform_with_confounds(method, n_parcel, test_image_2, rng):
    fmri_imgs = [test_image_2] * 3
    confounds = rng.standard_normal(size=(10, 3))
    confounds_list = [confounds] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel, verbose=0)
    signals = parcellator.fit_transform(fmri_imgs, confounds=confounds_list)

    assert isinstance(signals, list)
    assert signals[0].shape == (10, n_parcel)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_inverse_transform_single_nifti_image(method, n_parcel, test_image_2):
    parcellate = Parcellations(method=method, n_parcels=n_parcel, verbose=0)
    parcellate.fit(test_image_2)

    assert parcellate.labels_img_ is not None

    fmri_reduced = parcellate.transform(test_image_2)

    assert isinstance(fmri_reduced, np.ndarray)
    # Shape matching with (scans, regions)
    assert fmri_reduced.shape == (10, n_parcel)

    fmri_compressed = parcellate.inverse_transform(fmri_reduced)

    # A single Nifti image for single subject input
    assert isinstance(fmri_compressed, Nifti1Image)
    # returns shape of fmri_img
    assert fmri_compressed.shape == test_image_2.shape

    # fmri_reduced in a list
    fmri_compressed = parcellate.inverse_transform([fmri_reduced])

    # A single Nifti image for single subject input
    assert isinstance(fmri_compressed, Nifti1Image)
    # returns shape of fmri_img
    assert fmri_compressed.shape == test_image_2.shape


def test_transform_3d_input_images(affine_eye):
    # test list of 3D images
    data = np.ones((10, 11, 12))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    img = Nifti1Image(data, affine=affine_eye)
    imgs = [img] * 3

    parcellate = Parcellations(method="ward", n_parcels=20, verbose=0)
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


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcels", [5, 25])
def test_parcellation_all_methods_with_surface(method, n_parcels, rng):
    """Test if all parcellation methods work on surface."""
    n_samples = 35
    mesh = {
        "left": flat_mesh(10, 8),
        "right": flat_mesh(9, 7),
    }
    data = {
        "left": rng.standard_normal(
            size=(mesh["left"].coordinates.shape[0], n_samples)
        ),
        "right": rng.standard_normal(
            size=(mesh["right"].coordinates.shape[0], n_samples)
        ),
    }
    surf_img = SurfaceImage(mesh=mesh, data=data)
    parcellate = Parcellations(method=method, n_parcels=n_parcels)
    # fit and transform the data
    X_transformed = parcellate.fit_transform(surf_img)
    # inverse transform the transformed data
    X_inverse = parcellate.inverse_transform(X_transformed)

    # make sure the n_features in transformed data were reduced to n_clusters
    assert X_transformed.shape == (n_samples, n_parcels)

    # make sure the inverse transformed data has the same shape as the original
    assert X_inverse.shape == surf_img.shape


@pytest.mark.parametrize("method", METHODS)
def test_parcellation_with_surface_and_confounds(method, rng):
    """Test if parcellation works on surface with confounds."""
    n_samples = 36
    mesh = {
        "left": flat_mesh(10, 8),
        "right": flat_mesh(9, 7),
    }
    data = {
        "left": rng.standard_normal(
            size=(mesh["left"].coordinates.shape[0], n_samples)
        ),
        "right": rng.standard_normal(
            size=(mesh["right"].coordinates.shape[0], n_samples)
        ),
    }
    surf_img = SurfaceImage(mesh=mesh, data=data)
    confounds = rng.standard_normal(size=(n_samples, 3))
    parcellate = Parcellations(method=method, n_parcels=5)
    X_transformed = parcellate.fit_transform(surf_img, confounds=[confounds])

    assert X_transformed.shape == (n_samples, 5)


@pytest.mark.parametrize("method", METHODS)
def test_parcellation_with_multi_surface(method, rng):
    """Test if parcellation works with surface data from multiple
    'subjects'.
    """
    n_samples = 36
    mesh = {
        "left": flat_mesh(10, 8),
        "right": flat_mesh(9, 7),
    }
    data = {
        "left": rng.standard_normal(
            size=(mesh["left"].coordinates.shape[0], n_samples)
        ),
        "right": rng.standard_normal(
            size=(mesh["right"].coordinates.shape[0], n_samples)
        ),
    }
    surf_img = SurfaceImage(mesh=mesh, data=data)
    surf_imgs = [surf_img] * 3
    parcellate = Parcellations(method=method, n_parcels=5)
    X_transformed = parcellate.fit_transform(surf_imgs)

    assert X_transformed[0].shape == (n_samples, 5)
    assert len(X_transformed) == 3


@pytest.mark.parametrize("method", METHODS)
def test_parcellation_with_surface_mask(method, rng):
    """Test if parcellation works with surface data and a mask."""
    n_samples = 36
    mesh = {
        "left": flat_mesh(10, 8),
        "right": flat_mesh(9, 7),
    }
    data = {
        "left": rng.standard_normal(
            size=(mesh["left"].coordinates.shape[0], n_samples)
        ),
        "right": rng.standard_normal(
            size=(mesh["right"].coordinates.shape[0], n_samples)
        ),
    }
    surf_img = SurfaceImage(mesh=mesh, data=data)
    mask_data = {
        "left": np.ones(mesh["left"].coordinates.shape[0]).astype(bool),
        "right": np.ones(mesh["right"].coordinates.shape[0]).astype(bool),
    }
    surf_mask = SurfaceImage(mesh=mesh, data=mask_data)
    parcellate = Parcellations(method=method, n_parcels=5, mask=surf_mask)
    X_transformed = parcellate.fit_transform(surf_img)

    assert X_transformed.shape == (n_samples, 5)
