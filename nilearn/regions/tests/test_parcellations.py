"""Test the parcellations tools module."""

import warnings

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
)
from nilearn._utils.helpers import is_windows_platform
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.conftest import _affine_eye
from nilearn.maskers import (
    MultiNiftiMasker,
    MultiSurfaceMasker,
    NiftiMasker,
    SurfaceMasker,
)
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

ESTIMATORS_TO_CHECK = [Parcellations(method=x, n_parcels=5) for x in METHODS]

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
    from sklearn.utils.estimator_checks import parametrize_with_checks

    from nilearn._utils.estimator_checks import (
        return_expected_failed_checks,
    )

    @pytest.mark.slow
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
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.fixture
def image_1():
    data = np.zeros((10, 11, 12, 5))
    data[9, 10, 2] = 1
    data[4, 9, 3] = 2
    return Nifti1Image(data, affine=_affine_eye())


@pytest.fixture
def image_2():
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    return Nifti1Image(data, affine=_affine_eye())


@pytest.mark.slow
def test_error_parcellation_method_none(image_1):
    with pytest.raises(
        ValueError, match=r"Parcellation method is specified as None. "
    ):
        Parcellations(method=None).fit(image_1)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["kmens", "avg", "completed"])
def test_errors_raised_in_check_parameters_fit(method, image_1):
    """Test whether an error is raised or not given a false method type."""
    with pytest.raises(
        ValueError,
        match=("'method' must be one of"),
    ):
        Parcellations(method=method).fit(image_1)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5, 10, 15])
def test_fit_on_single_nifti_image(method, n_parcel, image_1):
    """Test return attributes for each method."""
    parcellator = Parcellations(method=method, n_parcels=n_parcel)

    with warnings.catch_warnings(record=True) as warning_list:
        parcellator.fit(image_1)

    # The input image used for testing may not give the requested n_parcels.
    # So we check that the proper warning is thrown
    # and that n_elements is always 5
    if not any(
        (
            "The number of generated labels does not "
            "match the requested number of parcels."
        )
        in str(x)
        for x in warning_list
    ):
        assert parcellator.n_elements_ == n_parcel

    labels_img = parcellator.labels_img_

    # Test that object returns attribute labels_img_
    assert labels_img is not None

    # After inverse_transform, shape must match with
    # original input data
    assert labels_img.shape == image_1.shape[:3]
    # Test object returns attribute masker_
    assert parcellator.masker_ is not None
    assert parcellator.mask_img_ is not None
    if method not in ["kmeans", "rena", "hierarchical_kmeans"]:
        # Test that object returns attribute connectivity_
        # only for AgglomerativeClustering methods
        assert parcellator.connectivity_ is not None


@pytest.mark.slow
def test_warnings(img_4d_zeros_eye):
    parcellator = Parcellations(method="kmeans", n_parcels=7)

    with pytest.warns(UserWarning):
        parcellator.fit(img_4d_zeros_eye)


@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
def test_no_warnings(img_4d_zeros_eye):
    parcellator = Parcellations(method="kmeans", n_parcels=1)
    with warnings.catch_warnings(record=True) as record:
        parcellator.fit(img_4d_zeros_eye)
    assert all(r.category is not UserWarning for r in record)


@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
def test_no_int64_warnings(img_4d_zeros_eye):
    parcellator = Parcellations(method="kmeans", n_parcels=1)
    with warnings.catch_warnings(record=True) as record:
        parcellator.fit(img_4d_zeros_eye)
    for r in record:
        if issubclass(r.category, UserWarning):
            assert "image contains 64-bit ints" not in str(r.message)


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_fit_on_multi_nifti_images(method, image_1, affine_eye):
    fmri_imgs = [image_1] * 3

    parcellator = Parcellations(method=method, n_parcels=5)
    parcellator.fit(fmri_imgs)

    assert parcellator.labels_img_ is not None

    # Smoke test with explicit mask image
    mask_img = np.ones((10, 11, 12))
    mask_img = Nifti1Image(mask_img, affine_eye)
    parcellator = Parcellations(method=method, n_parcels=5, mask=mask_img)
    parcellator.fit(fmri_imgs)


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_transform_single_nifti_image(method, n_parcel, image_2):
    """Test with NiftiLabelsMasker extraction of timeseries data \
       after building a parcellations image.
    """
    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit(image_2)
    # transform to signals
    signals = parcellator.transform(image_2)

    # Test if the signals extracted are of same shape as inputs
    # Here, we simply return numpy array for single subject input
    assert signals.shape == (image_2.shape[3], n_parcel)
    # Test for single subject but in a list.
    signals = parcellator.transform([image_2])
    assert signals.shape == (image_2.shape[3], n_parcel)


@pytest.mark.slow
@pytest.mark.parametrize("verbose", [True, False, -1, 0, 1, 2])
def test_transform_verbose(image_2, verbose):
    """Test verbose mostly for coverage purpose."""
    parcellator = Parcellations(method="kmeans", n_parcels=5, verbose=verbose)
    parcellator.fit(image_2)
    parcellator.transform(image_2)


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_transform_multi_nifti_images(method, n_parcel, image_2):
    fmri_imgs = [image_2] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit(fmri_imgs)

    # transform multi images to signals.
    # In return, we have length equal to the number of images
    signals = parcellator.transform(fmri_imgs)

    assert signals[0].shape == (image_2.shape[3], n_parcel)
    assert signals[1].shape == (image_2.shape[3], n_parcel)
    assert signals[2].shape == (image_2.shape[3], n_parcel)
    assert len(signals) == len(fmri_imgs)


@pytest.mark.slow
@pytest.mark.parametrize("masker", [NiftiMasker, MultiNiftiMasker])
def test_transform_nifti_masker(masker, image_2, affine_eye):
    """Smoke test that 'mask' can be (multi)NiftiMasker.

    Regression test for https://github.com/nilearn/nilearn/issues/5926
    """
    fmri_imgs = [image_2] * 3

    mask_img = np.ones((10, 11, 12))
    mask_img = Nifti1Image(mask_img, affine_eye)

    mask = masker(mask_img=mask_img)

    parcellator = Parcellations(method="kmeans", mask=mask)
    parcellator.fit(fmri_imgs)
    parcellator.transform(fmri_imgs)


def test_check_parameters_transform(image_2, rng):
    # single confound
    confounds = rng.standard_normal(size=(10, 3))
    # Tests to check whether imgs, confounds returned are
    # list or not. Pre-check in parameters to work for list
    # of multi images and multi confounds
    imgs, confounds, single_subject = _check_parameters_transform(
        image_2, confounds
    )

    assert isinstance(imgs, (list, tuple))
    assert isinstance(confounds, (list, tuple))
    assert single_subject

    # confounds as pandas DataFrame
    imgs, confounds, single_subject = _check_parameters_transform(
        image_2, pd.DataFrame(np.array(confounds)[0])
    )

    assert isinstance(confounds, (list, tuple))

    # multi images
    fmri_imgs = [image_2] * 3
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


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_transform_with_multi_confounds_multi_images(
    method, n_parcel, image_2, rng
):
    fmri_imgs = [image_2] * 3
    confounds = rng.standard_normal(size=(10, 3))
    confounds_list = [confounds] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel)

    parcellator.fit(fmri_imgs)
    signals = parcellator.transform(fmri_imgs, confounds=confounds_list)

    assert isinstance(signals, list)
    # n_parcels=5, length of data=10
    assert signals[0].shape == (10, n_parcel)


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_fit_transform(method, n_parcel, image_2):
    fmri_imgs = [image_2] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    parcellator.fit_transform(fmri_imgs)

    assert parcellator.labels_img_ is not None
    if method not in ["kmeans", "rena", "hierarchical_kmeans"]:
        assert parcellator.connectivity_ is not None
    assert parcellator.masker_ is not None


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_fit_transform_with_confounds(method, n_parcel, image_2, rng):
    fmri_imgs = [image_2] * 3
    confounds = rng.standard_normal(size=(10, 3))
    confounds_list = [confounds] * 3

    parcellator = Parcellations(method=method, n_parcels=n_parcel)
    signals = parcellator.fit_transform(fmri_imgs, confounds=confounds_list)

    assert isinstance(signals, list)
    assert signals[0].shape == (10, n_parcel)


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcel", [5])
def test_inverse_transform_single_nifti_image(method, n_parcel, image_2):
    parcellate = Parcellations(method=method, n_parcels=n_parcel)
    parcellate.fit(image_2)

    assert parcellate.labels_img_ is not None

    fmri_reduced = parcellate.transform(image_2)

    assert isinstance(fmri_reduced, np.ndarray)
    # Shape matching with (scans, regions)
    assert fmri_reduced.shape == (10, n_parcel)

    fmri_compressed = parcellate.inverse_transform(fmri_reduced)

    # A single Nifti image for single subject input
    assert isinstance(fmri_compressed, Nifti1Image)
    # returns shape of fmri_img
    assert fmri_compressed.shape == image_2.shape

    # fmri_reduced in a list
    fmri_compressed = parcellate.inverse_transform([fmri_reduced])

    # A single Nifti image for single subject input
    assert isinstance(fmri_compressed, Nifti1Image)
    # returns shape of fmri_img
    assert fmri_compressed.shape == image_2.shape


@pytest.mark.slow
def test_transform_single_3d_input_images(affine_eye):
    """Test fit_transform single 3D image."""
    data = np.ones((10, 11, 12))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    img = Nifti1Image(data, affine=affine_eye)

    parcellate = Parcellations(method="ward", n_parcels=20)

    X = parcellate.fit_transform(img)

    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 20)


@pytest.mark.slow
def test_transform_list_3d_input_images(affine_eye):
    """Test fit_transform list 3D image."""
    data = np.ones((10, 11, 12))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    img = Nifti1Image(data, affine=affine_eye)
    imgs = [img] * 2

    parcellate = Parcellations(method="ward", n_parcels=20)
    X = parcellate.fit_transform(imgs)

    assert isinstance(X, list)
    # (number of samples, number of features)
    assert np.concatenate(X).shape == (2, 20)

    # inverse transform
    imgs_ = parcellate.inverse_transform(X)

    assert isinstance(imgs_, list)


@pytest.fixture
def n_samples():
    return 10


@pytest.fixture
def surface_img_for_parcellation(rng, n_samples):
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
    return SurfaceImage(mesh=mesh, data=data)


@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_parcels", [5, 9])
def test_parcellation_all_methods_with_surface(
    method, n_parcels, surface_img_for_parcellation, n_samples
):
    """Test if all parcellation methods work on surface."""
    parcellate = Parcellations(method=method, n_parcels=n_parcels)
    # fit and transform the data
    X_transformed = parcellate.fit_transform(surface_img_for_parcellation)
    # inverse transform the transformed data
    X_inverse = parcellate.inverse_transform(X_transformed)

    # make sure the n_features in transformed data were reduced to n_clusters
    assert X_transformed.shape == (n_samples, n_parcels)

    # make sure the inverse transformed data has the same shape as the original
    assert X_inverse.shape == surface_img_for_parcellation.shape


@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
@pytest.mark.parametrize("method", METHODS)
def test_parcellation_with_surface_and_confounds(
    method, rng, surface_img_for_parcellation, n_samples
):
    """Test if parcellation works on surface with confounds."""
    confounds = rng.standard_normal(size=(n_samples, 3))

    n_parcels = 5

    parcellate = Parcellations(method=method, n_parcels=n_parcels)

    X_transformed = parcellate.fit_transform(
        surface_img_for_parcellation, confounds=[confounds]
    )

    assert parcellate.n_elements_ == n_parcels

    assert X_transformed.shape == (n_samples, n_parcels)


@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
@pytest.mark.parametrize("method", METHODS)
def test_parcellation_with_multi_surface(
    method, surface_img_for_parcellation, n_samples
):
    """Test if parcellation works with surface data from multiple
    'subjects'.
    """
    surf_imgs = [surface_img_for_parcellation] * 3
    parcellate = Parcellations(method=method, n_parcels=5)
    X_transformed = parcellate.fit_transform(surf_imgs)

    assert X_transformed[0].shape == (n_samples, 5)
    assert len(X_transformed) == 3


@pytest.mark.flaky(reruns=5, reruns_delay=2, condition=is_windows_platform())
@pytest.mark.parametrize("method", METHODS)
def test_parcellation_with_surface_mask(
    method, surface_img_for_parcellation, n_samples
):
    """Test if parcellation works with surface data and a mask."""
    mask_data = {
        "left": np.ones(
            surface_img_for_parcellation.mesh.parts["left"].coordinates.shape[
                0
            ]
        ).astype(bool),
        "right": np.ones(
            surface_img_for_parcellation.mesh.parts["right"].coordinates.shape[
                0
            ]
        ).astype(bool),
    }
    mask_img = SurfaceImage(
        mesh=surface_img_for_parcellation.mesh, data=mask_data
    )
    parcellate = Parcellations(method=method, n_parcels=5, mask=mask_img)
    X_transformed = parcellate.fit_transform(surface_img_for_parcellation)

    assert X_transformed.shape == (n_samples, 5)


@pytest.mark.slow
@pytest.mark.parametrize("masker", [SurfaceMasker, MultiSurfaceMasker])
def test_transform_surface_masker(masker, surface_img_for_parcellation):
    """Smoke test that 'mask' can be (multi)SurfaceMasker.

    Regression test for https://github.com/nilearn/nilearn/issues/5926
    """
    fmri_imgs = [surface_img_for_parcellation] * 3

    mask_data = {
        "left": np.ones(
            surface_img_for_parcellation.mesh.parts["left"].coordinates.shape[
                0
            ]
        ).astype(bool),
        "right": np.ones(
            surface_img_for_parcellation.mesh.parts["right"].coordinates.shape[
                0
            ]
        ).astype(bool),
    }
    mask_img = SurfaceImage(
        mesh=surface_img_for_parcellation.mesh, data=mask_data
    )

    mask = masker(mask_img=mask_img)

    parcellator = Parcellations(method="kmeans", mask=mask)
    parcellator.fit(fmri_imgs)
    parcellator.transform(fmri_imgs)
