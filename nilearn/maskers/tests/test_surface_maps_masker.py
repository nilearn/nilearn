import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.conftest import _surf_maps_img
from nilearn.maskers import SurfaceMapsMasker
from nilearn.surface import SurfaceImage

ESTIMATORS_TO_CHECK = [SurfaceMapsMasker(_surf_maps_img())]

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
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_fit_transform_mask_vs_no_mask(
    surf_maps_img, surf_img_2d, surf_mask_1d
):
    """Test that fit_transform returns the different results when a mask is
    used vs. when no mask is used.
    """
    masker_with_mask = SurfaceMapsMasker(
        surf_maps_img, surf_mask_1d, standardize=None
    ).fit()
    region_signals_with_mask = masker_with_mask.transform(surf_img_2d(50))

    masker_no_mask = SurfaceMapsMasker(surf_maps_img, standardize=None).fit()
    region_signals_no_mask = masker_no_mask.transform(surf_img_2d(50))

    assert not (region_signals_with_mask == region_signals_no_mask).all()


def test_fit_transform_actual_output(surf_mesh, rng):
    """Test that fit_transform returns the expected output.
    Meaning that the SurfaceMapsMasker gives the solution to equation Ax = B,
    where A is the maps_img, x is the region_signals, and B is the img.
    """
    # create a maps_img with 9 vertices and 2 regions
    A = rng.random((9, 2))
    maps_data = {"left": A[:4, :], "right": A[4:, :]}
    surf_maps_img = SurfaceImage(surf_mesh, maps_data)

    # random region signals x
    expected_region_signals = rng.random((50, 2))

    # create an img with 9 vertices and 50 timepoints as B = A @ x
    B = np.dot(A, expected_region_signals.T)
    img_data = {"left": B[:4, :], "right": B[4:, :]}
    surf_img = SurfaceImage(surf_mesh, img_data)

    # get the region signals x using the SurfaceMapsMasker
    region_signals = SurfaceMapsMasker(
        surf_maps_img, standardize=None
    ).fit_transform(surf_img)

    assert region_signals.shape == expected_region_signals.shape
    assert np.allclose(region_signals, expected_region_signals)


def test_inverse_transform_actual_output(surf_mesh, rng):
    """Test that inverse_transform returns the expected output."""
    # create a maps_img with 9 vertices and 2 regions
    A = rng.random((9, 2))
    maps_data = {"left": A[:4, :], "right": A[4:, :]}
    surf_maps_img = SurfaceImage(surf_mesh, maps_data)

    # random region signals x
    expected_region_signals = rng.random((50, 2))

    # create an img with 9 vertices and 50 timepoints as B = A @ x
    B = np.dot(A, expected_region_signals.T)
    img_data = {"left": B[:4, :], "right": B[4:, :]}
    surf_img = SurfaceImage(surf_mesh, img_data)

    # get the region signals x using the SurfaceMapsMasker
    masker = SurfaceMapsMasker(surf_maps_img, standardize=None).fit()
    region_signals = masker.fit_transform(surf_img)
    X_inverse_transformed = masker.inverse_transform(region_signals)

    assert np.allclose(
        X_inverse_transformed.data.parts["left"], img_data["left"]
    )
    assert np.allclose(
        X_inverse_transformed.data.parts["right"], img_data["right"]
    )


def test_1d_maps_img(surf_img_1d):
    """Test that an error is raised when maps_img has 1D data."""
    with pytest.raises(
        ValueError,
        match="maps_img should be 2D",
    ):
        SurfaceMapsMasker(maps_img=surf_img_1d).fit()


def test_labels_img_none():
    """Test that an error is raised when maps_img is None."""
    with pytest.raises(
        ValueError,
        match="provide a maps_img during initialization",
    ):
        SurfaceMapsMasker(maps_img=None).fit()


@pytest.fixture
def non_overlapping_maps(rng, surf_mesh):
    """Generate maps with non-overlapping regions.

    Each vertex belong to only 1 region.
    """
    data = {
        "left": np.asarray(
            [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ]
        ),
        "right": np.asarray(
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 0],
            ]
        ),
    }
    # multiply with random "probability" values
    data = {part: data[part] * rng.random(data[part].shape) for part in data}
    return SurfaceImage(surf_mesh, data)


@pytest.fixture
def overlapping_maps(rng, surf_mesh):
    """Generate maps with overlapping regions.

    Some vertices have non null value for 2 different regions.
    """
    data = {
        "left": np.asarray(
            [
                [1, 1],  # overlap
                [0, 1],
                [1, 0],
                [1, 1],  # overlap
            ]
        ),
        "right": np.asarray(
            [
                [1, 0],
                [1, 1],  # overlap
                [0, 1],
                [1, 1],  # overlap
                [0, 0],
            ]
        ),
    }
    # multiply with random "probability" values
    data = {part: data[part] * rng.random(data[part].shape) for part in data}
    return SurfaceImage(surf_mesh, data)


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_non_overlapping_maps(
    allow_overlap, non_overlapping_maps, surf_img_2d
):
    """Test allow_overlap in SurfaceMapsMasker with non overlapping maps."""
    masker = SurfaceMapsMasker(
        non_overlapping_maps, allow_overlap=allow_overlap, standardize=None
    )
    masker.fit_transform(surf_img_2d(50))


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_overlapping_maps(allow_overlap, overlapping_maps, surf_img_2d):
    """Test allow_overlap in SurfaceMapsMasker with overlapping maps."""
    masker = SurfaceMapsMasker(
        overlapping_maps, allow_overlap=allow_overlap, standardize=None
    )
    if allow_overlap is False:
        with pytest.raises(ValueError, match="Overlap detected"):
            masker.fit_transform(surf_img_2d(50))
    else:
        masker.fit_transform(surf_img_2d(50))
