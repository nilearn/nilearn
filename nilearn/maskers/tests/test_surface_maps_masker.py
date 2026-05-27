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

ESTIMATORS_TO_CHECK = [
    SurfaceMapsMasker(_surf_maps_img()),
    SurfaceMapsMasker(_surf_maps_img(n_regions=1)),
]

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
    with pytest.raises(ValueError, match="maps_img should be 2D"):
        SurfaceMapsMasker(maps_img=surf_img_1d).fit()


def test_labels_img_none():
    """Test that an error is raised when maps_img is None."""
    with pytest.raises(
        ValueError,
        match="provide a maps_img during initialization",
    ):
        SurfaceMapsMasker(maps_img=None).fit()


def test_surface_maps_masker_empty_map_img_error(surf_mesh):
    """Raise error if map_img is empty."""
    maps_img = SurfaceImage(
        mesh=surf_mesh,
        data={
            "left": np.asarray([[0, 0, 0, 0]]).T,
            "right": np.asarray([[0, 0, 0, 0, 0]]).T,
        },
    )
    with pytest.raises(
        ValueError,
        match="maps_img contains no map",
    ):
        SurfaceMapsMasker(maps_img=maps_img).fit()


def test_surface_maps_masker_mask_img_masks_all_maps_error(surf_mesh):
    """Raise error if mask_img excludes all vertices with map value."""
    maps_img = SurfaceImage(
        mesh=surf_mesh,
        data={
            "left": np.asarray([[0.5, 0.3, 0, 0]]).T,
            "right": np.asarray([[0.6, 0.3, 0.7, 0, 0]]).T,
        },
    )
    mask_img = SurfaceImage(
        mesh=surf_mesh,
        data={
            "left": np.asarray([0, 0, 1, 1]),
            "right": np.asarray([0, 0, 0, 1, 1]),
        },
    )
    with pytest.raises(
        ValueError,
        match="No map left after applying mask to the maps image",
    ):
        SurfaceMapsMasker(maps_img=maps_img, mask_img=mask_img).fit()


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


@pytest.fixture
def overlapping_maps2(surf_mesh):
    """Generate maps with overlapping regions.

    Some vertices have non null value for 2 different regions.
    """
    data = {
        "left": np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0.7, 0.1],  # overlap
            ]
        ),
        "right": np.asarray(
            [
                [0, 0],
                [0.3, 0.6],  # overlap
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        ),
    }
    return SurfaceImage(surf_mesh, data)


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_non_overlapping_maps(
    allow_overlap, non_overlapping_maps, surf_img_2d
):
    """Test allow_overlap in SurfaceMapsMasker with non overlapping maps."""
    masker = SurfaceMapsMasker(
        non_overlapping_maps, allow_overlap=allow_overlap, standardize=None
    )
    masker.fit()
    surf_img = surf_img_2d(10)
    region_signals = masker.fit_transform(surf_img)
    assert np.allclose(
        region_signals,
        np.array(
            [
                [0.30, 13.22],
                [16.12, 52.14],
                [31.94, 91.07],
                [47.77, 129.99],
                [63.59, 168.92],
                [79.41, 207.84],
                [95.24, 246.77],
                [111.06, 285.69],
                [126.88, 324.61],
                [142.70, 363.54],
            ]
        ),
        atol=1e-02,
    )


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_overlapping_maps(allow_overlap, overlapping_maps, surf_img_2d):
    """Test allow_overlap in SurfaceMapsMasker with overlapping maps."""
    masker = SurfaceMapsMasker(
        overlapping_maps, allow_overlap=allow_overlap, standardize=None
    )
    surf_img = surf_img_2d(10)
    if allow_overlap is False:
        with pytest.raises(ValueError, match="Overlap detected"):
            masker.fit_transform(surf_img)
    else:
        region_signals = masker.fit_transform(surf_img)
        assert np.allclose(
            region_signals,
            np.array(
                [
                    [2.76, 11.01],
                    [5.71, 46.53],
                    [8.65, 82.05],
                    [11.60, 117.58],
                    [14.54, 153.10],
                    [17.49, 188.62],
                    [20.43, 224.14],
                    [23.38, 259.67],
                    [26.32, 295.19],
                    [29.26, 330.71],
                ]
            ),
            atol=1e-02,
        )


def test_overlapping_maps2(overlapping_maps2, surf_img_2d):
    """Test `allow_overlap=True` in SurfaceMapsMasker with overlapping maps
    containing only one vertex on each hemisphere.
    """
    masker = SurfaceMapsMasker(
        overlapping_maps2, allow_overlap=True, standardize=None
    )
    surf_img = surf_img_2d(10)
    region_signals = masker.fit_transform(surf_img)
    assert np.allclose(
        region_signals,
        np.array(
            [
                [2.05, 15.64],
                [-4.61, 102.30],
                [-11.28, 188.97],
                [-17.94, 275.64],
                [-24.61, 362.30],
                [-31.28, 448.97],
                [-37.94, 535.64],
                [-44.61, 622.30],
                [-51.28, 708.97],
                [-57.94, 795.64],
            ]
        ),
        atol=1e-02,
    )
