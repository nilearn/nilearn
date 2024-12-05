import numpy as np
import pytest

from nilearn.maskers import SurfaceMapsMasker
from nilearn.surface import SurfaceImage


def test_surface_maps_masker_fit_transform_shape(
    surf_maps_img, surf_img, surf_mask
):
    """Test that the fit_transform method returns the expected shape."""
    # TODO: remove after #4897 is merged
    surf_mask = surf_mask()
    surf_mask.data.parts["left"] = surf_mask.data.parts["left"].squeeze()
    surf_mask.data.parts["right"] = surf_mask.data.parts["right"].squeeze()

    masker = SurfaceMapsMasker(surf_maps_img, surf_mask).fit()
    region_signals = masker.transform(surf_img(50))
    # surf_img has shape (n_vertices, n_timepoints) = (9, 50)
    # surf_maps_img has shape (n_vertices, n_regions) = (9, 6)
    # region_signals should have shape (n_timepoints, n_regions) = (50, 6)
    assert region_signals.shape == (
        surf_img(50).shape[-1],
        surf_maps_img.shape[-1],
    )


def test_surface_maps_masker_fit_transform_mask_vs_no_mask(
    surf_maps_img, surf_img, surf_mask
):
    """Test that fit_transform returns the different results when a mask is
    used vs. when no mask is used.
    """
    # TODO: remove after #4897 is merged
    surf_mask = surf_mask()
    surf_mask.data.parts["left"] = surf_mask.data.parts["left"].squeeze()
    surf_mask.data.parts["right"] = surf_mask.data.parts["right"].squeeze()

    masker_with_mask = SurfaceMapsMasker(surf_maps_img, surf_mask).fit()
    region_signals_with_mask = masker_with_mask.transform(surf_img(50))

    masker_no_mask = SurfaceMapsMasker(surf_maps_img).fit()
    region_signals_no_mask = masker_no_mask.transform(surf_img(50))

    assert not (region_signals_with_mask == region_signals_no_mask).all()


def test_surface_maps_masker_fit_transform_actual_output(surf_mesh, rng):
    """Test that fit_transform returns the expected output.
    Meaning that the SurfaceMapsMasker gives the solution to equation Ax = B,
    where A is the maps_img, x is the region_signals, and B is the img.
    """
    # create a maps_img with 9 vertices and 2 regions
    A = rng.random((9, 2))
    maps_data = {"left": A[:4, :], "right": A[4:, :]}
    surf_maps_img = SurfaceImage(surf_mesh(), maps_data)

    # random region signals x
    expected_region_signals = rng.random((50, 2))

    # create an img with 9 vertices and 50 timepoints as B = A @ x
    B = np.dot(A, expected_region_signals.T)
    img_data = {"left": B[:4, :], "right": B[4:, :]}
    surf_img = SurfaceImage(surf_mesh(), img_data)

    # get the region signals x using the SurfaceMapsMasker
    region_signals = SurfaceMapsMasker(surf_maps_img).fit_transform(surf_img)

    assert region_signals.shape == expected_region_signals.shape
    assert np.allclose(region_signals, expected_region_signals)


def test_surface_maps_masker_inverse_transform_shape(
    surf_maps_img, surf_img, surf_mask
):
    """Test that inverse_transform returns an image with the same shape as the
    input.
    """
    # TODO: remove after #4897 is merged
    surf_mask = surf_mask()
    surf_mask.data.parts["left"] = surf_mask.data.parts["left"].squeeze()
    surf_mask.data.parts["right"] = surf_mask.data.parts["right"].squeeze()

    masker = SurfaceMapsMasker(surf_maps_img, surf_mask).fit()
    region_signals = masker.fit_transform(surf_img(50))
    X_inverse_transformed = masker.inverse_transform(region_signals)
    assert X_inverse_transformed.shape == surf_img(50).shape


def test_surface_maps_masker_inverse_transform_wrong_region_signals_shape(
    surf_maps_img, surf_img
):
    """Test that an error is raised when the region_signals shape is wrong."""
    masker = SurfaceMapsMasker(surf_maps_img).fit()
    region_signals = masker.fit_transform(surf_img(50))
    wrong_region_signals = region_signals[:, :-1]

    with pytest.raises(
        ValueError,
        match="Expected 6 regions, but got 5",
    ):
        masker.inverse_transform(wrong_region_signals)


def test_surface_maps_masker_1d_maps_img(surf_img):
    """Test that an error is raised when maps_img has 1D data."""
    # TODO: remove after #4897 is merged
    surf_maps_img_1d = surf_img()
    surf_maps_img_1d.data.parts["left"] = surf_maps_img_1d.data.parts[
        "left"
    ].squeeze()
    surf_maps_img_1d.data.parts["right"] = surf_maps_img_1d.data.parts[
        "right"
    ].squeeze()

    with pytest.raises(
        ValueError,
        match="maps_img should be 2D",
    ):
        SurfaceMapsMasker(maps_img=surf_maps_img_1d).fit()


def test_surface_maps_masker_1d_img(surf_maps_img, surf_img):
    """Test that an error is raised when img has 1D data."""
    surf_img_1d = surf_img()
    surf_img_1d.data.parts["left"] = surf_img_1d.data.parts["left"].squeeze()
    surf_img_1d.data.parts["right"] = surf_img_1d.data.parts["right"].squeeze()

    with pytest.raises(
        ValueError,
        match="img should be 2D",
    ):
        masker = SurfaceMapsMasker(maps_img=surf_maps_img).fit()
        masker.transform(surf_img_1d)


def test_surface_maps_masker_not_fitted_error(surf_maps_img):
    """Test that an error is raised when transform or inverse_transform is
    called before fit.
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    with pytest.raises(
        ValueError,
        match="SurfaceMapsMasker has not been fitted",
    ):
        masker.transform(None)
    with pytest.raises(
        ValueError,
        match="SurfaceMapsMasker has not been fitted",
    ):
        masker.inverse_transform(None)


def test_surface_maps_masker_smoothing_not_supported_error(
    surf_maps_img, surf_img
):
    """Test that an error is raised when smoothing_fwhm is not None."""
    masker = SurfaceMapsMasker(maps_img=surf_maps_img, smoothing_fwhm=1).fit()
    with pytest.warns(match="smoothing_fwhm is not yet supported"):
        masker.transform(surf_img(50))
        assert masker.smoothing_fwhm is None
