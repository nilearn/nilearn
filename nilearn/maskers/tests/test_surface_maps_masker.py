import pytest

from nilearn.maskers import SurfaceMapsMasker


def test_surface_maps_masker_fit_transform_shape(
    surf_maps_img, surf_img, surf_mask
):
    """Test that the fit_transform method returns the expected shape."""
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask()).fit()
    region_signals = masker.transform(surf_img(50))
    assert region_signals.shape == (
        surf_img(50).shape[0],
        surf_maps_img.shape[-1],
    )


def test_surface_maps_masker_inverse_transform_shape(
    surf_maps_img, surf_img, surf_mask
):
    """Test that inverse_transform returns an image with the same shape as the
    input.
    """
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask()).fit()
    region_signals = masker.fit_transform(surf_img(50))
    X_inverse_transformed = masker.inverse_transform(region_signals)
    assert X_inverse_transformed.shape == surf_img(50).shape


def test_surface_maps_masker_1d_maps_img(surf_img):
    """Test that an error is raised when maps_img has 1D data."""
    with pytest.raises(
        ValueError,
        match="each hemisphere of maps_img should have 2 dimensions",
    ):
        SurfaceMapsMasker(maps_img=surf_img()).fit()


def test_surface_maps_masker_1d_img(surf_maps_img, surf_img):
    """Test that an error is raised when img has 1D data."""
    with pytest.raises(
        ValueError,
        match="each hemisphere of img should have 2 dimensions",
    ):
        masker = SurfaceMapsMasker(maps_img=surf_maps_img).fit()
        masker.transform(surf_img())
