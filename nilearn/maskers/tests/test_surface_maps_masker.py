import pytest

from nilearn.maskers import SurfaceMapsMasker


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
        match="each hemisphere of maps_img should have 2 dimensions",
    ):
        SurfaceMapsMasker(maps_img=surf_maps_img_1d).fit()


def test_surface_maps_masker_1d_img(surf_maps_img, surf_img):
    """Test that an error is raised when img has 1D data."""
    surf_img_1d = surf_img()
    surf_img_1d.data.parts["left"] = surf_img_1d.data.parts["left"].squeeze()
    surf_img_1d.data.parts["right"] = surf_img_1d.data.parts["right"].squeeze()

    with pytest.raises(
        ValueError,
        match="each hemisphere of img should have 2 dimensions",
    ):
        masker = SurfaceMapsMasker(maps_img=surf_maps_img).fit()
        masker.transform(surf_img_1d)
