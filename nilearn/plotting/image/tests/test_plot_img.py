"""Tests for :func:`nilearn.plotting.plot_img`."""

# ruff: noqa: ARG001

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils.niimg import is_binary_niimg
from nilearn.image import get_data
from nilearn.plotting import plot_img


def _testdata_3d_for_plotting_for_resampling(img, binary):
    """Return testing data for resampling tests.

    Data can be binarize or not.
    """
    data = get_data(img)
    if binary:
        data[data > 0] = 1
        data[data < 0] = 0
    affine = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return Nifti1Image(data, affine)


@pytest.mark.slow
def test_display_methods(matplotlib_pyplot, img_3d_mni):
    """Tests display methods."""
    display = plot_img(img_3d_mni)
    display.add_overlay(img_3d_mni, threshold=0)
    display.add_edges(img_3d_mni, color="c")
    display.add_contours(
        img_3d_mni, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )


def test_display_methods_invalid_threshold(matplotlib_pyplot, img_3d_mni):
    """Tests display methods for negative threshold."""
    with pytest.raises(
        ValueError, match="Threshold should be a non-negative number!"
    ):
        display = plot_img(img_3d_mni)
        display.add_overlay(img_3d_mni, threshold=-1)

    with pytest.raises(
        ValueError, match="Threshold should be a non-negative number!"
    ):
        display = plot_img(img_3d_mni)
        display.add_contours(
            img_3d_mni, contours=2, linewidth=4, threshold=-1, filled=True
        )


def test_plot_with_axes_or_figure(matplotlib_pyplot, img_3d_mni):
    """Smoke tests for plot_img with providing figure or Axes."""
    figure = plt.figure()
    plot_img(img_3d_mni, figure=figure)
    ax = plt.subplot(111)
    plot_img(img_3d_mni, axes=ax)


def test_plot_empty_slice(matplotlib_pyplot, affine_mni):
    """Test that things don't crash when we give a map \
       with nothing above threshold. This is only a smoke test.
    """
    img = Nifti1Image(np.zeros((20, 20, 20)), affine_mni)
    plot_img(img, display_mode="y", threshold=1)


@pytest.mark.parametrize("binary_img", [True, False])
def test_plot_img_with_resampling(matplotlib_pyplot, binary_img, img_3d_mni):
    """Tests for plot_img with resampling of the data image."""
    img = _testdata_3d_for_plotting_for_resampling(img_3d_mni, binary_img)
    if binary_img:
        assert is_binary_niimg(img)
    else:
        assert not is_binary_niimg(img)
    display = plot_img(img)
    display.add_overlay(img)
    display.add_contours(
        img, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )
    display.add_edges(img, color="c")


@pytest.mark.slow
def test_display_methods_with_display_mode_tiled(
    matplotlib_pyplot, img_3d_mni
):
    """Smoke tests for display methods with tiled display mode."""
    display = plot_img(img_3d_mni, display_mode="tiled")
    display.add_overlay(img_3d_mni, threshold=0)
    display.add_edges(img_3d_mni, color="c")
    display.add_contours(
        img_3d_mni, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )


@pytest.mark.parametrize("transparency", [-1, 10])
def test_plot_img_transparency_warning(
    matplotlib_pyplot, img_3d_ones_mni, transparency
):
    """Test transparency is reset to proper values."""
    with pytest.warns(
        UserWarning, match="'transparency' must be in the interval"
    ):
        plot_img(img_3d_ones_mni, transparency=transparency)


@pytest.mark.parametrize("transparency_range", [[10, -1], [5]])
def test_plot_img_transparency_range_error(
    matplotlib_pyplot, img_3d_ones_mni, transparency_range, transparency_image
):
    """Test transparency_range invalid values."""
    with pytest.raises(
        ValueError, match="list or tuple of 2 non-negative numbers"
    ):
        plot_img(
            img_3d_ones_mni,
            transparency=transparency_image,
            transparency_range=transparency_range,
        )


def test_plot_img_transparency_binary_image(
    matplotlib_pyplot,
    shape_3d_default,
    affine_mni,
    rng,
    img_3d_ones_mni,
):
    """Smoke test with transparency image as binary."""
    transparency_data = rng.choice(
        [0, 1], size=shape_3d_default, p=[0.5, 0.5]
    ).astype("int8")
    transparency_image = Nifti1Image(transparency_data, affine_mni)

    plot_img(
        img_3d_ones_mni,
        transparency=transparency_image,
    )


@pytest.mark.parametrize(
    "display_mode, cut_coords",
    [
        ("x", [7]),
        ("x", [7, 8]),
        ("y", [8]),
        ("y", [8, 9]),
        ("z", [9]),
        ("z", [9, 10]),
        ("xz", [7, 9]),
        ("yz", [8, 9]),
        ("yx", [7, 8]),
        ("ortho", [7, 8, 9]),
        ("tiled", [7, 8, 9]),
    ],
)
def test_cut_coords_out_of_bounds_error(
    matplotlib_pyplot, img_3d_rand_eye, display_mode, cut_coords
):
    """Test if nilearn.plotting.image.plot_img raises error when all elements
    of cut_coords are out of bounds of the image for corresponding coordinate.
    """
    # img_3d_rand_eye has bounds:
    # [(0.0, 6.0), (0.0, 7.0), (0.0, 8.0)]
    with pytest.raises(ValueError, match="is out of the bounds of the image"):
        plot_img(
            img_3d_rand_eye, display_mode=display_mode, cut_coords=cut_coords
        )


@pytest.mark.parametrize(
    "display_mode, cut_coords",
    [
        ("x", [7, 6]),
        ("x", [6, 7, 8]),
        ("y", [7, 8]),
        ("y", [6, 8, 9]),
        ("z", [8, 9]),
        ("z", [9, 10, 8]),
        ("xz", [6, 9]),
        ("xz", [7, 8]),
        ("yz", [7, 9]),
        ("yz", [8, 7]),
        ("yx", [6, 8]),
        ("yx", [8, 7]),
        ("ortho", [6, 8, 9]),
        ("tiled", [8, 7, 9]),
    ],
)
def test_cut_coords_out_of_bounds_warning(
    matplotlib_pyplot, img_3d_rand_eye, display_mode, cut_coords
):
    """Test if nilearn.plotting.image.plot_img raises error when at least one
    but not all elements of cut_coords is out of bounds of the image for
    corresponding coordinate.
    """
    # img_3d_rand_eye has bounds:
    # [(0.0, 6.0), (0.0, 7.0), (0.0, 8.0)]
    with pytest.warns(
        UserWarning,
        match=("The following 'cut_coords'"),
    ):
        plot_img(
            img_3d_rand_eye, display_mode=display_mode, cut_coords=cut_coords
        )


@pytest.mark.parametrize("display_mode", ["ortho", "tiled"])
@pytest.mark.parametrize("cut_coords", [5, [3, 5]])
def test_error_incompatible_cut_coords_3d(
    matplotlib_pyplot, img_3d_rand_eye, display_mode, cut_coords
):
    """Test error when incompatible cut_coords is specified for slicers of type
    `ortho` and `tiled`.
    """
    with pytest.raises(
        ValueError,
        match=("cut_coords passed does not match the display mode"),
    ):
        plot_img(
            img_3d_rand_eye, display_mode=display_mode, cut_coords=cut_coords
        )


def test_error_incompatible_cut_coords_mosaic(
    matplotlib_pyplot, img_3d_rand_eye
):
    """Test error when incompatible cut_coords is specified for `mosaic`
    slicer.
    """
    with pytest.raises(
        ValueError,
        match=("cut_coords passed does not match the display mode"),
    ):
        plot_img(img_3d_rand_eye, display_mode="mosaic", cut_coords=[3, 5])


@pytest.mark.parametrize("display_mode", ["xz", "yz", "yx"])
@pytest.mark.parametrize("cut_coords", [5, [3, 5, 7]])
def test_error_incompatible_cut_coords_2d(
    matplotlib_pyplot, img_3d_rand_eye, display_mode, cut_coords
):
    """Test error when incompatible cut_coords is specified for slicers of type
    `xy`, `yz` and `yx`.
    """
    with pytest.raises(
        ValueError,
        match=("cut_coords passed does not match the display mode"),
    ):
        plot_img(
            img_3d_rand_eye, display_mode=display_mode, cut_coords=cut_coords
        )
