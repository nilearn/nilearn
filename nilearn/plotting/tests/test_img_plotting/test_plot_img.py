"""Tests for :func:`nilearn.plotting.plot_img`."""

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


def test_display_methods(img_3d_mni):
    """Tests display methods."""
    display = plot_img(img_3d_mni)
    display.add_overlay(img_3d_mni, threshold=0)
    display.add_edges(img_3d_mni, color="c")
    display.add_contours(
        img_3d_mni, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )


def test_plot_with_axes_or_figure(img_3d_mni):
    """Smoke tests for plot_img with providing figure or Axes."""
    figure = plt.figure()
    plot_img(img_3d_mni, figure=figure)
    ax = plt.subplot(111)
    plot_img(img_3d_mni, axes=ax)
    plt.close()


def test_plot_empty_slice(affine_mni):
    """Test that things don't crash when we give a map \
       with nothing above threshold. This is only a smoke test.
    """
    img = Nifti1Image(np.zeros((20, 20, 20)), affine_mni)
    plot_img(img, display_mode="y", threshold=1)
    plt.close()


@pytest.mark.parametrize("display_mode", ["x", "y", "z"])
def test_plot_img_with_auto_cut_coords(affine_eye, display_mode):
    """Smoke test for plot_img with cut_coords set in auto mode."""
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = Nifti1Image(data, affine_eye)
    plot_img(img, cut_coords=None, display_mode=display_mode, black_bg=True)
    plt.close()


@pytest.mark.parametrize("binary_img", [True, False])
def test_plot_img_with_resampling(binary_img, img_3d_mni):
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
    plt.close()


def test_display_methods_with_display_mode_tiled(img_3d_mni):
    """Smoke tests for display methods with tiled display mode."""
    display = plot_img(img_3d_mni, display_mode="tiled")
    display.add_overlay(img_3d_mni, threshold=0)
    display.add_edges(img_3d_mni, color="c")
    display.add_contours(
        img_3d_mni, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )
