"""Tests for :func:`nilearn.plotting.plot_img`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils.niimg import _is_binary_niimg
from nilearn.image import get_data
from nilearn.plotting import plot_img


def _testdata_3d_for_plotting_for_resampling(img, binary):
    """Returns testing data for resampling tests.
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


def test_display_methods(testdata_3d_for_plotting):
    """Tests display methods."""
    img = testdata_3d_for_plotting["img"]
    display = plot_img(img)
    display.add_overlay(img, threshold=0)
    display.add_edges(img, color="c")
    display.add_contours(
        img, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )


def test_plot_with_axes_or_figure(testdata_3d_for_plotting):
    """Smoke tests for plot_img with providing figure or Axes."""
    img = testdata_3d_for_plotting["img"]
    figure = plt.figure()
    plot_img(img, figure=figure)
    ax = plt.subplot(111)
    plot_img(img, axes=ax)
    plt.close()


def test_plot_empty_slice(mni_affine):
    """Test that things don't crash when we give a map with
    nothing above threshold. This is only a smoke test.
    """
    img = Nifti1Image(np.zeros((20, 20, 20)), mni_affine)
    plot_img(img, display_mode="y", threshold=1)
    plt.close()


@pytest.mark.parametrize("display_mode", ["x", "y", "z"])
def test_plot_img_with_auto_cut_coords(display_mode, testdata_3d_for_plotting):
    """Smoke test for plot_img with cut_coords set in auto mode."""
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = Nifti1Image(data, np.eye(4))
    plot_img(img, cut_coords=None, display_mode=display_mode, black_bg=True)
    plt.close()


@pytest.mark.parametrize("binary_img", [True, False])
def test_plot_img_with_resampling(binary_img, testdata_3d_for_plotting):
    """Tests for plot_img with resampling of the data image."""
    img = _testdata_3d_for_plotting_for_resampling(
        testdata_3d_for_plotting["img"], binary_img
    )
    if binary_img:
        assert _is_binary_niimg(img)
    else:
        assert not _is_binary_niimg(img)
    display = plot_img(img)
    display.add_overlay(img)
    display.add_contours(
        img, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )
    display.add_edges(img, color="c")
    plt.close()


def test_display_methods_with_display_mode_tiled(testdata_3d_for_plotting):
    """Smoke tests for display methods with tiled display mode."""
    img = testdata_3d_for_plotting["img"]
    display = plot_img(img, display_mode="tiled")
    display.add_overlay(img, threshold=0)
    display.add_edges(img, color="c")
    display.add_contours(
        img, contours=2, linewidth=4, colors=["limegreen", "yellow"]
    )
