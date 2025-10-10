"""Tests for :func:`nilearn.plotting.plot_glass_brain`."""

# ruff: noqa: ARG001

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.image import get_data
from nilearn.plotting import plot_glass_brain


@pytest.mark.parametrize("plot_abs", [True, False])
@pytest.mark.parametrize("resampling_interpolation", ["nearest", "continuous"])
def test_plot_glass_brain_absolute(
    matplotlib_pyplot, img_3d_mni, plot_abs, resampling_interpolation
):
    """Smoke test resampling_interpolation and for absolute value plotting."""
    plot_glass_brain(img_3d_mni, plot_abs=plot_abs)


def test_plot_glass_brain_file_output(matplotlib_pyplot, img_3d_mni, tmp_path):
    """Smoke-test for hemispheric glass brain with file output."""
    filename = tmp_path / "test.png"
    plot_glass_brain(
        img_3d_mni,
        output_file=filename,
        display_mode="lzry",
    )


def test_plot_noncurrent_axes(matplotlib_pyplot, rng):
    """Regression test for Issue #450."""
    maps_img = Nifti1Image(rng.random((10, 10, 10)), np.eye(4))
    fh1 = plt.figure()
    fh2 = plt.figure()
    ax1 = fh1.add_subplot(1, 1, 1)

    assert plt.gcf() == fh2, "fh2 was the last plot created."

    # Since we gave ax1, the figure should be plotted in fh1.
    # Before #451, it was plotted in fh2.
    slicer = plot_glass_brain(maps_img, axes=ax1, title="test")
    for ax_name, niax in slicer.axes.items():
        ax_fh = niax.ax.get_figure()
        assert ax_fh == fh1, f"New axis {ax_name} should be in fh1."


def test_add_markers_using_plot_glass_brain(matplotlib_pyplot):
    """Tests for adding markers through plot_glass_brain."""
    fig = plot_glass_brain(None)
    coords = [(-34, -39, -9)]
    fig.add_markers(coords)
    fig.close()


def test_add_markers_right_hemi(matplotlib_pyplot):
    """Add a single marker in right hemisphere.

    No marker should appear in the left hemisphere when plotting.
    """
    display = plot_glass_brain(None, display_mode="lyrz")
    display.add_markers([[20, 20, 20]])

    # Check that Axe 'l' has no marker
    assert display.axes["l"].ax.collections[0].get_offsets().data.shape == (
        0,
        2,
    )
    # Check that all other Axes have one marker
    for d in "ryz":
        assert display.axes[d].ax.collections[0].get_offsets().data.shape == (
            1,
            2,
        )


def test_add_markers_left_hemi(matplotlib_pyplot):
    """Add two markers in left hemisphere.

    No marker should appear in the right hemisphere when plotting.
    """
    display = plot_glass_brain(None, display_mode="lyrz")
    display.add_markers(
        [[-20, 20, 20], [-10, 10, 10]], marker_color=["r", "b"]
    )
    # Check that Axe 'r' has no marker
    assert display.axes["r"].ax.collections[0].get_offsets().data.shape == (
        0,
        2,
    )
    # Check that all other Axes have two markers
    for d in "lyz":
        assert display.axes[d].ax.collections[0].get_offsets().data.shape == (
            2,
            2,
        )


def test_plot_glass_brain_colorbar_having_nans(
    matplotlib_pyplot, affine_eye, img_3d_mni
):
    """Smoke-test for plot_glass_brain and nans in the data image."""
    data = get_data(img_3d_mni)
    data[6, 5, 2] = np.inf
    plot_glass_brain(Nifti1Image(data, affine_eye), colorbar=True)


@pytest.mark.parametrize("display_mode", ["lr", "lzry"])
def test_plot_glass_brain_display_modes_without_img(
    matplotlib_pyplot, display_mode
):
    """Smoke test for work around from PR #1888."""
    plot_glass_brain(None, display_mode=display_mode)


@pytest.mark.parametrize("display_mode", ["lr", "lzry"])
def test_plot_glass_brain_with_completely_masked_img(
    matplotlib_pyplot, img_3d_mni, display_mode
):
    """Smoke test for PR #1888 with display modes having 'l'."""
    plot_glass_brain(img_3d_mni, display_mode=display_mode)


def test_plot_glass_brain_negative_vmin_with_plot_abs(
    matplotlib_pyplot, img_3d_mni
):
    """Test that warning is thrown if plot_abs is True and vmin is negative."""
    warning_message = "vmin is negative but plot_abs is True"
    with pytest.warns(UserWarning, match=warning_message):
        plot_glass_brain(img_3d_mni, vmin=-2, plot_abs=True)
