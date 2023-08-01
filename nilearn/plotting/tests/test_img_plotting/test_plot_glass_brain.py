"""Tests for :func:`nilearn.plotting.plot_glass_brain`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.image import get_data
from nilearn.plotting import plot_glass_brain


def test_plot_glass_brain(testdata_3d_for_plotting, tmpdir):
    """Smoke tests for plot_glass_brain with colorbar and negative values."""
    img = testdata_3d_for_plotting["img"]
    plot_glass_brain(img, colorbar=True, resampling_interpolation="nearest")
    # test plot_glass_brain with negative values
    plot_glass_brain(
        img, colorbar=True, plot_abs=False, resampling_interpolation="nearest"
    )


def test_plot_glass_brain_file_output(testdata_3d_for_plotting, tmpdir):
    """Smoke-test for hemispheric glass brain with file output."""
    filename = str(tmpdir.join("test.png"))
    plot_glass_brain(
        testdata_3d_for_plotting["img"],
        output_file=filename,
        display_mode="lzry",
    )
    plt.close()


def test_plot_noncurrent_axes():
    """Regression test for Issue #450."""
    rng = np.random.RandomState(42)
    maps_img = Nifti1Image(rng.random_sample((10, 10, 10)), np.eye(4))
    fh1 = plt.figure()
    fh2 = plt.figure()
    ax1 = fh1.add_subplot(1, 1, 1)

    assert plt.gcf() == fh2, "fh2  was the last plot created."

    # Since we gave ax1, the figure should be plotted in fh1.
    # Before #451, it was plotted in fh2.
    slicer = plot_glass_brain(maps_img, axes=ax1, title="test")
    for ax_name, niax in slicer.axes.items():
        ax_fh = niax.ax.get_figure()
        assert ax_fh == fh1, f"New axis {ax_name} should be in fh1."
    plt.close()


def test_add_markers_using_plot_glass_brain():
    """Tests for adding markers through plot_glass_brain."""
    fig = plot_glass_brain(None)
    coords = [(-34, -39, -9)]
    fig.add_markers(coords)
    fig.close()
    # Add a single marker in right hemisphere such that no marker
    # should appear in the left hemisphere when plotting
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
    # Add two markers in left hemisphere such that no marker
    # should appear in the right hemisphere when plotting
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


def test_plot_glass_brain_colorbar_having_nans(testdata_3d_for_plotting):
    """Smoke-test for plot_glass_brain and nans in the data image."""
    data = get_data(testdata_3d_for_plotting["img"])
    data[6, 5, 2] = np.inf
    plot_glass_brain(Nifti1Image(data, np.eye(4)), colorbar=True)
    plt.close()


@pytest.mark.parametrize("display_mode", ["lr", "lzry"])
def test_plot_glass_brain_display_modes_without_img(display_mode):
    """Smoke test for work around from PR #1888."""
    plot_glass_brain(None, display_mode=display_mode)
    plt.close()


@pytest.mark.parametrize("display_mode", ["lr", "lzry"])
def test_plot_glass_brain_with_completely_masked_img(display_mode):
    """Smoke test for PR #1888 with display modes having 'l'."""
    img = Nifti1Image(np.zeros((10, 20, 30)), np.eye(4))
    plot_glass_brain(img, display_mode=display_mode)
    plt.close()
