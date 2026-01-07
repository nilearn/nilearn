"""Tests common to multiple image plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _affine_mni
from nilearn.datasets import load_mni152_template
from nilearn.image import get_data, reorder_img
from nilearn.plotting import (
    plot_anat,
    plot_carpet,
    plot_epi,
    plot_glass_brain,
    plot_img,
    plot_prob_atlas,
    plot_roi,
    plot_stat_map,
)
from nilearn.plotting.image.utils import MNI152TEMPLATE

ALL_PLOTTING_FUNCS = {
    plot_img,
    plot_anat,
    plot_stat_map,
    plot_roi,
    plot_epi,
    plot_glass_brain,
    plot_carpet,
    plot_prob_atlas,
}


PLOTTING_FUNCS_4D = {plot_prob_atlas, plot_carpet}


PLOTTING_FUNCS_3D = ALL_PLOTTING_FUNCS.difference(PLOTTING_FUNCS_4D)


def _add_nans_to_img(img, affine_mni=None):
    """Add nans in test image data."""
    if affine_mni is None:
        affine_mni = _affine_mni()
    data = get_data(img)
    data[6, 5, 1] = np.nan
    data[1, 5, 2] = np.nan
    data[1, 3, 2] = np.nan
    data[6, 5, 2] = np.inf
    return Nifti1Image(data, affine_mni)


def test_mni152template_is_reordered():
    """See issue #2550."""
    reordered_mni = reorder_img(load_mni152_template(resolution=2))
    assert np.allclose(get_data(reordered_mni), get_data(MNI152TEMPLATE))
    assert np.allclose(reordered_mni.affine, MNI152TEMPLATE.affine)
    assert np.allclose(reordered_mni.shape, MNI152TEMPLATE.shape)


@pytest.mark.parametrize(
    "plot_func",
    {
        plot_img,
        plot_anat,
        plot_stat_map,
        plot_roi,
        plot_glass_brain,
        plot_prob_atlas,
    },
)
def test_plot_functions_invalid_threshold(plot_func, img_3d_mni, tmp_path):
    """Test plot functions for negative threshold value."""
    filename = tmp_path / "temp.png"

    with pytest.raises(
        ValueError, match="Threshold should be a non-negative number!"
    ):
        plot_func(img_3d_mni, output_file=filename, threshold=-1)
    plt.close()


@pytest.mark.parametrize(
    "plot_func", PLOTTING_FUNCS_3D.difference({plot_glass_brain})
)
@pytest.mark.parametrize("cut_coords", [None, 5, (5, 4, 3)])
def test_plot_functions_mosaic_mode(plot_func, cut_coords, img_3d_rand_eye):
    """Smoke-test for plotting functions in mosaic mode."""
    plot_func(
        img_3d_rand_eye,
        display_mode="mosaic",
        title="mosaic mode",
        cut_coords=cut_coords,
    )
    plt.close()


@pytest.mark.parametrize("display_mode", ["x", "y", "z"])
def test_plot_functions_same_cut(display_mode, img_3d_rand_eye, tmp_path):
    """Make sure that passing several times the same cut for stacked slicers
       does not crash.

    Should also throw a warning that a cut has been removed.

    Regression test for:
    https://github.com/nilearn/nilearn/issues/5903
    """
    with pytest.warns(UserWarning, match="Dropping duplicates cuts from"):
        display = plot_img(
            img_3d_rand_eye,
            display_mode=display_mode,
            cut_coords=[3, 3],
        )
        display.savefig(tmp_path / "tmp.png")
        plt.close()


@pytest.mark.slow
@pytest.mark.parametrize("plot_func", [plot_stat_map, plot_glass_brain])
def test_plot_threshold_for_uint8(affine_eye, plot_func):
    """Mask was applied in [-threshold, threshold] which is problematic \
       for uint8 data.

    See https://github.com/nilearn/nilearn/issues/611 for more details.
    """
    data = 10 * np.ones((10, 10, 10), dtype="uint8")
    # Having a zero minimum value is important to reproduce
    # https://github.com/nilearn/nilearn/issues/762
    if plot_func is plot_stat_map:
        data[0, 0, 0] = 0
    else:
        data[0, 0] = 0
    img = Nifti1Image(data, affine_eye)
    threshold = 5
    kwargs = {"threshold": threshold, "display_mode": "z"}
    if plot_func is plot_stat_map:
        kwargs["bg_img"] = None
        kwargs["cut_coords"] = [0]
    display = plot_func(img, colorbar=True, **kwargs)
    # Next two lines retrieve the numpy array from the plot
    ax = next(iter(display.axes.values())).ax
    plotted_array = ax.images[0].get_array()
    # Make sure that there is one value masked
    assert plotted_array.mask.sum() == 1
    # Make sure that the value masked is in the corner. Note that the
    # axis orientation seem to be flipped, hence (0, 0) -> (-1, 0)
    assert plotted_array.mask[-1, 0]
    # Save execution time and memory
    plt.close()


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize(
    "display_mode,cut_coords",
    [("ortho", 2), ("tiled", 2), ("tiled", (2, 2)), ("mosaic", (2, 2))],
)
def test_invalid_cut_coords_with_display_mode(
    plot_func,
    display_mode,
    cut_coords,
    img_3d_mni,
):
    """Tests for invalid combinations of cut_coords and display_mode."""
    if plot_func is plot_glass_brain:
        return
    with pytest.raises(
        ValueError, match="cut_coords passed does not match the display mode"
    ):
        plot_func(
            img_3d_mni,
            display_mode=display_mode,
            cut_coords=cut_coords,
        )


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_with_nans(plot_func, img_3d_mni):
    """Smoke test for plotting functions with nans in data image."""
    plot_func(_add_nans_to_img(img_3d_mni))


@pytest.mark.slow
@pytest.mark.parametrize(
    "plot_func", [plot_roi, plot_stat_map, plot_glass_brain]
)
@pytest.mark.parametrize("cmap", ["Paired", "Set1", "Set2", "Set3", "viridis"])
def test_plotting_functions_with_cmaps(plot_func, cmap, img_3d_mni):
    """Some test for plotting functions with different cmaps."""
    plot_func(img_3d_mni, cmap=cmap, colorbar=True)
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_anat, plot_roi, plot_stat_map])
def test_plotting_functions_with_nans_in_bg_img(plot_func, img_3d_mni):
    """Smoke test for plotting functions with nans in background image."""
    bg_img = _add_nans_to_img(img_3d_mni)
    if plot_func is plot_anat:
        plot_func(bg_img)
    else:
        plot_func(img_3d_mni, bg_img=bg_img)
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_stat_map, plot_anat, plot_img])
def test_plotting_functions_with_display_mode_tiled(plot_func, img_3d_mni):
    """Smoke test for plotting functions with tiled display mode."""
    if plot_func is plot_anat:
        plot_func(display_mode="tiled")
    else:
        plot_func(img_3d_mni, display_mode="tiled")
    plt.close()


@pytest.mark.slow
@pytest.mark.parametrize("plot_func", [plot_stat_map, plot_img])
@pytest.mark.parametrize(
    "threshold, expected_ticks",
    [
        (0, [-10, -5, 0, 5, 10]),
        (0.1, [-10, -5, -0.1, 0.1, 5, 10]),
        (1.3, [-10, -5, -1.3, 1.3, 5, 10]),
        (3, [-10, -5, -3, 0, 3, 5, 10]),
        (3.5, [-10, -3.5, 0, 3.5, 10]),
        (7.5, [-10, -7.5, 0, 7.5, 10]),
        (9.9, [-10, -9.9, 0, 9.9, 10]),
    ],
)
def test_plot_symmetric_colorbar_threshold(
    tmp_path, plot_func, threshold, expected_ticks
):
    img_data = np.zeros((10, 10, 10))
    img_data[4:6, 2:4, 4:6] = -10
    img_data[5:7, 3:7, 3:6] = 10
    img = Nifti1Image(img_data, affine=np.eye(4))
    display = plot_func(img, threshold=threshold, colorbar=True)
    plt.savefig(tmp_path / "test.png")
    assert [
        float(tick.get_text()) for tick in display._cbar.ax.get_yticklabels()
    ] == expected_ticks
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_stat_map])
@pytest.mark.parametrize(
    "threshold, expected_ticks",
    [
        (0, [0, 2.5, 5, 7.5, 10]),
        (0.1, [0, 0.1, 2.5, 5, 7.5, 10]),
        (1.6, [0, 1.6, 2.5, 5, 7.5, 10]),
        (1.7, [0, 1.7, 5, 7.5, 10]),
        (3.3, [0, 3.3, 5, 7.5, 10]),
        (3.4, [0, 3.4, 5, 7.5, 10]),
        (6.6, [0, 6.6, 7.5, 10]),
        (6.7, [0, 6.7, 10]),
        (9.96, [0, 10]),
    ],
)
def test_plot_asymmetric_colorbar_threshold(
    tmp_path, plot_func, threshold, expected_ticks
):
    img_data = np.zeros((10, 10, 10))
    img_data[4:6, 2:4, 4:6] = 5
    img_data[5:7, 3:7, 3:6] = 10
    img = Nifti1Image(img_data, affine=np.eye(4))
    display = plot_func(img, threshold=threshold, colorbar=True)
    plt.savefig(tmp_path / "test.png")
    assert [
        float(tick.get_text()) for tick in display._cbar.ax.get_yticklabels()
    ] == expected_ticks
    plt.close()


@pytest.mark.slow
@pytest.mark.parametrize("plot_func", [plot_stat_map, plot_img])
@pytest.mark.parametrize("vmax", [None, 0])
def test_img_plotting_vmax_equal_to_zero(plot_func, vmax):
    """Make sure image plotting works if the maximum value is zero.

    Regression test for: https://github.com/nilearn/nilearn/issues/4203
    """
    img_data = np.zeros((10, 10, 10))
    img_data[4:6, 2:4, 4:6] = -5
    img = Nifti1Image(img_data, affine=np.eye(4))
    plot_func(img, colorbar=True, vmax=vmax)
