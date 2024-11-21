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
from nilearn.plotting.img_plotting import MNI152TEMPLATE

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
    reordered_mni = reorder_img(
        load_mni152_template(resolution=2), copy_header=True
    )
    assert np.allclose(get_data(reordered_mni), get_data(MNI152TEMPLATE))
    assert np.allclose(reordered_mni.affine, MNI152TEMPLATE.affine)
    assert np.allclose(reordered_mni.shape, MNI152TEMPLATE.shape)


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_3d_default_params(plot_func, img_3d_mni, tmp_path):
    """Smoke tests for 3D plotting functions with default parameters."""
    filename = tmp_path / "temp.png"
    plot_func(img_3d_mni, output_file=filename)
    plt.close()


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("cbar_tick_format", ["%f", "%i"])
def test_cbar_tick_format(plot_func, img_3d_mni, cbar_tick_format, tmp_path):
    """Test different colorbar tick format with 3D plotting functions."""
    filename = tmp_path / "temp.png"
    plot_func(
        img_3d_mni,
        output_file=filename,
        colorbar=True,
        cbar_tick_format=cbar_tick_format,
    )
    plt.close()


def test_plot_carpet_default_params(img_4d_mni, img_3d_ones_mni, tmp_path):
    """Smoke-test for 4D plot_carpet with default arguments."""
    plot_carpet(
        img_4d_mni, mask_img=img_3d_ones_mni, output_file=tmp_path / "temp.png"
    )
    plt.close()


def test_plot_prob_atlas_default_params(img_3d_mni, img_4d_mni, tmp_path):
    """Smoke-test for plot_prob_atlas with default arguments."""
    plot_prob_atlas(
        img_4d_mni, bg_img=img_3d_mni, output_file=tmp_path / "temp.png"
    )
    plt.close()


@pytest.mark.parametrize(
    "plot_func", PLOTTING_FUNCS_3D.difference({plot_glass_brain})
)
@pytest.mark.parametrize("cut_coords", [None, 5, (5, 4, 3)])
def test_plot_functions_mosaic_mode(plot_func, cut_coords, img_3d_mni):
    """Smoke-test for plotting functions in mosaic mode."""
    plot_func(
        img_3d_mni,
        display_mode="mosaic",
        title="mosaic mode",
        cut_coords=cut_coords,
    )
    plt.close()


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
    threshold = np.array(5, dtype="uint8")
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


@pytest.fixture
def expected_error_message(display_mode, cut_coords):
    """Return the expected error message depending on display_mode \
       and cut_coords. Used in test_invalid_cut_coords_with_display_mode.
    """
    if display_mode == "ortho" or (
        display_mode == "tiled" and cut_coords == 2
    ):
        return (
            f"The input given for display_mode='{display_mode}' needs to "
            "be a list of 3d world coordinates."
        )
    return "The number cut_coords passed does not match the display_mode"


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
    expected_error_message,
):
    """Tests for invalid combinations of cut_coords and display_mode."""
    if plot_func is plot_glass_brain and display_mode != "ortho":
        return
    with pytest.raises(ValueError, match=expected_error_message):
        plot_func(
            img_3d_mni,
            display_mode=display_mode,
            cut_coords=cut_coords,
        )


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_with_nans(plot_func, img_3d_mni):
    """Smoke test for plotting functions with nans in data image."""
    plot_func(_add_nans_to_img(img_3d_mni))


@pytest.mark.parametrize(
    "plot_func", [plot_roi, plot_stat_map, plot_glass_brain]
)
@pytest.mark.parametrize("cmap", ["Paired", "Set1", "Set2", "Set3", "viridis"])
def test_plotting_functions_with_cmaps(plot_func, cmap):
    """Some test for plotting functions with different cmaps."""
    plot_func(load_mni152_template(resolution=2), cmap=cmap, colorbar=True)
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


@pytest.mark.parametrize(
    "plotting_func",
    [
        plot_img,
        plot_anat,
        plot_stat_map,
        plot_roi,
        plot_epi,
        plot_glass_brain,
    ],
)
def test_plotting_functions_radiological_view(img_3d_mni, plotting_func):
    """Smoke test for radiological view."""
    result = plotting_func(img_3d_mni, radiological=True)
    assert result.axes.get("y").radiological is True
    plt.close()


functions = [plot_stat_map, plot_img]
EXPECTED = [(i, ["-10", "-5", "0", "5", "10"]) for i in [0, 0.1, 0.9, 1]]
EXPECTED += [
    (i, ["-10", f"-{i}", "0", f"{i}", "10"]) for i in [1.3, 2.5, 3, 4.9, 7.5]
]
EXPECTED += [(i, [f"-{i}", "-5", "0", "5", f"{i}"]) for i in [7.6, 8, 9.9]]


@pytest.mark.parametrize(
    "plot_func, threshold, expected_ticks",
    [(f, e[0], e[1]) for e in EXPECTED for f in functions],
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
        tick.get_text() for tick in display._cbar.ax.get_yticklabels()
    ] == expected_ticks
    plt.close()


functions = [plot_stat_map]
EXPECTED2 = [(0, ["0", "2.5", "5", "7.5", "10"])]
EXPECTED2 += [(i, [f"{i}", "2.5", "5", "7.5", "10"]) for i in [0.1, 0.3, 1.2]]
EXPECTED2 += [
    (i, ["0", f"{i}", "5", "7.5", "10"]) for i in [1.3, 1.9, 2.5, 3, 3.7]
]
EXPECTED2 += [(i, ["0", "2.5", f"{i}", "7.5", "10"]) for i in [3.8, 4, 5, 6.2]]
EXPECTED2 += [(i, ["0", "2.5", "5", f"{i}", "10"]) for i in [6.3, 7.5, 8, 8.7]]
EXPECTED2 += [(i, ["0", "2.5", "5", "7.5", f"{i}"]) for i in [8.8, 9, 9.9]]


@pytest.mark.parametrize(
    "plot_func, threshold, expected_ticks",
    [(f, e[0], e[1]) for e in EXPECTED2 for f in functions],
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
        tick.get_text() for tick in display._cbar.ax.get_yticklabels()
    ] == expected_ticks
    plt.close()


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
