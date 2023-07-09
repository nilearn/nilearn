"""Tests common to multiple image plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import MNI_AFFINE
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


def _test_data_with_nans(img):
    """Add nans in test image data."""
    data = get_data(img)
    data[6, 5, 1] = np.nan
    data[1, 5, 2] = np.nan
    data[1, 3, 2] = np.nan
    data[6, 5, 2] = np.inf
    return Nifti1Image(data, MNI_AFFINE)


def test_mni152template_is_reordered():
    """See issue #2550."""
    reordered_mni = reorder_img(load_mni152_template(resolution=2))
    assert np.allclose(get_data(reordered_mni), get_data(MNI152TEMPLATE))
    assert np.allclose(reordered_mni.affine, MNI152TEMPLATE.affine)
    assert np.allclose(reordered_mni.shape, MNI152TEMPLATE.shape)


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_3d_default_params(
    plot_func, testdata_3d_for_plotting, tmpdir
):
    """Smoke tests for 3D plotting functions with default parameters."""
    filename = str(tmpdir.join("temp.png"))
    plot_func(testdata_3d_for_plotting["img"], output_file=filename)
    plt.close()


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("cbar_tick_format", ["%f", "%i"])
def test_cbar_tick_format(
    plot_func, testdata_3d_for_plotting, cbar_tick_format, tmpdir
):
    """Test different colorbar tick format with 3D plotting functions."""
    filename = str(tmpdir.join("temp.png"))
    plot_func(
        testdata_3d_for_plotting["img"],
        output_file=filename,
        colorbar=True,
        cbar_tick_format=cbar_tick_format,
    )
    plt.close()


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_4D)
def test_plot_functions_4d_default_params(
    plot_func, testdata_3d_for_plotting, testdata_4d_for_plotting, tmpdir
):
    """Smoke-test for 4D plotting functions with default arguments."""
    filename = str(tmpdir.join("temp.png"))
    kwargs = {"output_file": filename}
    if plot_func == plot_carpet:
        kwargs["mask_img"] = testdata_4d_for_plotting["img_mask"]
    else:
        kwargs["bg_img"] = testdata_3d_for_plotting["img"]
    plot_func(testdata_4d_for_plotting["img_4d"], **kwargs)
    plt.close()


@pytest.mark.parametrize(
    "plot_func", PLOTTING_FUNCS_3D.difference({plot_glass_brain})
)
@pytest.mark.parametrize("cut_coords", [None, 5, (5, 4, 3)])
def test_plot_functions_mosaic_mode(
    plot_func, cut_coords, testdata_3d_for_plotting
):
    """Smoke-test for plotting functions in mosaic mode."""
    plot_func(
        testdata_3d_for_plotting["img"],
        display_mode="mosaic",
        title="mosaic mode",
        cut_coords=cut_coords,
    )
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_stat_map, plot_glass_brain])
def test_plot_threshold_for_uint8(plot_func, testdata_3d_for_plotting):
    """Mask was applied in [-threshold, threshold] which is problematic
    for uint8 data. See https://github.com/nilearn/nilearn/issues/611
    for more details.
    """
    data = 10 * np.ones((10, 10, 10), dtype="uint8")
    # Having a zero minimum value is important to reproduce
    # https://github.com/nilearn/nilearn/issues/762
    if plot_func == plot_stat_map:
        data[0, 0, 0] = 0
    else:
        data[0, 0] = 0
    affine = np.eye(4)
    img = Nifti1Image(data, affine)
    threshold = np.array(5, dtype="uint8")
    kwargs = {"threshold": threshold, "display_mode": "z"}
    if plot_func == plot_stat_map:
        kwargs["bg_img"] = None
        kwargs["cut_coords"] = [0]
    display = plot_func(img, colorbar=True, **kwargs)
    # Next two lines retrieve the numpy array from the plot
    ax = list(display.axes.values())[0].ax
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
    """Return the expected error message depending on display_mode and
    cut_coords. Used in test_invalid_cut_coords_with_display_mode.
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
    testdata_3d_for_plotting,
    expected_error_message,
):
    """Tests for invalid combinations of cut_coords and display_mode."""
    if plot_func == plot_glass_brain and display_mode != "ortho":
        return
    with pytest.raises(ValueError, match=expected_error_message):
        plot_func(
            testdata_3d_for_plotting["img"],
            display_mode=display_mode,
            cut_coords=cut_coords,
        )


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_with_nans(plot_func, testdata_3d_for_plotting):
    """Smoke test for plotting functions with nans in data image."""
    plot_func(_test_data_with_nans(testdata_3d_for_plotting["img"]))


@pytest.mark.parametrize(
    "plot_func", [plot_roi, plot_stat_map, plot_glass_brain]
)
@pytest.mark.parametrize("cmap", ["Paired", "Set1", "Set2", "Set3", "viridis"])
def test_plotting_functions_with_cmaps(plot_func, cmap):
    """Some test for plotting functions with different cmaps."""
    plot_func(load_mni152_template(resolution=2), cmap=cmap, colorbar=True)
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_anat, plot_roi, plot_stat_map])
def test_plotting_functions_with_nans_in_bg_img(
    plot_func, testdata_3d_for_plotting
):
    """Smoke test for plotting functions with nans in background image."""
    bg_img = _test_data_with_nans(testdata_3d_for_plotting["img"])
    if plot_func == plot_anat:
        plot_func(bg_img)
    else:
        plot_func(testdata_3d_for_plotting["img"], bg_img=bg_img)
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_stat_map, plot_anat, plot_img])
def test_plotting_functions_with_display_mode_tiled(
    plot_func, testdata_3d_for_plotting
):
    """Smoke test for plotting functions with tiled display mode."""
    if plot_func == plot_anat:
        plot_func(display_mode="tiled")
    else:
        plot_func(testdata_3d_for_plotting["img"], display_mode="tiled")
    plt.close()
