"""
Test if public plotting functions' output has changed.

Sometimes, the output of a plotting function may unintentionanly change as a
side effect of changing another function or piece of code that it depends on.
These tests ensure that the outputs are not accidentally changed.

Failures are expected at times when the output is changed intentionally
(e.g. fixing a bug or adding features) for a particular function. In such
cases, the output needs to be manually/visually checked as part of the PR
review process and then a new baseline set for comparison.

Set a new baseline by running:

pytest nilearn/plotting/tests/test_img_plotting/test_baseline_comparisons.py \
--mpl-generate-path=nilearn/plotting/tests/test_img_plotting/baseline

"""

import pytest

from nilearn.datasets import load_sample_motor_activation_image
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

PLOTTING_FUNCS_3D = {
    plot_img,
    plot_anat,
    plot_stat_map,
    plot_roi,
    plot_epi,
    plot_glass_brain,
}

PLOTTING_FUNCS_4D = {plot_prob_atlas, plot_carpet}


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("black_bg", [True, False])
def test_plot_functions_black_bg(plot_func, img_3d_mni, black_bg):
    """Test parameter for black background."""
    return plot_func(img_3d_mni, black_bg=black_bg)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("title", [None, "foo"])
def test_plot_functions_title(plot_func, img_3d_mni, title):
    """Test parameter for title."""
    return plot_func(img_3d_mni, title=title)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("annotate", [True, False])
def test_plot_functions_annotate(plot_func, img_3d_mni, annotate):
    """Test parameter for title."""
    return plot_func(img_3d_mni, annotate=annotate)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize(
    "display_mode", ["x", "y", "z", "yx", "xz", "yz", "ortho"]
)
def test_plot_functions_display_mode(plot_func, display_mode):
    """Test parameter for title."""
    return plot_func(
        load_sample_motor_activation_image(), display_mode=display_mode
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_functions_colorbar(plot_func, colorbar):
    """Test parameter for title."""
    return plot_func(load_sample_motor_activation_image(), colorbar=colorbar)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("vmin", [None, -1, 1])
@pytest.mark.parametrize("vmax", [None, 2, 3])
def test_plot_functions_3d_default_params(plot_func, vmin, vmax):
    """Smoke tests for 3D plotting functions with default parameters."""
    return plot_func(
        load_sample_motor_activation_image(), vmin=vmin, vmax=vmax
    )


@pytest.mark.parametrize("plotting_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("radiological", [True, False])
def test_plotting_functions_radiological_view(
    img_3d_mni, plotting_func, radiological
):
    """Smoke test for radiological view."""
    result = plotting_func(img_3d_mni, radiological=radiological)
    assert result.axes.get("y").radiological is radiological
    return result


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("cbar_tick_format", ["%f", "%i"])
def test_cbar_tick_format(plot_func, img_3d_mni, cbar_tick_format):
    """Test different colorbar tick format with 3D plotting functions."""
    return plot_func(
        img_3d_mni,
        colorbar=True,
        cbar_tick_format=cbar_tick_format,
    )


@pytest.mark.mpl_image_compare
def test_plot_carpet_default_params(img_4d_mni, img_3d_ones_mni):
    """Smoke-test for 4D plot_carpet with default arguments."""
    return plot_carpet(img_4d_mni, mask_img=img_3d_ones_mni)


@pytest.mark.mpl_image_compare
def test_plot_prob_atlas_default_params(img_3d_mni, img_4d_mni):
    """Smoke-test for plot_prob_atlas with default arguments."""
    return plot_prob_atlas(img_4d_mni, bg_img=img_3d_mni)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
def test_plot_anat_mni(anat_img):
    """Tests for plot_anat with MNI template."""
    return plot_anat(anat_img=anat_img)
