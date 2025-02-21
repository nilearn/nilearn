"""
Test if public plotting functions' output has changed.

See the  maintenance page of our documentation for more information
https://nilearn.github.io/dev/maintenance.html#generating-new-baseline-figures-for-plotting-tests
"""

import numpy as np
import pytest

from nilearn.datasets import (
    load_fsaverage_data,
    load_sample_motor_activation_image,
)
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.tests._testing import modulated_event_paradigm
from nilearn.plotting import (
    plot_anat,
    plot_carpet,
    plot_epi,
    plot_glass_brain,
    plot_img,
    plot_prob_atlas,
    plot_roi,
    plot_stat_map,
    plot_surf,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.plotting.matrix_plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
    plot_design_matrix_correlation,
    plot_event,
    plot_matrix,
)

PLOTTING_FUNCS_3D = {
    plot_img,
    plot_anat,
    plot_stat_map,
    plot_roi,
    plot_epi,
    plot_glass_brain,
}

PLOTTING_FUNCS_4D = {plot_prob_atlas, plot_carpet}

SURFACE_FUNCS = {
    plot_surf,
    plot_surf_stat_map,
    plot_surf_roi,
}


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
    """Test parameter for annotate."""
    return plot_func(img_3d_mni, annotate=annotate)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize(
    "display_mode", ["x", "y", "z", "yx", "xz", "yz", "ortho"]
)
def test_plot_functions_display_mode(plot_func, display_mode):
    """Test parameter for display_mode."""
    return plot_func(
        load_sample_motor_activation_image(), display_mode=display_mode
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("colorbar", [True, False])
@pytest.mark.parametrize("cbar_tick_format", ["%f", "%i"])
def test_plot_functions_colorbar(plot_func, colorbar, cbar_tick_format):
    """Test parameter for colorbar."""
    return plot_func(
        load_sample_motor_activation_image(),
        colorbar=colorbar,
        cbar_tick_format=cbar_tick_format,
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("vmin", [None, -1, 1])
@pytest.mark.parametrize("vmax", [None, 2, 3])
def test_plot_functions_3d_default_params(plot_func, vmin, vmax):
    """Test 3D plotting functions with vmin, vmax."""
    return plot_func(
        load_sample_motor_activation_image(), vmin=vmin, vmax=vmax
    )


@pytest.mark.parametrize("plotting_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("radiological", [True, False])
def test_plotting_functions_radiological_view(
    img_3d_mni, plotting_func, radiological
):
    """Test for radiological view."""
    result = plotting_func(img_3d_mni, radiological=radiological)
    assert result.axes.get("y").radiological is radiological
    return result


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


# ---------------------- surface plotting -------------------------------


@pytest.mark.mpl_image_compare(tolerance=5)
@pytest.mark.parametrize("plot_func", SURFACE_FUNCS)
@pytest.mark.parametrize(
    "view",
    [
        "anterior",
        "posterior",
        "dorsal",
        "ventral",
    ],
)
@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_plot_surf_surface(plot_func, view, hemi):
    """Test surface plotting functions with views and hemispheres."""
    surf_img = load_fsaverage_data()
    return plot_func(
        surf_img.mesh,
        surf_img,
        engine="matplotlib",
        view=view,
        hemi=hemi,
        title=f"{view=}, {hemi=}",
    )


@pytest.mark.mpl_image_compare(tolerance=5)
@pytest.mark.parametrize("plot_func", SURFACE_FUNCS)
@pytest.mark.parametrize("colorbar", [True, False])
@pytest.mark.parametrize("cbar_tick_format", ["auto", "%f"])
def test_plot_surf_surface_colorbar(plot_func, colorbar, cbar_tick_format):
    """Test surface plotting functions with colorbars."""
    surf_img = load_fsaverage_data()
    return plot_func(
        surf_img.mesh,
        surf_img,
        engine="matplotlib",
        colorbar=colorbar,
        cbar_tick_format=cbar_tick_format,
    )


# ---------------------- design matrix plotting -------------------------------


@pytest.mark.mpl_image_compare
def test_plot_event_duration_0():
    """Test plot event with events of duration 0."""
    return plot_event(modulated_event_paradigm())


@pytest.fixture
def matrix_to_plot(rng):
    return rng.random((50, 50)) * 10 - 5


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_matrix_colorbar(matrix_to_plot, colorbar):
    """Test plotting matrix with or without colorbar."""
    ax = plot_matrix(matrix_to_plot, colorbar=colorbar)

    return ax.get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "labels", [[], np.array([str(i) for i in range(50)]), None]
)
def test_plot_matrix_labels(matrix_to_plot, labels):
    """Test plotting labels on matrix."""
    ax = plot_matrix(matrix_to_plot, labels=labels)

    return ax.get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("tri", ["full", "lower", "diag"])
def test_plot_matrix_grid(matrix_to_plot, tri):
    """Test plotting full matrix or upper / lower half of it."""
    ax = plot_matrix(matrix_to_plot, tri=tri)

    return ax.get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("tri", ["full", "diag"])
def test_plot_design_matrix_correlation(tri):
    """Test plotting full matrix or lower half of it."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, events=modulated_event_paradigm()
    )

    ax = plot_design_matrix_correlation(
        dmtx,
        tri=tri,
    )

    return ax.get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_design_matrix_correlation_colorbar(colorbar):
    """Test plot_design_matrix_correlation with / without colorbar."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, events=modulated_event_paradigm()
    )

    ax = plot_design_matrix_correlation(dmtx, colorbar=colorbar)

    return ax.get_figure()


@pytest.mark.mpl_image_compare
def test_plot_design_matrix():
    """Test plot_design_matrix."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )

    ax = plot_design_matrix(dmtx)

    return ax.get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "contrast",
    [np.array([[1, 0, 0, 1], [0, -2, 1, 0]]), np.array([1, 0, 0, -1])],
)
def test_plot_contrast_matrix(contrast):
    """Test plot_contrast_matrix with T and F contrast."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )

    ax = plot_contrast_matrix(contrast, dmtx)

    return ax.get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_contrast_matrix_colorbar(colorbar):
    """Test plot_contrast_matrix colorbar."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    contrast = np.array([[1, 0, 0, 1], [0, -2, 1, 0]])

    ax = plot_contrast_matrix(contrast, dmtx, colorbar=colorbar)

    return ax.get_figure()
