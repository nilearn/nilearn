"""
Test if public plotting functions' output has changed.

See the  maintenance page of our documentation for more information
https://nilearn.github.io/dev/maintenance.html#generating-new-baseline-figures-for-plotting-tests
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from nilearn.datasets import (
    load_fsaverage_data,
    load_mni152_template,
    load_sample_motor_activation_image,
)
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.tests._testing import modulated_event_paradigm
from nilearn.plotting import (
    plot_anat,
    plot_bland_altman,
    plot_carpet,
    plot_connectome,
    plot_contrast_matrix,
    plot_design_matrix,
    plot_design_matrix_correlation,
    plot_epi,
    plot_event,
    plot_glass_brain,
    plot_img,
    plot_img_comparison,
    plot_matrix,
    plot_prob_atlas,
    plot_roi,
    plot_stat_map,
    plot_surf,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.image.utils import MNI152TEMPLATE

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
def test_plot_functions_black_bg(plot_func, img_3d_mni):
    """Test parameter for black background.

    black_bg=False being the default it should be covered by other tests.
    """
    return plot_func(img_3d_mni, black_bg=True)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_title(plot_func, img_3d_mni):
    """Test parameter for title.

    title=None being the default it should be covered by other tests.
    """
    return plot_func(img_3d_mni, title="foo")


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_annotate(plot_func, img_3d_mni):
    """Test parameter for annotate=False.

    annotate=True being the default it should be covered by other tests.
    """
    return plot_func(img_3d_mni, annotate=False)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "display_mode", ["x", "y", "z", "yx", "xz", "yz", "ortho"]
)
def test_plot_stat_map_display_mode(display_mode):
    """Test parameter for display_mode.

    Only test one function to speed up testing.
    """
    return plot_stat_map(
        load_sample_motor_activation_image(), display_mode=display_mode
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_no_colorbar(plot_func, img_3d_mni):
    """Test no colorbar.

    colorbar=True being the default it should be covered by other tests.
    """
    return plot_func(
        img_3d_mni,
        colorbar=False,
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_colorbar_ticks(plot_func, img_3d_mni):
    """Test parameter for colorbar."""
    return plot_func(
        img_3d_mni,
        cbar_tick_format="%f",
    )


@pytest.mark.mpl_image_compare(tolerance=5)
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("vmin", [-1, 1])
def test_plot_functions_vmin(plot_func, vmin):
    """Test 3D plotting functions with vmin."""
    return plot_func(load_sample_motor_activation_image(), vmin=vmin)


@pytest.mark.mpl_image_compare(tolerance=5)
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
@pytest.mark.parametrize("vmax", [2, 3])
def test_plot_functions_vmax(plot_func, vmax):
    """Test 3D plotting functions with vmax."""
    return plot_func(load_sample_motor_activation_image(), vmax=vmax)


@pytest.mark.mpl_image_compare(tolerance=5)
@pytest.mark.parametrize("plotting_func", PLOTTING_FUNCS_3D)
def test_plotting_functions_radiological_view(plotting_func):
    """Test for radiological view.

    radiological=False being the default it should be covered by other tests.
    """
    radiological = True
    result = plotting_func(
        load_sample_motor_activation_image(), radiological=radiological
    )
    assert result.axes.get("y").radiological is radiological
    return result


@pytest.mark.mpl_image_compare
def test_plot_carpet_default_params(img_4d_mni, img_3d_ones_mni):
    """Smoke-test for 4D plot_carpet with default arguments."""
    return plot_carpet(img_4d_mni, mask_img=img_3d_ones_mni)


@pytest.mark.timeout(0)
@pytest.mark.mpl_image_compare
def test_plot_prob_atlas_default_params(img_3d_mni, img_4d_mni):
    """Smoke-test for plot_prob_atlas with default arguments."""
    # TODO (nilearn >= 0.13.0)
    # using only 2 regions to speed up the test
    # maps = generate_maps(shape_3d_default, n_regions=2, affine=affine_mni)
    return plot_prob_atlas(img_4d_mni, bg_img=img_3d_mni)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
def test_plot_anat_mni(anat_img):
    """Tests for plot_anat with MNI template."""
    return plot_anat(anat_img=anat_img)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_connectome_colorbar(colorbar, adjacency, node_coords):
    """Smoke test for plot_connectome with default parameters \
       and with and without the colorbar.
    """
    return plot_connectome(adjacency, node_coords, colorbar=colorbar)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "node_color",
    [["green", "blue", "k", "cyan"], np.array(["red"]), ["red"], "green"],
)
def test_plot_connectome_node_colors(
    node_color, node_coords, adjacency, params_plot_connectome
):
    """Smoke test for plot_connectome with different values for node_color."""
    return plot_connectome(
        adjacency,
        node_coords,
        node_color=node_color,
        **params_plot_connectome,
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("alpha", [0.0, 0.7, 1.0])
def test_plot_connectome_alpha(alpha, adjacency, node_coords):
    """Smoke test for plot_connectome with various alpha values."""
    return plot_connectome(adjacency, node_coords, alpha=alpha)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "display_mode",
    [
        "ortho",
        "x",
        "y",
        "z",
        "xz",
        "yx",
        "yz",
        "l",
        "r",
        "lr",
        "lzr",
        "lyr",
        "lzry",
        "lyrz",
    ],
)
def test_plot_connectome_display_mode(
    display_mode, node_coords, adjacency, params_plot_connectome
):
    """Smoke test for plot_connectome with different values \
       for display_mode.
    """
    return plot_connectome(
        adjacency,
        node_coords,
        display_mode=display_mode,
        **params_plot_connectome,
    )


@pytest.mark.mpl_image_compare
def test_plot_connectome_node_and_edge_kwargs(adjacency, node_coords):
    """Smoke test for plot_connectome with node_kwargs, edge_kwargs, \
       and edge_cmap arguments.
    """
    return plot_connectome(
        adjacency,
        node_coords,
        edge_threshold="70%",
        node_size=[10, 20, 30, 40],
        node_color=np.zeros((4, 3)),
        edge_cmap="RdBu",
        colorbar=True,
        node_kwargs={"marker": "v"},
        edge_kwargs={"linewidth": 4},
    )


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


@pytest.mark.mpl_image_compare
def test_plot_event_x_lim(rng):
    """Test that x_lim is set after end of last event.

    Regression test for https://github.com/nilearn/nilearn/issues/4907
    """
    trial_types = ["foo", "bar", "baz"]

    n_runs = 3

    events = [
        pd.DataFrame(
            {
                "trial_type": trial_types,
                "onset": rng.random((3,)) * 5,
                "duration": rng.uniform(size=(3,)) * 2 + 1,
            }
        )
        for _ in range(n_runs)
    ]

    return plot_event(events)


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


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("fn", [plot_stat_map, plot_img, plot_glass_brain])
def test_plot_with_transparency(fn):
    """Test transparency parameter to determine alpha layer."""
    return fn(
        load_sample_motor_activation_image(), transparency=0.5, cmap="cold_hot"
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("fn", [plot_stat_map, plot_img, plot_glass_brain])
@pytest.mark.parametrize("transparency_range", [None, [0, 2], [2, 4]])
def test_plot_with_transparency_range(fn, transparency_range):
    """Test transparency range parameter to determine alpha layer."""
    return fn(
        load_sample_motor_activation_image(),
        transparency=load_sample_motor_activation_image(),
        transparency_range=transparency_range,
        cmap="cold_hot",
    )


IMG_COMPARISON_FUNCS = {plot_img_comparison, plot_bland_altman}


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", IMG_COMPARISON_FUNCS)
def test_img_comparison_default(
    plot_func,
):
    """Test img comparing plotting functions with defaults."""
    plot_func(load_mni152_template(), load_sample_motor_activation_image())
    # need to use gcf as plot_img_comparison does not return a figure
    return plt.gcf()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", IMG_COMPARISON_FUNCS)
@pytest.mark.parametrize("colorbar", [True, False])
def test_img_comparison_colorbar(
    plot_func,
    colorbar,
):
    """Test img comparing plotting functions with colorbar."""
    plot_func(
        load_mni152_template(),
        load_sample_motor_activation_image(),
        colorbar=colorbar,
    )
    # need to use gcf as plot_img_comparison does not return a figure
    return plt.gcf()
