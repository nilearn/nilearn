"""Test nilearn.plotting.surface.surf_plotting functions."""

# ruff: noqa: ARG001

import re
import tempfile

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from nilearn._utils.exceptions import MeshDimensionError
from nilearn._utils.helpers import (
    is_kaleido_installed,
    is_plotly_installed,
)
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_img_on_surf,
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)


@pytest.fixture
def surf_roi_data(rng, in_memory_mesh):
    roi_map = np.zeros((in_memory_mesh.n_vertices, 1))
    roi_idx = rng.integers(0, in_memory_mesh.n_vertices, size=10)
    roi_map[roi_idx] = 1
    return roi_map


@pytest.mark.parametrize(
    "fn",
    [
        plot_surf,
        plot_surf_stat_map,
        plot_surf_contours,
        plot_surf_roi,
    ],
)
def test_check_surface_plotting_inputs_error_mesh_and_data_none(fn):
    """Fail if no mesh or data is passed."""
    with pytest.raises(TypeError, match="cannot both be None"):
        fn(None, None)


@pytest.mark.parametrize(
    "fn",
    [
        plot_surf,
        plot_surf_stat_map,
        plot_img_on_surf,
        plot_surf_roi,
    ],
)
def test_check_surface_plotting_inputs_error_negative_threshold(
    fn, in_memory_mesh
):
    """Fail if negative threshold is passed."""
    with pytest.raises(ValueError, match="Threshold should be a"):
        fn(in_memory_mesh, threshold=-1)


@pytest.mark.parametrize(
    "fn",
    [
        plot_surf,
        plot_surf_contours,
        plot_surf_stat_map,
        plot_surf_roi,
    ],
)
@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_check_surface_plotting_inputs_single_hemi_data(
    in_memory_mesh, fn, hemi
):
    """Smoke test when single hemi data is passed."""
    parcellation = np.zeros((in_memory_mesh.n_vertices,))
    parcellation[in_memory_mesh.faces[3]] = 1
    fn(in_memory_mesh, parcellation, hemi=hemi)


def test_check_surface_plotting_inputs_errors():
    """Fail if mesh is None and data is not a SurfaceImage."""
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf(surf_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_stat_map(stat_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_contours(roi_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_roi(roi_map=1, surf_mesh=None)


def test_plot_surf_engine_error(in_memory_mesh):
    """Test error if unknown engine is specified."""
    with pytest.raises(ValueError, match="Unknown plotting engine"):
        plot_surf(in_memory_mesh, engine="foo")


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="This test is run only if plotly is not installed.",
)
def test_plot_surf_engine_error_plotly_not_installed(in_memory_mesh):
    """Test error if plotly is not installed but specified as engine."""
    with pytest.raises(ImportError, match="Using engine"):
        plot_surf(in_memory_mesh, engine="plotly")


@pytest.mark.timeout(0)
def test_plot_surf(plt, engine, tmp_path, in_memory_mesh, bg_map):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf function with
    available engine backends.
    """
    # to avoid extra warnings
    alpha = None
    cbar_vmin = None
    cbar_vmax = None
    if engine == "matplotlib":
        alpha = 0.5
        cbar_vmin = 0
        cbar_vmax = 150

    # Plot mesh only
    plot_surf(in_memory_mesh, engine=engine)

    # Plot mesh with background
    plot_surf(in_memory_mesh, bg_map=bg_map, engine=engine)
    plot_surf(
        in_memory_mesh,
        bg_map=bg_map,
        alpha=alpha,
        output_file=tmp_path / "tmp.png",
        engine=engine,
    )

    # Plot with colorbar
    plot_surf(in_memory_mesh, bg_map=bg_map, colorbar=True, engine=engine)
    plot_surf(
        in_memory_mesh,
        bg_map=bg_map,
        colorbar=True,
        cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax,
        cbar_tick_format="%i",
        engine=engine,
    )


@pytest.mark.parametrize("view", ["anterior", "posterior"])
@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_plot_surf_hemi_views(plt, engine, in_memory_mesh, hemi, view, bg_map):
    """Check plotting view and hemispheres."""
    plot_surf(
        in_memory_mesh, bg_map=bg_map, hemi=hemi, view=view, engine=engine
    )


@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_plot_surf_swap_hemi(plt, engine, surf_img_1d, hemi, flip_surf_img):
    """Check error is raised if background image is incompatible."""
    with pytest.raises(
        MeshDimensionError,
        match="Number of vertices do not match for between meshes.",
    ):
        plot_surf(
            surf_map=surf_img_1d,
            bg_map=flip_surf_img(surf_img_1d),
            hemi=hemi,
            surf_mesh=None,
            engine=engine,
        )


def test_plot_surf_error(plt, engine, rng, in_memory_mesh):
    """Check error if invalid parameters values are specified to
    nilearn.plotting.surface.surf_plotting.plot_surf.
    """
    # Wrong inputs for view or hemi
    with pytest.raises(ValueError, match="Invalid view definition"):
        plot_surf(in_memory_mesh, view="middle", engine=engine)
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        plot_surf(in_memory_mesh, hemi="lft", engine=engine)

    # Wrong size of background image
    with pytest.raises(
        ValueError, match="bg_map does not have the same number of vertices"
    ):
        plot_surf(
            in_memory_mesh,
            bg_map=rng.standard_normal(size=in_memory_mesh.n_vertices - 1),
            engine=engine,
        )

    # Wrong size of surface data
    with pytest.raises(
        ValueError, match="surf_map does not have the same number of vertices"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=rng.standard_normal(size=in_memory_mesh.n_vertices + 1),
            engine=engine,
        )

    with pytest.raises(
        ValueError, match="'surf_map' can only have one dimension"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=rng.standard_normal(size=(in_memory_mesh.n_vertices, 2)),
            engine=engine,
        )


@pytest.mark.parametrize(
    "kwargs", [{"symmetric_cmap": True}, {"title_font_size": 18}]
)
def test_plot_surf_warnings_not_implemented_in_matplotlib(
    matplotlib_pyplot, kwargs, in_memory_mesh, bg_map
):
    """Test if nilearn.plotting.surface.surf_plotting.plot_surf raises error
    when a parameter that is not supported by matplotlib is specified with a
    value other than None.
    """
    with pytest.warns(
        UserWarning, match="is not implemented for the matplotlib engine"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            engine="matplotlib",
            **kwargs,
        )


@pytest.mark.parametrize("kwargs", [{"avg_method": "mean"}, {"alpha": "auto"}])
def test_plot_surf_warnings_not_implemented_in_plotly(
    plotly, kwargs, in_memory_mesh, bg_map
):
    """Test if nilearn.plotting.surface.surf_plotting.plot_surf raises error
    when a parameter that is not supported by plotly is specified with a
    value other than None.
    """
    with pytest.warns(
        UserWarning, match="is not implemented for the plotly engine"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            engine="plotly",
            **kwargs,
        )


@pytest.mark.skipif(
    is_kaleido_installed(),
    reason=("This test only runs if Plotly is installed, but not kaleido."),
)
def test_plot_surf_error_when_kaleido_missing(
    plotly, tmp_path, in_memory_mesh, bg_map
):
    """Test if nilearn.plotting.surface.surf_plotting.plot_surf raises
    ImportError when engine is 'plotly' and kaleido is not installed.
    """
    with pytest.raises(ImportError, match="Saving figures"):
        # Plot with non None output file
        plot_surf(
            in_memory_mesh,
            bg_map=bg_map,
            engine="plotly",
            output_file=tmp_path / "tmp.png",
        )


def test_plot_surf_avg_method(matplotlib_pyplot, in_memory_mesh, bg_map):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf for valid
    values of avg_method.
    """
    # Plot with avg_method
    # Test all built-in methods and check
    faces = in_memory_mesh.faces
    ENGINE = "matplotlib"

    for method in ["mean", "median", "min", "max"]:
        display = plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=method,
            engine=ENGINE,
        )
        if method == "mean":
            agg_faces = np.mean(bg_map[faces], axis=1)
        elif method == "median":
            agg_faces = np.median(bg_map[faces], axis=1)
        elif method == "min":
            agg_faces = np.min(bg_map[faces], axis=1)
        elif method == "max":
            agg_faces = np.max(bg_map[faces], axis=1)
        vmin = np.min(agg_faces)
        vmax = np.max(agg_faces)
        agg_faces -= vmin
        agg_faces /= vmax - vmin
        cmap = matplotlib_pyplot.get_cmap(
            matplotlib_pyplot.rcParamsDefault["image.cmap"]
        )
        assert_array_equal(
            cmap(agg_faces),
            display._axstack.as_list()[0].collections[0]._facecolors,
        )

    #  Try custom avg_method
    def custom_avg_function(vertices):
        return vertices[0] * vertices[1] * vertices[2]

    plot_surf(
        in_memory_mesh,
        surf_map=bg_map,
        avg_method=custom_avg_function,
        engine=ENGINE,
    )


def test_plot_surf_avg_method_errors(
    matplotlib_pyplot, in_memory_mesh, bg_map
):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf for invalid
    values of avg_method.
    """
    ENGINE = "matplotlib"
    with pytest.raises(
        ValueError,
        match=(
            "Array computed with the custom "
            "function from avg_method does "
            "not have the correct shape"
        ),
    ):

        def custom_avg_function(vertices):
            return [vertices[0] * vertices[1], vertices[2]]

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=custom_avg_function,
            engine=ENGINE,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "avg_method should be either "
            "['mean', 'median', 'max', 'min'] "
            "or a custom function"
        ),
    ):
        custom_avg_function = {}

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=custom_avg_function,
            engine=ENGINE,
        )

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method="foo",
            engine=ENGINE,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Array computed with the custom function "
            "from avg_method should be an array of "
            "numbers (int or float)"
        ),
    ):

        def custom_avg_function(vertices):
            return "string"

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=custom_avg_function,
            engine=ENGINE,
        )


def test_plot_surf_with_title(matplotlib_pyplot, in_memory_mesh, bg_map):
    """Test if figure title is set correctly in
    nilearn.plotting.surface.surf_plotting.plot_surf.
    """
    display = plot_surf(
        in_memory_mesh, bg_map=bg_map, title="Test title", engine="matplotlib"
    )

    assert len(display.axes) == 1
    assert display.axes[0].title._text == "Test title"


def test_surface_plotting_axes_error(matplotlib_pyplot, surf_img_1d):
    """Test error msg for invalid axes."""
    figure, axes = matplotlib_pyplot.subplots()
    with pytest.raises(AttributeError, match="the projection must be '3d'"):
        plot_surf_stat_map(stat_map=surf_img_1d, axes=axes)


def test_plot_surf_contours(
    matplotlib_pyplot, in_memory_mesh, parcellation, surf_mask_1d
):
    """Test nilearn.plotting.surface.plot_surf_contours for valid input
    values.
    """
    plot_surf_contours(in_memory_mesh, parcellation)
    plot_surf_contours(in_memory_mesh, parcellation, levels=[1, 2])
    plot_surf_contours(
        in_memory_mesh, parcellation, levels=[1, 2], cmap="gist_ncar"
    )


def test_plot_surf_contour_roi_map_as_surface_image(
    matplotlib_pyplot, surf_mesh, surf_mask_1d
):
    """Check that mesh can be PolyMesh and roi_map can be a SurfaceImage."""
    plot_surf_contours(surf_mesh, roi_map=surf_mask_1d, hemi="both")


def test_plot_surf_contours_legend(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    """Test nilearn.plotting.surface.plot_surf_contours creates figure legend
    when `legend=True`.
    """
    fig = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        legend=True,
    )
    assert fig.legends is not None


def test_plot_surf_contours_colors(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    """Test nilearn.plotting.surface.plot_surf_contours for different inputs as
    `colors`.
    """
    plot_surf_contours(
        in_memory_mesh, parcellation, levels=[1, 2], colors=["r", "g"]
    )
    plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        labels=["1", "2"],
        colors=["r", "g"],
    )
    plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        colors=[[0, 0, 0, 1], [1, 1, 1, 1]],
    )


def test_plot_surf_contours_axis_title(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    """Test nilearn.plotting.surface.plot_surf_contours for axis title."""
    fig = plot_surf(in_memory_mesh)
    plot_surf_contours(in_memory_mesh, parcellation, figure=fig)
    display = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        labels=["1", "2"],
        colors=["r", "g"],
        legend=True,
        title="title",
        figure=fig,
    )
    # Non-regression assertion: we switched from _suptitle to axis title
    assert display._suptitle is None
    assert display.axes[0].get_title() == "title"

    fig = plot_surf(in_memory_mesh, title="title 2")
    display = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        labels=["1", "2"],
        colors=["r", "g"],
        legend=True,
        figure=fig,
    )
    # Non-regression assertion: we switched from _suptitle to axis title
    assert display._suptitle is None
    assert display.axes[0].get_title() == "title 2"
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_contours(
            in_memory_mesh, parcellation, output_file=tmp_file.name
        )


def test_plot_surf_contours_fig_axes(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf_contours with
    matplotlib figure and axes.
    """
    fig, axes = matplotlib_pyplot.subplots(
        1, 1, subplot_kw={"projection": "3d"}
    )
    plot_surf_contours(in_memory_mesh, parcellation, axes=axes)
    plot_surf_contours(in_memory_mesh, parcellation, figure=fig)


def test_plot_surf_contours_error(
    matplotlib_pyplot, rng, in_memory_mesh, parcellation
):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf_contours for
    invalid parameters.
    """
    # we need an invalid parcellation for testing
    invalid_parcellation = rng.uniform(size=(in_memory_mesh.n_vertices))
    with pytest.raises(
        ValueError, match="Vertices in parcellation do not form region."
    ):
        plot_surf_contours(in_memory_mesh, invalid_parcellation)

    _, axes = matplotlib_pyplot.subplots(1, 1)
    with pytest.raises(ValueError, match="Axes must be 3D."):
        plot_surf_contours(in_memory_mesh, parcellation, axes=axes)

    msg = "All elements of colors .* matplotlib .* RGBA"
    with pytest.raises(ValueError, match=msg):
        plot_surf_contours(
            in_memory_mesh, parcellation, levels=[1, 2], colors=[[1, 2], 3]
        )

    msg = "Levels, labels, and colors argument .* same length or None."
    with pytest.raises(ValueError, match=msg):
        plot_surf_contours(
            in_memory_mesh,
            parcellation,
            levels=[1, 2],
            colors=["r"],
            labels=["1", "2"],
        )


def test_plot_surf_contours_errors_with_plotly_figure(plotly, in_memory_mesh):
    """Test that plot_surf_contours raises error when given plotly obj."""
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(ValueError):
        plot_surf_contours(in_memory_mesh, np.ones((10,)), figure=figure)
    with pytest.raises(ValueError):
        plot_surf_contours(in_memory_mesh, np.ones((10,)), axes=figure)


def test_plot_surf_stat_map(plt, engine, in_memory_mesh, bg_map):
    """Smoke test when stat_map is specified to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map together with
    mesh.
    """
    alpha = 1 if engine == "matplotlib" else None

    plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, engine=engine)
    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, alpha=alpha, engine=engine
    )


def test_plot_surf_stat_map_with_background(
    plt, engine, in_memory_mesh, bg_map
):
    """Smoke test when background map is specified also as stat_map to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, bg_map=bg_map, engine=engine
    )
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        bg_map=bg_map,
        bg_on_data=True,
        darkness=0.5,
        engine=engine,
    )


def test_plot_surf_stat_map_with_title(plt, engine, in_memory_mesh, bg_map):
    """Test if nilearn.plotting.surface.surf_plotting.plot_surf_stat_map adds
    title when specified.
    """
    display = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, title="Stat map title"
    )
    assert display.axes[0].title._text == "Stat map title"


def test_plot_surf_stat_map_with_threshold(
    plt, engine, in_memory_mesh, bg_map
):
    """Smoke test when threshold is specified to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        threshold=0.3,
        engine=engine,
    )


def test_plot_surf_stat_map_vmax(plt, engine, in_memory_mesh, bg_map):
    """Smoke test when vmax is specified to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, vmax=5, engine=engine)


@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_surf_stat_map_error_vmax_equal_vmin(
    plt, engine, in_memory_mesh, bg_map, colorbar
):
    """Smoke test when vmax == vmin.

    Make sure matplotlib does not raise error.
    """
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        vmin=5,
        vmax=5,
        engine=engine,
        colorbar=colorbar,
    )


def test_plot_surf_stat_map_colormap(plt, engine, in_memory_mesh, bg_map):
    """Smoke test when colormap is specified to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, cmap="cubehelix", engine=engine
    )


def test_plot_surf_stat_map_error(in_memory_mesh, bg_map):
    """Test if nilearn.plotting.surface.surf_plotting.plot_surf_stat_map
    raises error with wrong size of stat map data.
    """
    # Wrong size of stat map data
    with pytest.raises(
        ValueError, match="surf_map does not have the same number of vertices"
    ):
        plot_surf_stat_map(
            in_memory_mesh, stat_map=np.hstack((bg_map, bg_map))
        )

    with pytest.raises(
        ValueError, match="'surf_map' can only have one dimension"
    ):
        plot_surf_stat_map(
            in_memory_mesh, stat_map=np.vstack((bg_map, bg_map)).T
        )


def test_plot_surf_stat_map_colorbar_tick(plotly, in_memory_mesh, bg_map):
    """Smoke test when colorbar tick format with plotly engine is specified to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        cbar_tick_format="%.2g",
        engine="plotly",
    )


@pytest.mark.parametrize("symmetric_cmap", [True, False, None])
def test_plot_surf_stat_map_symmetric_cmap_plotly(
    plotly, in_memory_mesh, bg_map, symmetric_cmap
):
    """Smoke test when symmetric_cmap with plotly engine is specified to
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        symmetric_cmap=symmetric_cmap,
        engine="plotly",
    )


def test_plot_surf_stat_map_symmetric_cmap_matplotlib(
    matplotlib_pyplot, in_memory_mesh, bg_map
):
    """Smoke test when symmetric_cmap is specified as None for matplotlib
    engine to nilearn.plotting.surface.surf_plotting.plot_surf_stat_map.
    """
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        symmetric_cmap=None,
        engine="matplotlib",
    )


@pytest.mark.parametrize("symmetric_cmap", [True, False])
def test_plot_surf_stat_map_symmetric_cmap_matplotlib_error(
    matplotlib_pyplot, in_memory_mesh, bg_map, symmetric_cmap
):
    """Test if
    nilearn.plotting.surface.surf_plotting.plot_surf_stat_map raises error when
    True or False is specified as symmetric_cmap for matplotlib engine.
    """
    with pytest.warns(UserWarning, match="'symmetric_cmap' is not implement"):
        plot_surf_stat_map(
            in_memory_mesh,
            stat_map=bg_map,
            symmetric_cmap=symmetric_cmap,
            engine="matplotlib",
        )


def test_plot_surf_stat_map_matplotlib_specific(
    matplotlib_pyplot, in_memory_mesh, bg_map
):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf_stat_map for
    matplotlib engine specific parameters.
    """
    # Plot to axes
    axes = matplotlib_pyplot.subplots(
        ncols=2, subplot_kw={"projection": "3d"}
    )[1]
    for ax in axes.flatten():
        plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, axes=ax)
    axes = matplotlib_pyplot.subplots(
        ncols=2, subplot_kw={"projection": "3d"}
    )[1]
    for ax in axes.flatten():
        plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, axes=ax)

    fig = plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, colorbar=False)

    assert len(fig.axes) == 1

    # symmetric_cbar
    fig = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, symmetric_cbar=True
    )
    fig.canvas.draw()

    assert len(fig.axes) == 2

    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()

    assert float(first) == -float(last)

    # no symmetric_cbar
    fig = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, symmetric_cbar=False
    )
    fig.canvas.draw()

    assert len(fig.axes) == 2

    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()

    assert float(first) != -float(last)

    # Test handling of nan values in texture data
    # Add nan values in the texture
    bg_map[2] = np.nan
    # Plot the surface stat map
    fig = plot_surf_stat_map(in_memory_mesh, stat_map=bg_map)
    # Check that the resulting plot facecolors contain no transparent faces
    # (last column equals zero) even though the texture contains nan values
    tmp = fig._axstack.as_list()[0].collections[0]

    assert (
        in_memory_mesh.faces.shape[0] == ((tmp._facecolors[:, 3]) != 0).sum()
    )


@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_surf_roi(plt, engine, surface_image_roi, colorbar):
    """Smoke test for nilearn.plotting.surface.surf_plotting.plot_surf_roi
    for colorbar parameter.
    """
    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        colorbar=colorbar,
        engine=engine,
    )


def test_plot_surf_roi_cmap_as_lookup_table(surface_image_roi):
    """Test colormap passed as BIDS lookup table."""
    lut = pd.DataFrame(
        {"index": [0, 1], "name": ["foo", "bar"], "color": ["#000", "#fff"]}
    )
    plot_surf_roi(surface_image_roi.mesh, roi_map=surface_image_roi, cmap=lut)

    lut = pd.DataFrame({"index": [0, 1], "name": ["foo", "bar"]})
    with pytest.warns(
        UserWarning, match="No 'color' column found in the look-up table."
    ):
        plot_surf_roi(
            surface_image_roi.mesh, roi_map=surface_image_roi, cmap=lut
        )


def test_plot_surf_roi_error(engine, rng, in_memory_mesh, surf_roi_data):
    """Test for nilearn.plotting.surface.surf_plotting.plot_surf_roi
    for invalid parameter values.
    """
    # too many axes
    with pytest.raises(
        ValueError, match="roi_map can only have one dimension but has"
    ):
        plot_surf_roi(
            in_memory_mesh,
            roi_map=np.array([surf_roi_data, surf_roi_data]),
            engine=engine,
        )

    # wrong number of vertices
    roi_idx = rng.integers(0, in_memory_mesh.n_vertices, size=5)
    with pytest.raises(
        ValueError, match="roi_map does not have the same number of vertices"
    ):
        plot_surf_roi(in_memory_mesh, roi_map=roi_idx, engine=engine)

    # negative value in roi map
    surf_roi_data[0] = -1
    with pytest.warns(
        DeprecationWarning,
        match="Negative values in roi_map will no longer be allowed",
    ):
        plot_surf_roi(in_memory_mesh, roi_map=surf_roi_data, engine=engine)

    # float value in roi map
    surf_roi_data[0] = 1.2
    with pytest.warns(
        DeprecationWarning,
        match="Non-integer values in roi_map will no longer be allowed",
    ):
        plot_surf_roi(in_memory_mesh, roi_map=surf_roi_data, engine=engine)


def test_plot_surf_roi_matplotlib_specific(
    matplotlib_pyplot, surface_image_roi
):
    """Test for nilearn.plotting.surface.surf_plotting.plot_surf_roi
    for matplotlib engine specific parameters.
    """
    ENGINE = "matplotlib"
    # change vmin, vmax
    img = plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        avg_method="median",
        cbar_tick_format="%i",
        vmin=1.2,
        vmax=8.9,
        colorbar=True,
        engine=ENGINE,
    )
    img.canvas.draw()
    cbar = img.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())

    assert cbar_vmin == 1.0
    assert cbar_vmax == 8.0

    img2 = plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        vmin=1.2,
        vmax=8.9,
        colorbar=True,
        cbar_tick_format="%.2g",
        engine=ENGINE,
    )
    img2.canvas.draw()
    cbar = img2.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())

    assert cbar_vmin == 1.2
    assert cbar_vmax == 8.9


def test_plot_surf_roi_matplotlib_specific_nan_handling(
    matplotlib_pyplot,
    surface_image_parcellation,
):
    """Test for nilearn.plotting.surface.surf_plotting.plot_surf_roi
    for NAN handling with matplotlib engine.
    """
    # Test nans handling
    surface_image_parcellation.data.parts["left"][::2] = np.nan
    img = plot_surf_roi(
        surface_image_parcellation.mesh,
        roi_map=surface_image_parcellation,
        engine="matplotlib",
        hemi="left",
    )
    # Check that the resulting plot facecolors contain no transparent faces
    # (last column equals zero) even though the texture contains nan values
    tmp = img._axstack.as_list()[0].collections[0]
    n_faces = surface_image_parcellation.mesh.parts["left"].faces.shape[0]

    assert n_faces == ((tmp._facecolors[:, 3]) != 0).sum()


def test_plot_surf_roi_matplotlib_specific_plot_to_axes(
    matplotlib_pyplot, surface_image_roi
):
    """Test plotting directly on some axes."""
    ENGINE = "matplotlib"

    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        axes=None,
        figure=matplotlib_pyplot.gcf(),
        engine=ENGINE,
    )

    _, ax = matplotlib_pyplot.subplots(subplot_kw={"projection": "3d"})

    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(
            surface_image_roi.mesh,
            roi_map=surface_image_roi,
            axes=ax,
            figure=None,
            output_file=tmp_file.name,
            engine=ENGINE,
        )

    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(
            surface_image_roi.mesh,
            roi_map=surface_image_roi,
            axes=ax,
            figure=None,
            output_file=tmp_file.name,
            colorbar=True,
            engine=ENGINE,
        )


@pytest.mark.parametrize("colorbar", [True, False])
@pytest.mark.parametrize("cbar_tick_format", ["auto", "%f"])
def test_plot_surf_roi_parcellation_plotly(
    plotly,
    colorbar,
    surface_image_parcellation,
    cbar_tick_format,
):
    """Smoke test for nilearn.plotting.surface.surf_plotting.plot_surf_roi
    for plotly parameters.
    """
    plot_surf_roi(
        surface_image_parcellation.mesh,
        roi_map=surface_image_parcellation,
        engine="plotly",
        colorbar=colorbar,
        cbar_tick_format=cbar_tick_format,
    )


@pytest.mark.parametrize("avg_method", ["mean", "median"])
@pytest.mark.parametrize("symmetric_cmap", [True, False, None])
def test_plot_surf_roi_default_arguments(
    plt, engine, symmetric_cmap, avg_method, surface_image_roi
):
    """Regression test for https://github.com/nilearn/nilearn/issues/3941."""
    # To avoid extra warnings
    if engine == "plotly":
        avg_method = None

    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        engine=engine,
        symmetric_cmap=symmetric_cmap,
        darkness=None,  # to avoid deprecation warning
        cmap="RdYlBu_r",
        avg_method=avg_method,
    )


@pytest.mark.parametrize(
    "kwargs", [{"vmin": 2}, {"vmin": 2, "threshold": 5}, {"threshold": 5}]
)
def test_plot_surf_roi_colorbar_vmin_equal_across_engines(
    matplotlib_pyplot, plotly, kwargs, in_memory_mesh
):
    """Regression test for https://github.com/nilearn/nilearn/issues/3944."""
    roi_map = np.arange(0, len(in_memory_mesh.coordinates))

    mpl_plot = plot_surf_roi(
        in_memory_mesh,
        roi_map=roi_map,
        colorbar=True,
        engine="matplotlib",
        **kwargs,
    )
    plotly_plot = plot_surf_roi(
        in_memory_mesh,
        roi_map=roi_map,
        colorbar=True,
        engine="plotly",
        **kwargs,
    )
    assert (
        mpl_plot.axes[-1].get_ylim()[0] == plotly_plot.figure.data[1]["cmin"]
    )


@pytest.mark.parametrize(
    "hemispheres, views",
    [
        (["right"], ["lateral"]),
        ("left", ["lateral"]),
        (["both"], ["lateral"]),
        (["left", "right"], ["anterior"]),
        (["right"], ["medial", "lateral"]),
        (["left", "right"], ["dorsal", "ventral"]),
        # Check that manually set view angles work.
        (["left", "right"], [(210.0, 90.0), (15.0, -45.0)]),
    ],
)
def test_plot_img_on_surf_hemispheres_and_orientations(
    matplotlib_pyplot, img_3d_mni, hemispheres, views
):
    """Smoke test for nilearn.plotting.surface.plot_img_on_surf for
    combinations of 1D or 2D hemis and orientations.
    """
    # Check that all combinations of 1D or 2D hemis and orientations work.
    plot_img_on_surf(img_3d_mni, hemispheres=hemispheres, views=views)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "colorbar": True,
            "vmin": -5,
            "vmax": 5,
            "threshold": 3,
        },
        {
            "colorbar": True,
            "vmin": -1,
            "vmax": 5,
            "symmetric_cbar": False,
            "threshold": 3,
        },
        {"colorbar": False},
        {
            "colorbar": False,
            "cmap": "roy_big_bl",
        },
        {
            "colorbar": True,
            "cmap": "roy_big_bl",
            "vmax": 2,
        },
    ],
)
def test_plot_img_on_surf_colorbar(matplotlib_pyplot, img_3d_mni, kwargs):
    """Smoke test for nilearn.plotting.surface.plot_img_on_surf colorbar
    parameter.
    """
    plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], **kwargs
    )


def test_plot_img_on_surf_inflate(matplotlib_pyplot, img_3d_mni):
    """Smoke test for nilearn.plotting.surface.plot_img_on_surf inflate
    parameter.
    """
    plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], inflate=True
    )


@pytest.mark.parametrize("surf_mesh", ["fsaverage5", fetch_surf_fsaverage()])
def test_plot_img_on_surf_surf_mesh(matplotlib_pyplot, img_3d_mni, surf_mesh):
    """Smoke test for nilearn.plotting.surface.plot_img_on_surf for surf_mesh
    parameter.
    """
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right", "left"],
        views=["anterior"],
        surf_mesh=surf_mesh,
    )


def test_plot_img_on_surf_surf_mesh_low_alpha(matplotlib_pyplot, img_3d_mni):
    """Check that low alpha value do not cause floating point error.

    regression test for: https://github.com/nilearn/nilearn/issues/4900
    """
    plot_img_on_surf(img_3d_mni, threshold=3, alpha=0.1)


def test_plot_img_on_surf_with_invalid_orientation(img_3d_mni):
    """Test if nilearn.plotting.surface.plot_img_on_surf raises error when
    invalid views parameter is specified.
    """
    kwargs = {"hemisphere": ["right"], "inflate": True}
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["latral"], **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["dorsal", "post"], **kwargs)
    with pytest.raises(TypeError):
        plot_img_on_surf(img_3d_mni, views=0, **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["medial", {"a": "a"}], **kwargs)


@pytest.mark.parametrize(
    "hemispheres", [["lft]"], "lft", 0, ["left", "right", "middle"]]
)
def test_plot_img_on_surf_with_invalid_hemisphere(img_3d_mni, hemispheres):
    """Test if nilearn.plotting.surface.plot_img_on_surf raises error when
    invalid hemispheres parameter is specified.
    """
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["lateral"],
            inflate=True,
            hemispheres=hemispheres,
        )


def test_plot_img_on_surf_with_figure_kwarg(img_3d_mni):
    """Test if nilearn.plotting.surface.plot_img_on_surf raises error when
    figure parameter is specified.
    """
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            figure=True,
        )


def test_plot_img_on_surf_with_axes_kwarg(img_3d_mni):
    """Test if nilearn.plotting.surface.plot_img_on_surf raises error when axes
    parameter is specified.
    """
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            axes="something",
        )


def test_plot_img_on_surf_with_engine_kwarg(img_3d_mni):
    """Test if nilearn.plotting.surface.plot_img_on_surf raises error when
    engine parameter is specified.
    """
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            engine="something",
        )


def test_plot_img_on_surf_title(matplotlib_pyplot, img_3d_mni):
    """Test nilearn.plotting.surface.plot_img_on_surf with and without title
    specified.
    .
    """
    title = "Title"
    fig, _ = plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"]
    )
    assert fig._suptitle is None, "Created title without title kwarg."
    fig, _ = plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], title=title
    )
    assert fig._suptitle is not None, "Title not created."
    assert fig._suptitle.get_text() == title, "Title text not assigned."


def test_plot_img_on_surf_output_file(matplotlib_pyplot, tmp_path, img_3d_mni):
    """Test nilearn.plotting.surface.plot_img_on_surf for output_file."""
    fname = tmp_path / "tmp.png"
    return_value = plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        output_file=str(fname),
    )
    assert return_value is None, "Returned figure and axes on file output."
    assert fname.is_file(), "Saved image file could not be found."


def test_plot_img_on_surf_input_as_file(matplotlib_pyplot, img_3d_mni_as_file):
    """Test nifti is supported when passed as string or path to a file."""
    plot_img_on_surf(stat_map=img_3d_mni_as_file)
    plot_img_on_surf(stat_map=str(img_3d_mni_as_file))


@pytest.mark.parametrize(
    "function",
    [plot_surf_roi, plot_surf_stat_map, plot_surf_contours, plot_surf],
)
def test_error_nifti_not_supported(
    function, img_3d_mni_as_file, in_memory_mesh
):
    """Test nifti file not supported by several surface plotting functions."""
    with pytest.raises(ValueError, match="The input type is not recognized"):
        function(in_memory_mesh, img_3d_mni_as_file)
    with pytest.raises(ValueError, match="The input type is not recognized"):
        function(in_memory_mesh, str(img_3d_mni_as_file))
