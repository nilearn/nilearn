# Tests for functions in surf_plotting.py

# ruff: noqa: ARG001

import re
import tempfile
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from numpy.testing import assert_array_equal

from nilearn._utils.helpers import is_kaleido_installed, is_plotly_installed
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_img_on_surf,
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.displays import PlotlySurfaceFigure, SurfaceFigure
from nilearn.surface import SurfaceImage

try:
    import IPython.display  # noqa:F401
except ImportError:
    IPYTHON_INSTALLED = False
else:
    IPYTHON_INSTALLED = True


@pytest.fixture
def bg_map(rng, in_memory_mesh):
    """Return a background map with positive value."""
    return np.abs(rng.standard_normal(size=in_memory_mesh.n_vertices))


@pytest.mark.parametrize(
    "fn",
    [
        plot_surf,
        plot_surf_stat_map,
        plot_surf_contours,
        plot_surf_roi,
    ],
)
def test_check_surface_plotting_inputs_error_mash_and_data_none(fn):
    """Fail if no mesh or data is passed."""
    with pytest.raises(TypeError, match="cannot both be None"):
        fn(None, None)


def test_check_surface_plotting_inputs_errors():
    """Fail if mesh is none and data is not a SurfaceImage."""
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf(surf_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_stat_map(stat_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_contours(roi_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_roi(roi_map=1, surf_mesh=None)


def test_surface_plotting_axes_error(surf_img_1d):
    """Test error msg for invalid axes."""
    figure, axes = plt.subplots()
    with pytest.raises(AttributeError, match="the projection must be '3d'"):
        plot_surf_stat_map(stat_map=surf_img_1d, axes=axes)


def test_plot_surf_contours_warning_hemi(in_memory_mesh):
    """Test warning that hemi will be ignored."""
    parcellation = np.zeros((in_memory_mesh.n_vertices,))
    parcellation[in_memory_mesh.faces[3]] = 1
    with pytest.warns(UserWarning, match="This value will be ignored"):
        plot_surf_contours(in_memory_mesh, parcellation, hemi="left")


def test_surface_figure():
    s = SurfaceFigure()
    assert s.output_file is None
    assert s.figure is None
    with pytest.raises(NotImplementedError):
        s.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        s._check_output_file()
    s._check_output_file("foo.png")
    assert s.output_file == "foo.png"
    s = SurfaceFigure(output_file="bar.png")
    assert s.output_file == "bar.png"


@pytest.mark.skipif(
    is_plotly_installed(),
    reason=("This test only runs if Plotly is not installed."),
)
def test_plotly_surface_figure_import_error():
    """Test that an ImportError is raised when instantiating \
       a PlotlySurfaceFigure without having Plotly installed.
    """
    with pytest.raises(ImportError, match="Plotly is required"):
        PlotlySurfaceFigure()


@pytest.mark.skipif(
    not is_plotly_installed() or is_kaleido_installed(),
    reason=("This test only runs if Plotly is installed, but not kaleido."),
)
def test_plotly_surface_figure_savefig_error():
    """Test that an ImportError is raised when saving \
       a PlotlySurfaceFigure without having kaleido installed.
    """
    with pytest.raises(ImportError, match="`kaleido` is required"):
        PlotlySurfaceFigure().savefig()


@pytest.mark.skipif(
    not is_plotly_installed() or not is_kaleido_installed(),
    reason=("Plotly and/or kaleido not installed; required for this test."),
)
def test_plotly_surface_figure():
    ps = PlotlySurfaceFigure()
    assert ps.output_file is None
    assert ps.figure is None
    ps.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        ps.savefig()
    ps.savefig("foo.png")


@pytest.mark.skipif(
    not is_plotly_installed()
    or not IPYTHON_INSTALLED
    or not is_kaleido_installed(),
    reason=(
        "Plotly, Kaleido and/or Ipython is not installed; required for this"
        " test."
    ),
)
@pytest.mark.parametrize("renderer", ["png", "jpeg", "svg"])
def test_plotly_show(renderer):
    import plotly.graph_objects as go

    ps = PlotlySurfaceFigure(go.Figure())
    assert ps.output_file is None
    assert ps.figure is not None
    with mock.patch("IPython.display.display") as mock_display:
        ps.show(renderer=renderer)
    assert len(mock_display.call_args.args) == 1
    key = "svg+xml" if renderer == "svg" else renderer
    assert f"image/{key}" in mock_display.call_args.args[0]


@pytest.mark.skipif(
    not is_plotly_installed() or not is_kaleido_installed(),
    reason=("Plotly and/or kaleido not installed; required for this test."),
)
def test_plotly_savefig(tmp_path):
    import plotly.graph_objects as go

    ps = PlotlySurfaceFigure(go.Figure(), output_file=tmp_path / "foo.png")
    assert ps.output_file == tmp_path / "foo.png"
    assert ps.figure is not None
    ps.savefig()
    assert (tmp_path / "foo.png").exists()


@pytest.mark.parametrize("input_obj", ["foo", Figure(), ["foo", "bar"]])
def test_instantiation_error_plotly_surface_figure(plotly, input_obj):
    with pytest.raises(
        TypeError,
        match=("`PlotlySurfaceFigure` accepts only plotly figure objects."),
    ):
        PlotlySurfaceFigure(input_obj)


def test_value_error_get_faces_on_edge(plotly, in_memory_mesh):
    """Test that calling _get_faces_on_edge raises a ValueError when \
       called with with indices that do not form a region.
    """
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(
        ValueError, match=("Vertices in parcellation do not form region.")
    ):
        figure._get_faces_on_edge([91])


def test_plot_surf_contours_errors_with_plotly_figure(plotly, in_memory_mesh):
    """Test that plot_surf_contours rasises error when given plotly obj."""
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(ValueError):
        plot_surf_contours(in_memory_mesh, np.ones((10,)), figure=figure)


def test_plot_surf_contours_errors_with_plotly_axes(plotly, in_memory_mesh):
    """Test that plot_surf_contours rasises error when given plotly \
        obj as axis.
    """
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(ValueError):
        plot_surf_contours(in_memory_mesh, np.ones((10,)), axes=figure)


def test_plotly_surface_figure_warns_on_isolated_roi(plotly, in_memory_mesh):
    """Test that a warning is generated for ROIs with isolated vertices."""
    figure = plot_surf(in_memory_mesh, engine="plotly")
    # the method raises an error because the (randomly generated)
    # vertices don't form regions
    try:
        with pytest.raises(UserWarning, match="contains isolated vertices:"):
            figure.add_contours(levels=[0], roi_map=np.array([0, 1] * 10))
    except Exception:
        pass


def test_distant_line_segments_detected_as_not_intersecting(plotly):
    """Test that distant lines are detected as not intersecting."""
    assert not PlotlySurfaceFigure._do_segs_intersect(0, 0, 1, 1, 5, 5, 6, 6)


@pytest.mark.parametrize("levels,labels", [([0], ["a", "b"]), ([0, 1], ["a"])])
def test_value_error_add_contours_levels_labels(
    plotly, levels, labels, in_memory_mesh
):
    """Test that add_contours raises a ValueError when called with levels and \
    labels that have incompatible lengths.
    """
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(
        ValueError,
        match=("levels and labels need to be either the same length or None."),
    ):
        figure.add_contours(
            levels=levels, labels=labels, roi_map=np.ones((10,))
        )


@pytest.mark.parametrize(
    "levels,lines",
    [([0], [{}, {}]), ([0, 1], [{}, {}, {}])],
)
def test_value_error_add_contours_levels_lines(
    plotly, levels, lines, in_memory_mesh
):
    """Test that add_contours raises a ValueError when called with levels and \
    lines that have incompatible lengths.
    """
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(
        ValueError,
        match=("levels and lines need to be either the same length or None."),
    ):
        figure.add_contours(levels=levels, lines=lines, roi_map=np.ones((10,)))


@pytest.fixture
def surf_roi_data(rng, in_memory_mesh):
    roi_map = np.zeros((in_memory_mesh.n_vertices, 1))
    roi_idx = rng.integers(0, in_memory_mesh.n_vertices, size=10)
    roi_map[roi_idx] = 1
    return roi_map


@pytest.fixture
def surface_image_roi(surf_mask_1d):
    """SurfaceImage for plotting."""
    return surf_mask_1d


@pytest.fixture
def surface_image_parcellation(rng, in_memory_mesh):
    data = rng.integers(100, size=(in_memory_mesh.n_vertices, 1)).astype(float)
    parcellation = SurfaceImage(
        mesh={"left": in_memory_mesh, "right": in_memory_mesh},
        data={"left": data, "right": data},
    )
    return parcellation


def test_add_contours(plotly, surface_image_roi):
    """Test that add_contours updates data in PlotlySurfaceFigure."""
    figure = plot_surf(surface_image_roi.mesh, engine="plotly")
    figure.add_contours(surface_image_roi)
    assert len(figure.figure.to_dict().get("data")) == 4

    figure.add_contours(surface_image_roi, levels=[1])
    assert len(figure.figure.to_dict().get("data")) == 5


@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_add_contours_hemi(
    plotly,
    surface_image_roi,
    hemi,
):
    """Test that add_contours works with all hemi inputs."""
    if hemi == "both":
        n_vertices = surface_image_roi.mesh.n_vertices
    else:
        n_vertices = surface_image_roi.data.parts[hemi].shape[0]
    figure = plot_surf(
        surface_image_roi.mesh,
        engine="plotly",
        hemi=hemi,
    )
    figure.add_contours(surface_image_roi)
    assert figure._coords.shape[0] == n_vertices


def test_add_contours_plotly_surface_image(plotly, surface_image_roi):
    """Test that add_contours works with SurfaceImage."""
    figure = plot_surf(
        surf_map=surface_image_roi, hemi="left", engine="plotly"
    )
    figure.add_contours(roi_map=surface_image_roi)


def test_surface_figure_add_contours_raises_not_implemented(plotly):
    """Test that calling add_contours method of SurfaceFigure raises a \
    NotImplementedError.
    """
    figure = SurfaceFigure()
    with pytest.raises(NotImplementedError):
        figure.add_contours()


def test_add_contours_has_name(plotly, surface_image_roi):
    """Test that contours added to a PlotlySurfaceFigure can be named."""
    figure = plot_surf(surface_image_roi.mesh, engine="plotly")
    figure.add_contours(surface_image_roi, levels=[1], labels=["x"])
    assert figure.figure.to_dict().get("data")[2].get("name") == "x"


def test_add_contours_lines_duplicated(plotly, surface_image_roi):
    """Test that the specifications of length 1 line provided to \
     add_contours are duplicated to all requested contours.
    """
    figure = plot_surf(surface_image_roi.mesh, engine="plotly")
    figure.add_contours(surface_image_roi, lines=[{"width": 10}])
    newlines = figure.figure.to_dict().get("data")[2:]
    assert all(x.get("line").__contains__("width") for x in newlines)


@pytest.mark.parametrize(
    "key,value",
    [
        ("color", "yellow"),
        ("width", 10),
    ],
)
def test_add_contours_line_properties(plotly, key, value, surface_image_roi):
    """Test that the specifications of a line provided to add_contours are \
    stored in the PlotlySurfaceFigure data.
    """
    figure = plot_surf(surface_image_roi.mesh, engine="plotly")
    figure.add_contours(surface_image_roi, levels=[1], lines=[{key: value}])
    newline = figure.figure.to_dict().get("data")[2].get("line")
    assert newline.get(key) == value


def test_plot_surf_engine_error(in_memory_mesh):
    with pytest.raises(ValueError, match="Unknown plotting engine"):
        plot_surf(in_memory_mesh, engine="foo")


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="This test is run only if plotly is not installed.",
)
def test_plot_surf_engine_error_plotly_not_installed(in_memory_mesh):
    with pytest.raises(ImportError, match="Using engine"):
        plot_surf(in_memory_mesh, engine="plotly")


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf(
    matplotlib_pyplot, engine, tmp_path, in_memory_mesh, bg_map
):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

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
    plot_surf(in_memory_mesh, bg_map=bg_map, darkness=0.5, engine=engine)
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


@pytest.mark.skipif(
    not is_plotly_installed() or is_kaleido_installed(),
    reason=("This test only runs if Plotly is installed, but not kaleido."),
)
def test_plot_surf_error_when_kaleido_missing(
    tmp_path, in_memory_mesh, bg_map
):
    with pytest.raises(ImportError, match="Saving figures"):
        engine = "plotly"
        # Plot with non None output file
        plot_surf(
            in_memory_mesh,
            bg_map=bg_map,
            engine=engine,
            output_file=tmp_path / "tmp.png",
        )


@pytest.mark.parametrize("view", ["anterior", "posterior"])
@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_plot_surf_hemi_views_plotly(
    matplotlib_pyplot, plotly, in_memory_mesh, hemi, view, bg_map
):
    """Check plotting view and hemispheres."""
    plot_surf(
        in_memory_mesh, bg_map=bg_map, hemi=hemi, view=view, engine="plotly"
    )


def test_plot_surf_with_title(matplotlib_pyplot, in_memory_mesh, bg_map):
    """Check title in figure."""
    display = plot_surf(
        in_memory_mesh, bg_map=bg_map, title="Test title", engine="matplotlib"
    )

    assert len(display.axes) == 1
    assert display.axes[0].title._text == "Test title"


def test_plot_surf_avg_method(matplotlib_pyplot, in_memory_mesh, bg_map):
    # Plot with avg_method
    # Test all built-in methods and check
    faces = in_memory_mesh.faces

    for method in ["mean", "median", "min", "max"]:
        display = plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=method,
            engine="matplotlib",
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
        cmap = plt.get_cmap(plt.rcParamsDefault["image.cmap"])
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
        engine="matplotlib",
    )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_error(engine, rng, in_memory_mesh):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")
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


@pytest.mark.parametrize("kwargs", [{"avg_method": "mean"}, {"alpha": "auto"}])
def test_plot_surf_warnings_not_implemented_in_plotly(
    plotly, kwargs, in_memory_mesh, bg_map
):
    with pytest.warns(
        UserWarning, match="is not implemented for the plotly engine"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            engine="plotly",
            **kwargs,
        )


def test_plot_surf_avg_method_errors(in_memory_mesh, bg_map):
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
            engine="matplotlib",
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
            engine="matplotlib",
        )

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method="foo",
            engine="matplotlib",
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
            engine="matplotlib",
        )


# @pytest.mark.parametrize()
@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map(matplotlib_pyplot, engine, in_memory_mesh, bg_map):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    alpha = 1 if engine == "matplotlib" else None
    # Plot mesh with stat map
    plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, engine=engine)
    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, alpha=alpha, engine=engine
    )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_with_background(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Plot mesh with background and stat map."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

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


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_with_title(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Check title is added."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    display = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, title="Stat map title"
    )
    assert display.axes[0].title._text == "Stat map title"


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_with_threshold(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Check title is added."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        threshold=0.3,
        engine=engine,
    )


def test_plot_surf_stat_map_colorbar_tick(plotly, in_memory_mesh, bg_map):
    """Change colorbar tick format."""
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        cbar_tick_format="%.2g",
        engine="plotly",
    )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_vmax(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Change vmax."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, vmax=5, engine=engine)


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_colormap(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Change colormap."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, cmap="cubehelix", engine=engine
    )


def test_plot_surf_stat_map_matplotlib_specific(
    matplotlib_pyplot, in_memory_mesh, bg_map
):
    # Plot to axes
    axes = plt.subplots(ncols=2, subplot_kw={"projection": "3d"})[1]
    for ax in axes.flatten():
        plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, axes=ax)
    axes = plt.subplots(ncols=2, subplot_kw={"projection": "3d"})[1]
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


def test_plot_surf_stat_map_error(in_memory_mesh, bg_map):
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


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_surf_roi(matplotlib_pyplot, engine, surface_image_roi, colorbar):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")
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


@pytest.mark.parametrize("colorbar", [True, False])
@pytest.mark.parametrize("cbar_tick_format", ["auto", "%f"])
def test_plot_surf_parcellation_plotly(
    plotly,
    colorbar,
    surface_image_parcellation,
    cbar_tick_format,
):
    plot_surf_roi(
        surface_image_parcellation.mesh,
        roi_map=surface_image_parcellation,
        engine="plotly",
        colorbar=colorbar,
        cbar_tick_format=cbar_tick_format,
    )


def test_plot_surf_roi_matplotlib_specific(
    matplotlib_pyplot, surface_image_roi
):
    # change vmin, vmax
    img = plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        vmin=1.2,
        vmax=8.9,
        colorbar=True,
        engine="matplotlib",
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
        engine="matplotlib",
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
    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        axes=None,
        figure=plt.gcf(),
        engine="matplotlib",
    )

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(
            surface_image_roi.mesh,
            roi_map=surface_image_roi,
            axes=ax,
            figure=None,
            output_file=tmp_file.name,
            engine="matplotlib",
        )

    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(
            surface_image_roi.mesh,
            roi_map=surface_image_roi,
            axes=ax,
            figure=None,
            output_file=tmp_file.name,
            colorbar=True,
            engine="matplotlib",
        )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_roi_error(engine, rng, in_memory_mesh, surf_roi_data):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")
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


@pytest.mark.skipif(
    not is_plotly_installed(),
    reason=("This test only runs if Plotly is installed."),
)
@pytest.mark.parametrize(
    "kwargs", [{"vmin": 2}, {"vmin": 2, "threshold": 5}, {"threshold": 5}]
)
def test_plot_surf_roi_colorbar_vmin_equal_across_engines(
    matplotlib_pyplot, kwargs, in_memory_mesh
):
    """See issue https://github.com/nilearn/nilearn/issues/3944."""
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
    # Check that all combinations of 1D or 2D hemis and orientations work.
    plot_img_on_surf(img_3d_mni, hemispheres=hemispheres, views=views)


def test_plot_img_on_surf_colorbar(matplotlib_pyplot, img_3d_mni):
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=True,
        vmin=-5,
        vmax=5,
        threshold=3,
    )
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=True,
        vmin=-1,
        vmax=5,
        symmetric_cbar=False,
        threshold=3,
    )
    plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], colorbar=False
    )
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=False,
        cmap="roy_big_bl",
    )
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=True,
        cmap="roy_big_bl",
        vmax=2,
    )


def test_plot_img_on_surf_inflate(matplotlib_pyplot, img_3d_mni):
    plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], inflate=True
    )


@pytest.mark.parametrize("surf_mesh", ["fsaverage5", fetch_surf_fsaverage()])
def test_plot_img_on_surf_surf_mesh(matplotlib_pyplot, img_3d_mni, surf_mesh):
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
    kwargs = {"hemisphere": ["right"], "inflate": True}
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["latral"], **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["dorsal", "post"], **kwargs)
    with pytest.raises(TypeError):
        plot_img_on_surf(img_3d_mni, views=0, **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["medial", {"a": "a"}], **kwargs)


def test_plot_img_on_surf_with_invalid_hemisphere(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni, views=["lateral"], inflate=True, hemispheres=["lft]"]
        )
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni, views=["medial"], inflate=True, hemispheres=["lef"]
        )
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior", "posterior"],
            inflate=True,
            hemispheres=["left", "right", "middle"],
        )


def test_plot_img_on_surf_with_figure_kwarg(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            figure=True,
        )


def test_plot_img_on_surf_with_axes_kwarg(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            axes="something",
        )


def test_plot_img_on_surf_with_engine_kwarg(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            engine="something",
        )


def test_plot_img_on_surf_title(matplotlib_pyplot, img_3d_mni):
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


@pytest.fixture
def parcellation(in_memory_mesh):
    parcellation = np.zeros((in_memory_mesh.n_vertices,))
    parcellation[in_memory_mesh.faces[3]] = 1
    parcellation[in_memory_mesh.faces[5]] = 2
    return parcellation


def test_plot_surf_contours(
    matplotlib_pyplot, in_memory_mesh, parcellation, surf_mask_1d
):
    plot_surf_contours(in_memory_mesh, parcellation)
    plot_surf_contours(in_memory_mesh, parcellation, levels=[1, 2])
    plot_surf_contours(
        in_memory_mesh, parcellation, levels=[1, 2], cmap="gist_ncar"
    )


def test_plot_surf_contour_roi_map_as_surface_image(
    matplotlib_pyplot, surf_mesh, surf_mask_1d
):
    """Check that mesh can be PolyMesh and roi_map can be a SurfaceImage."""
    plot_surf_contours(surf_mesh, roi_map=surf_mask_1d, hemi=None)


def test_plot_surf_contours_legend(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    fig = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        legend=True,
    )
    assert fig.legends is not None


def test_plot_surf_contours_colors(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
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


def test_plot_surf_contours_fig_axes(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    fig, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_surf_contours(in_memory_mesh, parcellation, axes=axes)
    plot_surf_contours(in_memory_mesh, parcellation, figure=fig)


def test_plot_surf_contours_axis_title(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
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


def test_plot_surf_contours_error(rng, in_memory_mesh, parcellation):
    # we need an invalid parcellation for testing
    invalid_parcellation = rng.uniform(size=(in_memory_mesh.n_vertices))
    with pytest.raises(
        ValueError, match="Vertices in parcellation do not form region."
    ):
        plot_surf_contours(in_memory_mesh, invalid_parcellation)

    _, axes = plt.subplots(1, 1)
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


@pytest.mark.parametrize("avg_method", ["mean", "median"])
@pytest.mark.parametrize("symmetric_cmap", [True, False, None])
@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_roi_default_arguments(
    plotly, engine, symmetric_cmap, avg_method, surface_image_roi
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
