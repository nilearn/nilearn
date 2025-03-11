# Tests for functions in surf_plotting.py

# ruff: noqa: ARG001

import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from nilearn._utils.helpers import is_plotly_installed
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_img_on_surf,
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.displays import PlotlySurfaceFigure, SurfaceFigure
from nilearn.plotting.surface._matplotlib import (
    MATPLOTLIB_VIEWS,
    _compute_facecolors,
    _get_ticks,
    _get_view_plot_surf,
)
from nilearn.plotting.surface._plotly import (
    _get_view_plot_surf_plotly,
)
from nilearn.surface import (
    SurfaceImage,
    load_surf_data,
    load_surf_mesh,
)

EXPECTED_CAMERAS_PLOTLY = [
    (
        "left",
        "lateral",
        (0, 180),
        {
            "eye": {"x": -1.5, "y": 0, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    (
        "left",
        "medial",
        (0, 0),
        {
            "eye": {"x": 1.5, "y": 0, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Dorsal left
    (
        "left",
        "dorsal",
        (90, 0),
        {
            "eye": {"x": 0, "y": 0, "z": 1.5},
            "up": {"x": -1, "y": 0, "z": 0},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Ventral left
    (
        "left",
        "ventral",
        (270, 0),
        {
            "eye": {"x": 0, "y": 0, "z": -1.5},
            "up": {"x": 1, "y": 0, "z": 0},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Anterior left
    (
        "left",
        "anterior",
        (0, 90),
        {
            "eye": {"x": 0, "y": 1.5, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Posterior left
    (
        "left",
        "posterior",
        (0, 270),
        {
            "eye": {"x": 0, "y": -1.5, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Lateral right
    (
        "right",
        "lateral",
        (0, 0),
        {
            "eye": {"x": 1.5, "y": 0, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Medial right
    (
        "right",
        "medial",
        (0, 180),
        {
            "eye": {"x": -1.5, "y": 0, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Dorsal right
    (
        "right",
        "dorsal",
        (90, 0),
        {
            "eye": {"x": 0, "y": 0, "z": 1.5},
            "up": {"x": -1, "y": 0, "z": 0},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Ventral right
    (
        "right",
        "ventral",
        (270, 0),
        {
            "eye": {"x": 0, "y": 0, "z": -1.5},
            "up": {"x": 1, "y": 0, "z": 0},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Anterior right
    (
        "right",
        "anterior",
        (0, 90),
        {
            "eye": {"x": 0, "y": 1.5, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
    # Posterior right
    (
        "right",
        "posterior",
        (0, 270),
        {
            "eye": {"x": 0, "y": -1.5, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
        },
    ),
]


EXPECTED_VIEW_MATPLOTLIB = {
    "left": {
        "anterior": (0, 90),
        "posterior": (0, 270),
        "medial": (0, 0),
        "lateral": (0, 180),
        "dorsal": (90, 0),
        "ventral": (270, 0),
    },
    "right": {
        "anterior": (0, 90),
        "posterior": (0, 270),
        "medial": (0, 180),
        "lateral": (0, 0),
        "dorsal": (90, 0),
        "ventral": (270, 0),
    },
    "both": {
        "right": (0, 0),
        "left": (0, 180),
        "dorsal": (90, 0),
        "ventral": (270, 0),
        "anterior": (0, 90),
        "posterior": (0, 270),
    },
}


@pytest.fixture
def bg_map(rng, in_memory_mesh):
    """Return a background map with posive value."""
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


def test_check_surface_plotting_inputs_errors(surf_img_1d):
    """Fail if mesh is none and data is not not SurfaceImage."""
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


@pytest.mark.parametrize("full_view", EXPECTED_CAMERAS_PLOTLY)
def test_get_view_plot_surf_plotly(full_view):
    from nilearn.plotting.surface._plotly import (
        _get_camera_view_from_elevation_and_azimut,
        _get_camera_view_from_string_view,
    )

    hemi, view_name, (elev, azim), expected_camera_view = full_view
    camera_view = _get_view_plot_surf_plotly(hemi, view_name)
    camera_view_string = _get_camera_view_from_string_view(hemi, view_name)
    camera_view_elev_azim = _get_camera_view_from_elevation_and_azimut(
        (elev, azim)
    )
    # Check each camera view parameter
    for k in ["center", "eye", "up"]:
        # Check default camera view
        assert np.allclose(
            list(camera_view[k].values()),
            list(expected_camera_view[k].values()),
        )
        # Check camera view obtained from string view
        assert np.allclose(
            list(camera_view_string[k].values()),
            list(expected_camera_view[k].values()),
        )
        # Check camera view obtained from elevation & azimut
        assert np.allclose(
            list(camera_view_elev_azim[k].values()),
            list(expected_camera_view[k].values()),
        )


@pytest.mark.parametrize("hemi, views", MATPLOTLIB_VIEWS.items())
def test_get_view_plot_surf_matplotlib(hemi, views):
    for v in views:
        assert (
            _get_view_plot_surf(hemi, v) == EXPECTED_VIEW_MATPLOTLIB[hemi][v]
        )


def test_value_error_get_faces_on_edge(plotly, in_memory_mesh):
    """Test that calling _get_faces_on_edge raises a ValueError when \
       called with with indices that do not form a region.
    """
    figure = plot_surf(in_memory_mesh, engine="plotly")
    with pytest.raises(
        ValueError, match=("Vertices in parcellation do not form region.")
    ):
        figure._get_faces_on_edge([91])


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


@pytest.mark.parametrize(
    "view,is_valid",
    [
        ("lateral", True),
        ("medial", True),
        ("latreal", False),
        ((100, 100), True),
        ([100.0, 100.0], True),
        ((100, 100, 1), False),
        (("lateral", "medial"), False),
        ([100, "bar"], False),
    ],
)
def test_check_view_is_valid(view, is_valid):
    from nilearn.plotting.surface._utils import _check_view_is_valid

    assert _check_view_is_valid(view) is is_valid


@pytest.mark.parametrize(
    "hemi,is_valid",
    [
        ("left", True),
        ("right", True),
        ("both", True),
        ("lft", False),
    ],
)
def test_check_hemisphere_is_valid(hemi, is_valid):
    from nilearn.plotting.surface._utils import _check_hemisphere_is_valid

    assert _check_hemisphere_is_valid(hemi) is is_valid


@pytest.mark.parametrize("hemi,view", [("foo", "medial"), ("bar", "anterior")])
def test_get_view_plot_surf_hemisphere_errors(hemi, view):
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        _get_view_plot_surf(hemi, view)
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        _get_view_plot_surf_plotly(hemi, view)


@pytest.mark.parametrize(
    "hemi,view,f",
    [
        ("left", "foo", _get_view_plot_surf),
        ("right", "bar", _get_view_plot_surf_plotly),
        ("both", "lateral", _get_view_plot_surf),
        ("both", "medial", _get_view_plot_surf_plotly),
        ("both", "foo", _get_view_plot_surf),
        ("both", "bar", _get_view_plot_surf_plotly),
    ],
)
def test_get_view_plot_surf_view_errors(hemi, view, f):
    with pytest.raises(ValueError, match="Invalid view definition"):
        f(hemi, view)


def test_configure_title():
    from nilearn.plotting.surface._plotly import _configure_title

    assert _configure_title(None, None) == {}
    assert _configure_title(None, 22) == {}
    config = _configure_title("Test Title", 22, color="green")
    assert config["text"] == "Test Title"
    assert config["x"] == 0.5
    assert config["y"] == 0.96
    assert config["xanchor"] == "center"
    assert config["yanchor"] == "top"
    assert config["font"]["size"] == 22
    assert config["font"]["color"] == "green"


@pytest.mark.parametrize(
    "data,expected",
    [
        (np.linspace(0, 1, 100), (0, 1)),
        (np.linspace(-0.7, -0.01, 40), (-0.7, -0.01)),
    ],
)
def test_get_bounds(data, expected):
    from nilearn.plotting.surface._matplotlib import _get_bounds

    assert _get_bounds(data) == expected
    assert _get_bounds(data, vmin=0.2) == (0.2, expected[1])
    assert _get_bounds(data, vmax=0.8) == (expected[0], 0.8)
    assert _get_bounds(data, vmin=0.1, vmax=0.8) == (0.1, 0.8)


def test_plot_surf_engine_error(in_memory_mesh):
    with pytest.raises(ValueError, match="Unknown plotting engine"):
        plot_surf(in_memory_mesh, engine="foo")


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


@pytest.mark.parametrize(
    "vmin,vmax,cbar_tick_format,expected",
    [
        (0, 0, "%i", [0]),
        (0, 3, "%i", [0, 1, 2, 3]),
        (0, 4, "%i", [0, 1, 2, 3, 4]),
        (1, 5, "%i", [1, 2, 3, 4, 5]),
        (0, 5, "%i", [0, 1.25, 2.5, 3.75, 5]),
        (0, 10, "%i", [0, 2.5, 5, 7.5, 10]),
        (0, 0, "%.1f", [0]),
        (0, 1, "%.1f", [0, 0.25, 0.5, 0.75, 1]),
        (1, 2, "%.1f", [1, 1.25, 1.5, 1.75, 2]),
        (1.1, 1.2, "%.1f", [1.1, 1.125, 1.15, 1.175, 1.2]),
        (0, np.nextafter(0, 1), "%.1f", [0.0e000, 5.0e-324]),
    ],
)
def test_get_ticks(vmin, vmax, cbar_tick_format, expected):
    ticks = _get_ticks(vmin, vmax, cbar_tick_format, threshold=None)
    assert 1 <= len(ticks) <= 5
    assert ticks[0] == vmin and ticks[-1] == vmax
    assert (
        len(np.unique(ticks)) == len(expected)
        and (np.unique(ticks) == expected).all()
    )


def test_compute_facecolors_matplotlib():
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_left"])
    alpha = "auto"
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(load_surf_data(fsaverage["curv_left"]))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert bg_min < 0 or bg_max > 1

    facecolors_auto_normalized = _compute_facecolors(
        bg_map,
        mesh.faces,
        len(mesh.coordinates),
        None,
        alpha,
    )

    assert len(facecolors_auto_normalized) == len(mesh.faces)

    # Manually set values of background map between 0 and 1
    bg_map_normalized = (bg_map - bg_min) / (bg_max - bg_min)
    assert np.min(bg_map_normalized) == 0 and np.max(bg_map_normalized) == 1

    facecolors_manually_normalized = _compute_facecolors(
        bg_map_normalized,
        mesh.faces,
        len(mesh.coordinates),
        None,
        alpha,
    )

    assert len(facecolors_manually_normalized) == len(mesh.faces)
    assert np.allclose(
        facecolors_manually_normalized, facecolors_auto_normalized
    )

    # Scale background map between 0.25 and 0.75
    bg_map_scaled = bg_map_normalized / 2 + 0.25
    assert np.min(bg_map_scaled) == 0.25 and np.max(bg_map_scaled) == 0.75

    facecolors_manually_rescaled = _compute_facecolors(
        bg_map_scaled,
        mesh.faces,
        len(mesh.coordinates),
        None,
        alpha,
    )

    assert len(facecolors_manually_rescaled) == len(mesh.faces)
    assert not np.allclose(
        facecolors_manually_rescaled, facecolors_auto_normalized
    )


def test_compute_facecolors_matplotlib_deprecation():
    """Test warning deprecation."""
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_left"])
    alpha = "auto"
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(load_surf_data(fsaverage["curv_left"]))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert bg_min < 0 or bg_max > 1
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The `darkness` parameter will be deprecated in release 0.13. "
            "We recommend setting `darkness` to None"
        ),
    ):
        _compute_facecolors(
            bg_map,
            mesh.faces,
            len(mesh.coordinates),
            0.5,
            alpha,
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
