# ruff: noqa: ARG001

import numpy as np
import pytest

from nilearn._utils.helpers import is_kaleido_installed
from nilearn.plotting import (
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.surface._plotly_backend import (
    _configure_title,
    _get_camera_view_from_elevation_and_azimut,
    _get_camera_view_from_string_view,
    _get_view_plot_surf,
)

ENGINE = "plotly"

pytest.importorskip(
    ENGINE,
    reason="Plotly is not installed; required to run the tests!",
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


@pytest.mark.parametrize("full_view", EXPECTED_CAMERAS_PLOTLY)
def test_get_view_plot_surf(full_view):
    hemi, view_name, (elev, azim), expected_camera_view = full_view
    camera_view = _get_view_plot_surf(hemi, view_name)
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


@pytest.mark.parametrize("hemi,view", [("foo", "medial"), ("bar", "anterior")])
def test_get_view_plot_surf_hemisphere_errors(hemi, view):
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        _get_view_plot_surf(hemi, view)


@pytest.mark.parametrize(
    "hemi,view",
    [
        ("left", "foo"),
        ("right", "bar"),
        ("both", "lateral"),
        ("both", "medial"),
        ("both", "foo"),
    ],
)
def test_get_view_plot_surf_view_errors(hemi, view):
    with pytest.raises(ValueError, match="Invalid view definition"):
        _get_view_plot_surf(hemi, view)


def test_configure_title():
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


def test_plot_surf_contours_errors_with_plotly_figure(in_memory_mesh):
    """Test that plot_surf_contours rasises error when given plotly obj."""
    figure = plot_surf(in_memory_mesh, engine=ENGINE)
    with pytest.raises(ValueError):
        plot_surf_contours(in_memory_mesh, np.ones((10,)), figure=figure)


def test_plot_surf_contours_errors_with_plotly_axes(in_memory_mesh):
    """Test that plot_surf_contours rasises error when given plotly \
        obj as axis.
    """
    figure = plot_surf(in_memory_mesh, engine=ENGINE)
    with pytest.raises(ValueError):
        plot_surf_contours(in_memory_mesh, np.ones((10,)), axes=figure)


@pytest.mark.skipif(
    is_kaleido_installed(),
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


def test_plot_surf_stat_map_colorbar_tick(plotly, in_memory_mesh, bg_map):
    """Change colorbar tick format."""
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        cbar_tick_format="%.2g",
        engine="plotly",
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
