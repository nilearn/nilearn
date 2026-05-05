"""Test nilearn.plotting.surface._plotly_backend functions."""

import numpy as np
import pytest

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting.js_plotting_utils import decode
from nilearn.plotting.surface._plotly_backend import (
    _configure_title,
    _get_camera_view_from_elevation_and_azimut,
    _get_camera_view_from_string_view,
    _get_view_plot_surf,
)
from nilearn.plotting.surface._utils import get_surface_backend
from nilearn.plotting.tests.test_engine_utils import check_colors
from nilearn.surface.surface import load_surf_data, load_surf_mesh

pytest.importorskip(
    "plotly",
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


def test_configure_title():
    """Test nilearn.plotting.surface._plotly_backend._configure_title."""
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


@pytest.mark.parametrize("full_view", EXPECTED_CAMERAS_PLOTLY)
def test_get_camera_view_from_string_view(full_view):
    """Test if
    nilearn.plotting.surface._plotly_backend._get_camera_view_from_string_view
    returns expected values.
    """
    hemi, view_name, (_, _), expected_camera_view = full_view
    camera_view_string = _get_camera_view_from_string_view(hemi, view_name)

    # Check each camera view parameter
    for k in ["center", "eye", "up"]:
        # Check camera view obtained from string view
        assert np.allclose(
            list(camera_view_string[k].values()),
            list(expected_camera_view[k].values()),
        )


@pytest.mark.parametrize("full_view", EXPECTED_CAMERAS_PLOTLY)
def test_get_camera_view_from_elev_azim(full_view):
    """Test if
    nilearn.plotting.surface._plotly_backend._get_camera_view_from_elevation_and_azimut
    returns expected values.
    """
    _, _, (elev, azim), expected_camera_view = full_view
    camera_view_elev_azim = _get_camera_view_from_elevation_and_azimut(
        (elev, azim)
    )
    # Check each camera view parameter
    for k in ["center", "eye", "up"]:
        # Check camera view obtained from elevation & azimut
        assert np.allclose(
            list(camera_view_elev_azim[k].values()),
            list(expected_camera_view[k].values()),
        )


@pytest.mark.parametrize("full_view", EXPECTED_CAMERAS_PLOTLY)
def test_get_view_plot_surf(full_view):
    """Test if
    nilearn.plotting.surface._plotly_backend.PlotlySurfaceBackend._get_view_plot_surf
    returns expected values.
    """
    hemi, view_name, (elev, azim), expected_camera_view = full_view
    camera_view = _get_view_plot_surf(hemi, view_name)
    camera_view_elev_azim = _get_view_plot_surf(hemi, (elev, azim))
    # Check each camera view parameter
    for k in ["center", "eye", "up"]:
        # Check default camera view
        assert np.allclose(
            list(camera_view[k].values()),
            list(expected_camera_view[k].values()),
        )
        # Check camera view obtained from elevation & azimut
        assert np.allclose(
            list(camera_view_elev_azim[k].values()),
            list(expected_camera_view[k].values()),
        )


@pytest.mark.parametrize("hemi,view", [("foo", "medial"), ("bar", "anterior")])
def test_get_view_plot_surf_hemisphere_errors(hemi, view):
    """Test
    nilearn.plotting.surface._plotly_backend.PlotlySurfaceBackend._get_view_plot_surf
    for invalid hemisphere values.
    """
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
    """Test
    nilearn.plotting.surface._plotly_backend.PlotlySurfaceBackend._get_view_plot_surf
    for invalid view values.
    """
    with pytest.raises(ValueError, match="Invalid view definition"):
        _get_view_plot_surf(hemi, view)


def test_one_mesh_info():
    """Test nilearn.plotting.surface._plotly_backend._one_mesh_info."""
    fsaverage = fetch_surf_fsaverage()
    mesh = fsaverage["pial_left"]
    surf_map = load_surf_data(fsaverage["sulc_left"])
    mesh = load_surf_mesh(mesh)
    backend = get_surface_backend("plotly")
    info = backend._one_mesh_info(
        surf_map, mesh, "90%", black_bg=True, bg_map=surf_map
    )
    assert {"_x", "_y", "_z", "_i", "_j", "_k"}.issubset(
        info["inflated_both"].keys()
    )
    assert len(decode(info["inflated_both"]["_x"], "<f4")) == len(surf_map)
    assert len(info["vertexcolor_both"]) == len(surf_map)
    cmax = np.max(np.abs(surf_map))
    assert (info["cmin"], info["cmax"]) == (-cmax, cmax)
    assert isinstance(info["cmax"], float)
    assert info["black_bg"]
    assert not info["full_brain_mesh"]
    check_colors(info["colorscale"])
