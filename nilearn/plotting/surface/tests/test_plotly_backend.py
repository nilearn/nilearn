import numpy as np
import pytest

from nilearn.plotting.surface._backend import get_surface_backend
from nilearn.plotting.surface._plotly_backend import (
    _configure_title,
    _get_camera_view_from_elevation_and_azimut,
    _get_camera_view_from_string_view,
    _get_view_plot_surf_plotly,
)
from nilearn.plotting.surface.tests.test_backend import BaseTestSurfaceBackend

ENGINE = "plotly"
pytest.importorskip(
    ENGINE,
    reason="Plotly is not installed. It is required to run the tests!",
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


@pytest.fixture
def engine():
    return ENGINE


class TestPlotlyBackend(BaseTestSurfaceBackend):
    """Tests PlotlyBackend class methods."""

    def test_plot_img_on_surf(self, engine, img_3d_mni):
        """Smoke test for PlotlyBackend.plot_img_on_surf."""
        with pytest.raises(NotImplementedError):
            get_surface_backend(engine).plot_img_on_surf(img_3d_mni)

    def test_plot_surf_contours(self, engine, surf_mesh, surf_mask_1d):
        """Smoke test for PlotlyBackend.plot_surf_contours."""
        with pytest.raises(NotImplementedError):
            get_surface_backend(engine).plot_surf_contours(
                surf_mesh, roi_map=surf_mask_1d
            )


@pytest.mark.parametrize("full_view", EXPECTED_CAMERAS_PLOTLY)
def test_get_view_plot_surf_plotly(full_view):
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


@pytest.mark.parametrize("hemi,view", [("foo", "medial"), ("bar", "anterior")])
def test_get_view_plot_surf_hemisphere_errors(hemi, view):
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        _get_view_plot_surf_plotly(hemi, view)


@pytest.mark.parametrize(
    "hemi,view",
    [
        ("left", "foo"),
        ("right", "bar"),
        ("both", "lateral"),
        ("both", "medial"),
        ("both", "bar"),
    ],
)
def test_get_view_plot_surf_view_errors(hemi, view):
    with pytest.raises(ValueError, match="Invalid view definition"):
        _get_view_plot_surf_plotly(hemi, view)


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
