import numpy as np
import pytest

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import (
    load_surf_data,
    load_surf_mesh,
)

from nilearn.plotting.surface._matplotlib import (
    MATPLOTLIB_VIEWS,
    _compute_facecolors,
    _get_bounds,
    _get_ticks,
    _get_view_plot_surf,
)

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


@pytest.mark.parametrize("hemi, views", MATPLOTLIB_VIEWS.items())
def test_get_view_plot_surf_matplotlib(hemi, views):
    for v in views:
        assert (
            _get_view_plot_surf(hemi, v) == EXPECTED_VIEW_MATPLOTLIB[hemi][v]
        )


@pytest.mark.parametrize(
    "data,expected",
    [
        (np.linspace(0, 1, 100), (0, 1)),
        (np.linspace(-0.7, -0.01, 40), (-0.7, -0.01)),
    ],
)
def test_get_bounds(data, expected):
    assert _get_bounds(data) == expected
    assert _get_bounds(data, vmin=0.2) == (0.2, expected[1])
    assert _get_bounds(data, vmax=0.8) == (expected[0], 0.8)
    assert _get_bounds(data, vmin=0.1, vmax=0.8) == (0.1, 0.8)


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
        ("both", "bar"),
    ],
)
def test_get_view_plot_surf_view_errors(hemi, view):
    with pytest.raises(ValueError, match="Invalid view definition"):
        _get_view_plot_surf(hemi, view)
