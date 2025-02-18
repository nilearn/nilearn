import numpy as np
import pytest

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting.html.js_plotting_utils import colorscale
from nilearn.plotting.surface._utils import get_vertexcolor
from nilearn.surface.surface import (
    load_surf_data,
    load_surf_mesh,
)


def test_get_vertexcolor():
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_left"])
    surf_map = np.arange(len(mesh.coordinates))
    colors = colorscale("jet", surf_map, 10)
    vertexcolors = get_vertexcolor(
        surf_map,
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
        bg_map=fsaverage["sulc_left"],
    )
    assert len(vertexcolors) == len(mesh.coordinates)
    vertexcolors = get_vertexcolor(
        surf_map,
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
    )
    assert len(vertexcolors) == len(mesh.coordinates)
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(load_surf_data(fsaverage["curv_left"]))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert bg_min < 0 or bg_max > 1
    vertexcolors_auto_normalized = get_vertexcolor(
        surf_map,
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
        bg_map=bg_map,
    )
    assert len(vertexcolors_auto_normalized) == len(mesh.coordinates)
    # Manually set values of background map between 0 and 1
    bg_map_normalized = (bg_map - bg_min) / (bg_max - bg_min)
    assert np.min(bg_map_normalized) == 0 and np.max(bg_map_normalized) == 1
    vertexcolors_manually_normalized = get_vertexcolor(
        surf_map,
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
        bg_map=bg_map_normalized,
    )
    assert len(vertexcolors_manually_normalized) == len(mesh.coordinates)
    assert vertexcolors_manually_normalized == vertexcolors_auto_normalized
    # Scale background map between 0.25 and 0.75
    bg_map_scaled = bg_map_normalized / 2 + 0.25
    assert np.min(bg_map_scaled) == 0.25 and np.max(bg_map_scaled) == 0.75
    vertexcolors_manually_rescaled = get_vertexcolor(
        surf_map,
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
        bg_map=bg_map_scaled,
    )
    assert len(vertexcolors_manually_rescaled) == len(mesh.coordinates)
    assert vertexcolors_manually_rescaled != vertexcolors_auto_normalized
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The `darkness` parameter will be deprecated in release 0.13. "
            "We recommend setting `darkness` to None"
        ),
    ):
        vertexcolors = get_vertexcolor(
            surf_map,
            colors["cmap"],
            colors["norm"],
            absolute_threshold=colors["abs_threshold"],
            bg_map=bg_map,
            darkness=0.5,
        )
