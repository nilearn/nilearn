import pytest

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import html_surface
from nilearn.plotting.html_surface import (
    _matplotlib_cm_to_niivue_cm,
)
from nilearn.surface.surface import load_surf_mesh


def test_niivue_smoke():
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_right"])
    surf_map = mesh[0][:, 0]
    html_surface.view_surf(
        fsaverage["pial_right"],
        surf_map,
        fsaverage["sulc_right"],
        threshold="90%",
        engine="niivue",
        hemi="left",
    )


def test_view_surf_errors():
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_right"])

    with pytest.raises(ValueError):
        html_surface.view_surf(
            mesh, mesh.coordinates[::2, 0], engine="niivue", hemi="left"
        )

    with pytest.raises(ValueError):
        html_surface.view_surf(
            mesh,
            mesh.coordinates[:, 0],
            bg_map=mesh.coordinates[::2, 0],
            engine="niivue",
            hemi="left",
        )


def test_matplotlib_cm_to_niivue_cm():
    with pytest.warns(
        UserWarning, match="'cmap' must be a str or a Colormap. Got"
    ):
        niivue_cmap = _matplotlib_cm_to_niivue_cm(None)
        assert niivue_cmap is None

    with pytest.warns(
        UserWarning, match="'cmap' must be a str or a Colormap. Got"
    ):
        niivue_cmap = _matplotlib_cm_to_niivue_cm(1)
        assert niivue_cmap is None

    with pytest.raises(ValueError, match="spec"):
        _matplotlib_cm_to_niivue_cm("foo")
