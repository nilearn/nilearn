"""Test nilearn.plotting.surface.html_surface functions."""

import json

import numpy as np
import pytest

from nilearn._utils.helpers import is_plotly_installed
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.exceptions import DimensionError
from nilearn.image import get_data, new_img_like
from nilearn.plotting.js_plotting_utils import decode
from nilearn.plotting.surface._utils import get_surface_backend
from nilearn.plotting.surface.html_surface import (
    _fill_html_template,
    _full_brain_info,
    view_img_on_surf,
    view_surf,
)
from nilearn.plotting.tests.test_engine_utils import check_colors
from nilearn.plotting.tests.test_js_plotting_utils import (
    check_html_surface_plots,
)
from nilearn.surface.surface import (
    check_mesh_is_fsaverage,
    load_surf_data,
    load_surf_mesh,
)


def test_check_mesh():
    mesh = check_mesh_is_fsaverage("fsaverage5")
    assert mesh is check_mesh_is_fsaverage(mesh)
    with pytest.raises(ValueError):
        check_mesh_is_fsaverage("fsaverage2")
    mesh.pop("pial_left")
    with pytest.raises(ValueError):
        check_mesh_is_fsaverage(mesh)
    with pytest.raises(TypeError):
        check_mesh_is_fsaverage(load_surf_mesh(mesh["pial_right"]))
    mesh = fetch_surf_fsaverage()
    assert mesh is check_mesh_is_fsaverage(mesh)


def test_full_brain_info(mni152_template_res_2):
    surfaces = fetch_surf_fsaverage()

    info = _full_brain_info(mni152_template_res_2, surfaces)
    check_colors(info["colorscale"])
    assert {
        "pial_left",
        "pial_right",
        "inflated_left",
        "inflated_right",
        "vertexcolor_left",
        "vertexcolor_right",
    }.issubset(info.keys())
    assert info["cmin"] == -info["cmax"]
    assert info["full_brain_mesh"]
    assert not info["black_bg"]
    assert isinstance(info["cmax"], float)
    json.dumps(info)
    for hemi in ["left", "right"]:
        mesh = load_surf_mesh(surfaces[f"pial_{hemi}"])
        assert len(info[f"vertexcolor_{hemi}"]) == len(mesh.coordinates)
        assert len(decode(info[f"inflated_{hemi}"]["_z"], "<f4")) == len(
            mesh.coordinates
        )
        assert len(decode(info[f"pial_{hemi}"]["_j"], "<i4")) == len(
            mesh.faces
        )


@pytest.mark.parametrize("backend_engine", ["plotly", "niivue"])
def test_fill_html_template(tmp_path, mni152_template_res_2, backend_engine):
    fsaverage = fetch_surf_fsaverage()
    surf_mesh = load_surf_mesh(fsaverage["pial_right"])
    surf_map = surf_mesh.coordinates[:, 0]
    bg_map = load_surf_data(fsaverage["sulc_right"])

    surf_mesh = load_surf_mesh(surf_mesh)
    backend = get_surface_backend(backend_engine)
    info = backend._one_mesh_info(
        surf_map=surf_map,
        surf_mesh=surf_mesh,
        threshold="90%",
        black_bg=True,
        bg_map=bg_map,
    )
    info["title"] = None

    html = _fill_html_template(info, engine=backend_engine)

    check_html_surface_plots(tmp_path, html, engine=backend_engine)

    info = _full_brain_info(mni152_template_res_2)
    info["title"] = None

    html = _fill_html_template(info, engine=backend_engine)

    check_html_surface_plots(tmp_path, html, engine=backend_engine)


@pytest.mark.single_process
@pytest.mark.parametrize("backend_engine", ["plotly", "niivue"])
def test_view_surf(tmp_path, rng, backend_engine):
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_right"])
    surf_map = mesh.coordinates[:, 0]

    html = view_surf(
        fsaverage["pial_right"],
        surf_map,
        fsaverage["sulc_right"],
        threshold="90%",
        engine=backend_engine,
    )
    check_html_surface_plots(
        tmp_path,
        html,
        title="Surface plot",
        engine=backend_engine,
    )

    html = view_surf(
        fsaverage["pial_right"],
        surf_map,
        fsaverage["sulc_right"],
        threshold=0.3,
        title="SOME_TITLE",
        engine=backend_engine,
    )
    check_html_surface_plots(
        tmp_path, html, title="SOME_TITLE", engine=backend_engine
    )

    html = view_surf(fsaverage["pial_right"], engine=backend_engine)
    check_html_surface_plots(tmp_path, html, engine=backend_engine)

    atlas = rng.integers(0, 10, size=len(mesh.coordinates))
    html = view_surf(
        fsaverage["pial_left"],
        atlas,
        symmetric_cmap=False,
        engine=backend_engine,
    )
    check_html_surface_plots(tmp_path, html, engine=backend_engine)

    html = view_surf(
        fsaverage["pial_right"],
        fsaverage["sulc_right"],
        threshold=None,
        cmap="Greys",
        engine=backend_engine,
    )
    check_html_surface_plots(tmp_path, html, engine=backend_engine)


def test_view_surf_errors():
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_right"])

    with pytest.raises(ValueError):
        view_surf(mesh, mesh.coordinates[::2, 0])

    with pytest.raises(ValueError):
        view_surf(
            mesh, mesh.coordinates[:, 0], bg_map=mesh.coordinates[::2, 0]
        )


@pytest.mark.skipif(
    not is_plotly_installed(),
    reason="This test requires plotly to be installed",
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"threshold": "92.3%"},
        {"threshold": 0, "surf_mesh": fetch_surf_fsaverage()},
        {"threshold": 0.4, "title": "SOME_TITLE"},
        {"threshold": 0.4, "cmap": "hot", "black_bg": True},
    ],
)
def test_view_img_on_surf(tmp_path, mni152_template_res_2, kwargs):
    """Check output of view_img_on_surf."""
    html = view_img_on_surf(mni152_template_res_2, **kwargs)
    check_html_surface_plots(tmp_path, html, title=kwargs.get("title", None))


@pytest.mark.skipif(
    not is_plotly_installed(),
    reason="This test requires plotly to be installed",
)
def test_view_img_on_surf_clipped_image(tmp_path, mni152_template_res_2):
    """Check output of view_img_on_surf with clipped input."""
    img_4d = new_img_like(
        mni152_template_res_2,
        get_data(mni152_template_res_2)[:, :, :, np.newaxis],
    )
    assert len(img_4d.shape) == 4
    np.clip(
        get_data(mni152_template_res_2),
        0,
        None,
        out=get_data(mni152_template_res_2),
    )

    html = view_img_on_surf(mni152_template_res_2, symmetric_cmap=False)

    check_html_surface_plots(tmp_path, html)

    html = view_img_on_surf(
        mni152_template_res_2,
        symmetric_cmap=False,
        vol_to_surf_kwargs={
            "n_samples": 1,
            "radius": 0.0,
            "interpolation": "nearest_most_frequent",
        },
    )
    check_html_surface_plots(tmp_path, html)


@pytest.mark.skipif(
    not is_plotly_installed(),
    reason="This test requires plotly to be installed",
)
@pytest.mark.thread_unsafe
def test_view_img_on_surf_input_as_file(img_3d_mni_as_file):
    view_img_on_surf(img_3d_mni_as_file)
    view_img_on_surf(str(img_3d_mni_as_file))


@pytest.mark.skipif(
    not is_plotly_installed(),
    reason="This test requires plotly to be installed",
)
def test_view_img_on_surf_errors(img_3d_mni):
    with pytest.raises(DimensionError):
        view_img_on_surf([img_3d_mni, img_3d_mni])


@pytest.mark.skipif(
    not is_plotly_installed(),
    reason="This test requires plotly to be installed",
)
@pytest.mark.parametrize("view", ["left", "right"])
def test_view_img_on_surf_view(tmp_path, mni152_template_res_2, view):
    """Smoke test for different views of view_img_on_surf."""
    html = view_img_on_surf(mni152_template_res_2, view=view)

    assert f', "view": "{view}"' in str(html)
    check_html_surface_plots(tmp_path, html)
