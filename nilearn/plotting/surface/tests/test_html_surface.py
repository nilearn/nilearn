"""Test nilearn.plotting.surface.html_surface functions."""

import json

import numpy as np
import pytest

from nilearn import datasets, image
from nilearn._utils.exceptions import DimensionError
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.image import get_data
from nilearn.plotting.js_plotting_utils import decode
from nilearn.plotting.surface.html_surface import (
    _fill_html_template,
    _full_brain_info,
    _one_mesh_info,
    full_brain_info,
    one_mesh_info,
    view_img_on_surf,
    view_surf,
)
from nilearn.plotting.tests.test_engine_utils import check_colors
from nilearn.plotting.tests.test_js_plotting_utils import check_html
from nilearn.surface.surface import (
    check_mesh_is_fsaverage,
    load_surf_data,
    load_surf_mesh,
)


@pytest.fixture(scope="session")
def mni152_template_res_2():
    return datasets.load_mni152_template(resolution=2)


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
    mesh = datasets.fetch_surf_fsaverage()
    assert mesh is check_mesh_is_fsaverage(mesh)


def test_one_mesh_info():
    fsaverage = datasets.fetch_surf_fsaverage()
    mesh = fsaverage["pial_left"]
    surf_map = load_surf_data(fsaverage["sulc_left"])
    mesh = load_surf_mesh(mesh)
    info = _one_mesh_info(
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
    json.dumps(info)
    assert info["black_bg"]
    assert not info["full_brain_mesh"]
    check_colors(info["colorscale"])

    # TODO (nilearn >= 0.13.0)
    with pytest.warns(
        DeprecationWarning,
        match="one_mesh_info is a private function and is renamed "
        "to _one_mesh_info. Using the deprecated name will "
        "raise an error in release 0.13",
    ):
        one_mesh_info(surf_map, mesh)


def test_full_brain_info(mni152_template_res_2):
    surfaces = datasets.fetch_surf_fsaverage()

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

    # TODO (nilearn >= 0.13.0)
    with pytest.warns(
        DeprecationWarning,
        match="full_brain_info is a private function and is renamed to "
        "_full_brain_info. Using the deprecated name will raise an error "
        "in release 0.13",
    ):
        full_brain_info(mni152_template_res_2)


def test_fill_html_template(tmp_path, mni152_template_res_2):
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_right"])
    surf_map = mesh.coordinates[:, 0]
    info = _one_mesh_info(
        surf_map,
        fsaverage["pial_right"],
        "90%",
        black_bg=True,
        bg_map=fsaverage["sulc_right"],
    )
    info["title"] = None
    html = _fill_html_template(info, embed_js=False)
    check_html(tmp_path, html)
    assert "jquery.min.js" in html.html
    info = _full_brain_info(mni152_template_res_2)
    info["title"] = None
    html = _fill_html_template(info)
    check_html(tmp_path, html)
    assert "* plotly.js (gl3d - minified) v1." in html.html


def test_view_surf(tmp_path, rng):
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_right"])
    surf_map = mesh.coordinates[:, 0]
    html = view_surf(
        fsaverage["pial_right"], surf_map, fsaverage["sulc_right"], "90%"
    )
    check_html(tmp_path, html, title="Surface plot")
    html = view_surf(
        fsaverage["pial_right"],
        surf_map,
        fsaverage["sulc_right"],
        0.3,
        title="SOME_TITLE",
    )
    check_html(tmp_path, html, title="SOME_TITLE")
    assert "SOME_TITLE" in html.html
    html = view_surf(fsaverage["pial_right"])
    check_html(tmp_path, html)
    atlas = rng.integers(0, 10, size=len(mesh.coordinates))
    html = view_surf(fsaverage["pial_left"], atlas, symmetric_cmap=False)
    check_html(tmp_path, html)
    html = view_surf(
        fsaverage["pial_right"],
        fsaverage["sulc_right"],
        threshold=None,
        cmap="Greys",
    )
    check_html(tmp_path, html)
    with pytest.raises(ValueError):
        view_surf(mesh, mesh.coordinates[::2, 0])
    with pytest.raises(ValueError):
        view_surf(
            mesh, mesh.coordinates[:, 0], bg_map=mesh.coordinates[::2, 0]
        )


def test_view_img_on_surf(tmp_path, mni152_template_res_2):
    html = view_img_on_surf(mni152_template_res_2, threshold="92.3%")
    check_html(tmp_path, html)

    surfaces = datasets.fetch_surf_fsaverage()
    html = view_img_on_surf(
        mni152_template_res_2, threshold=0, surf_mesh=surfaces
    )
    check_html(tmp_path, html)

    html = view_img_on_surf(
        mni152_template_res_2, threshold=0.4, title="SOME_TITLE"
    )
    assert "SOME_TITLE" in html.html
    check_html(tmp_path, html)

    html = view_img_on_surf(
        mni152_template_res_2, threshold=0.4, cmap="hot", black_bg=True
    )
    check_html(tmp_path, html)

    img_4d = image.new_img_like(
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
    check_html(tmp_path, html)

    html = view_img_on_surf(
        mni152_template_res_2,
        symmetric_cmap=False,
        vol_to_surf_kwargs={
            "n_samples": 1,
            "radius": 0.0,
            "interpolation": "nearest",
        },
    )
    check_html(tmp_path, html)


def test_view_img_on_surf_input_as_file(img_3d_mni_as_file):
    view_img_on_surf(img_3d_mni_as_file)
    view_img_on_surf(str(img_3d_mni_as_file))


def test_view_img_on_surf_errors(img_3d_mni):
    with pytest.raises(DimensionError):
        view_img_on_surf([img_3d_mni, img_3d_mni])
