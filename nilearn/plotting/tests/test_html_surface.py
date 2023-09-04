import json

import numpy as np
import pytest

from nilearn import datasets, image, surface
from nilearn._utils.exceptions import DimensionError
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.image import get_data
from nilearn.plotting import html_surface
from nilearn.plotting.js_plotting_utils import decode

from .test_js_plotting_utils import check_colors, check_html


def _get_img():
    return datasets.load_mni152_template(resolution=2)


def test_get_vertexcolor():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_left'])
    surf_map = np.arange(len(mesh[0]))
    colors = html_surface.colorscale('jet', surf_map, 10)
    vertexcolors = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        absolute_threshold=colors['abs_threshold'],
        bg_map=fsaverage['sulc_left'])
    assert len(vertexcolors) == len(mesh[0])
    vertexcolors = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        absolute_threshold=colors['abs_threshold'])
    assert len(vertexcolors) == len(mesh[0])
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(surface.load_surf_data(fsaverage['curv_left']))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert (bg_min < 0 or bg_max > 1)
    vertexcolors_auto_normalized = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        absolute_threshold=colors['abs_threshold'],
        bg_map=bg_map)
    assert len(vertexcolors_auto_normalized) == len(mesh[0])
    # Manually set values of background map between 0 and 1
    bg_map_normalized = (bg_map - bg_min) / (bg_max - bg_min)
    assert np.min(bg_map_normalized) == 0 and np.max(bg_map_normalized) == 1
    vertexcolors_manually_normalized = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        absolute_threshold=colors['abs_threshold'],
        bg_map=bg_map_normalized)
    assert len(vertexcolors_manually_normalized) == len(mesh[0])
    assert vertexcolors_manually_normalized == vertexcolors_auto_normalized
    # Scale background map between 0.25 and 0.75
    bg_map_scaled = bg_map_normalized / 2 + 0.25
    assert np.min(bg_map_scaled) == 0.25 and np.max(bg_map_scaled) == 0.75
    vertexcolors_manually_rescaled = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        absolute_threshold=colors['abs_threshold'],
        bg_map=bg_map_scaled)
    assert len(vertexcolors_manually_rescaled) == len(mesh[0])
    assert vertexcolors_manually_rescaled != vertexcolors_auto_normalized
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The `darkness` parameter will be deprecated in release 0.13. "
            "We recommend setting `darkness` to None"
        ),
    ):
        vertexcolors = html_surface._get_vertexcolor(
            surf_map, colors['cmap'], colors['norm'],
            absolute_threshold=colors['abs_threshold'],
            bg_map=bg_map, darkness=0.5)


def test_check_mesh():
    mesh = html_surface._check_mesh('fsaverage5')
    assert mesh is html_surface._check_mesh(mesh)
    with pytest.raises(ValueError):
        html_surface._check_mesh('fsaverage2')
    mesh.pop('pial_left')
    with pytest.raises(ValueError):
        html_surface._check_mesh(mesh)
    with pytest.raises(TypeError):
        html_surface._check_mesh(surface.load_surf_mesh(mesh['pial_right']))
    mesh = datasets.fetch_surf_fsaverage()
    assert mesh is html_surface._check_mesh(mesh)


def test_one_mesh_info():
    fsaverage = datasets.fetch_surf_fsaverage()
    mesh = fsaverage["pial_left"]
    surf_map = surface.load_surf_data(fsaverage["sulc_left"])
    mesh = surface.load_surf_mesh(mesh)
    info = html_surface._one_mesh_info(
        surf_map, mesh, '90%', black_bg=True,
        bg_map=surf_map)
    assert {'_x', '_y', '_z', '_i', '_j', '_k'}.issubset(
        info['inflated_left'].keys())
    assert len(decode(
        info['inflated_left']['_x'], '<f4')) == len(surf_map)
    assert len(info['vertexcolor_left']) == len(surf_map)
    cmax = np.max(np.abs(surf_map))
    assert (info['cmin'], info['cmax']) == (-cmax, cmax)
    assert isinstance(info['cmax'], float)
    json.dumps(info)
    assert info['black_bg']
    assert not info['full_brain_mesh']
    check_colors(info['colorscale'])

    with pytest.warns(
        DeprecationWarning,
        match="one_mesh_info is a private function and is renamed "
        "to _one_mesh_info. Using the deprecated name will "
        "raise an error in release 0.13",
    ):
        html_surface.one_mesh_info(surf_map, mesh)


def test_full_brain_info():
    surfaces = datasets.fetch_surf_fsaverage()
    img = _get_img()
    info = html_surface._full_brain_info(img, surfaces)
    check_colors(info['colorscale'])
    assert {'pial_left', 'pial_right',
            'inflated_left', 'inflated_right',
            'vertexcolor_left', 'vertexcolor_right'}.issubset(info.keys())
    assert info['cmin'] == - info['cmax']
    assert info['full_brain_mesh']
    assert not info['black_bg']
    assert isinstance(info['cmax'], float)
    json.dumps(info)
    for hemi in ['left', 'right']:
        mesh = surface.load_surf_mesh(surfaces[f'pial_{hemi}'])
        assert len(info[f'vertexcolor_{hemi}']) == len(mesh[0])
        assert len(decode(
            info[f'inflated_{hemi}']['_z'], '<f4')) == len(mesh[0])
        assert len(decode(
            info[f'pial_{hemi}']['_j'], '<i4')) == len(mesh[1])

    with pytest.warns(
        DeprecationWarning,
        match="full_brain_info is a private function and is renamed to "
        "_full_brain_info. Using the deprecated name will raise an error "
        "in release 0.13",
    ):
        html_surface.full_brain_info(img)


def test_fill_html_template():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    img = _get_img()
    info = html_surface._one_mesh_info(
        surf_map, fsaverage['pial_right'], '90%', black_bg=True,
        bg_map=fsaverage['sulc_right'])
    info["title"] = None
    html = html_surface._fill_html_template(info, embed_js=False)
    check_html(html)
    assert "jquery.min.js" in html.html
    info = html_surface._full_brain_info(img)
    info["title"] = None
    html = html_surface._fill_html_template(info)
    check_html(html)
    assert "* plotly.js (gl3d - minified) v1." in html.html


def test_view_surf():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    html = html_surface.view_surf(fsaverage['pial_right'], surf_map,
                                  fsaverage['sulc_right'], '90%')
    check_html(html, title="Surface plot")
    html = html_surface.view_surf(fsaverage['pial_right'], surf_map,
                                  fsaverage['sulc_right'], .3,
                                  title="SOME_TITLE")
    check_html(html, title="SOME_TITLE")
    assert "SOME_TITLE" in html.html
    html = html_surface.view_surf(fsaverage['pial_right'])
    check_html(html)
    atlas = np.random.RandomState(42).randint(0, 10, size=len(mesh[0]))
    html = html_surface.view_surf(
        fsaverage['pial_left'], atlas, symmetric_cmap=False)
    check_html(html)
    html = html_surface.view_surf(fsaverage['pial_right'],
                                  fsaverage['sulc_right'],
                                  threshold=None, cmap='Greys')
    check_html(html)
    with pytest.raises(ValueError):
        html_surface.view_surf(mesh, mesh[0][::2, 0])
    with pytest.raises(ValueError):
        html_surface.view_surf(mesh, mesh[0][:, 0],
                               bg_map=mesh[0][::2, 0])


def test_view_img_on_surf():
    img = _get_img()
    surfaces = datasets.fetch_surf_fsaverage()
    html = html_surface.view_img_on_surf(img, threshold='92.3%')
    check_html(html)
    html = html_surface.view_img_on_surf(img, threshold=0, surf_mesh=surfaces)
    check_html(html)
    html = html_surface.view_img_on_surf(img, threshold=.4, title="SOME_TITLE")
    assert "SOME_TITLE" in html.html
    check_html(html)
    html = html_surface.view_img_on_surf(
        img, threshold=.4, cmap='hot', black_bg=True)
    check_html(html)
    with pytest.raises(DimensionError):
        html_surface.view_img_on_surf([img, img])
    img_4d = image.new_img_like(img, get_data(img)[:, :, :, np.newaxis])
    assert len(img_4d.shape) == 4
    html = html_surface.view_img_on_surf(img, threshold='92.3%')
    check_html(html)
    np.clip(get_data(img), 0, None, out=get_data(img))
    html = html_surface.view_img_on_surf(img, symmetric_cmap=False)
    check_html(html)
    html = html_surface.view_img_on_surf(img, symmetric_cmap=False,
                                         vol_to_surf_kwargs={
                                             "n_samples": 1,
                                             "radius": 0.,
                                             "interpolation": "nearest"})
    check_html(html)
