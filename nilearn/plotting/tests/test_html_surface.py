import json

import numpy as np
from numpy.testing import assert_raises

from nilearn import datasets, surface, image
from nilearn.plotting import html_surface
from nilearn.plotting.js_plotting_utils import decode
from nilearn.datasets import fetch_surf_fsaverage
from nilearn._utils.exceptions import DimensionError

from .test_js_plotting_utils import check_colors, check_html


def _get_img():
    return datasets.load_mni152_template()


def test_get_vertexcolor():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_left'])
    surf_map = np.arange(len(mesh[0]))
    colors = html_surface.colorscale('jet', surf_map, 10)
    vertexcolors = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'], colors['abs_threshold'],
        fsaverage['sulc_left'])
    assert len(vertexcolors) == len(mesh[0])
    vertexcolors = html_surface._get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'], colors['abs_threshold'])
    assert len(vertexcolors) == len(mesh[0])


def test_check_mesh():
    mesh = html_surface._check_mesh('fsaverage5')
    assert mesh is html_surface._check_mesh(mesh)
    assert_raises(ValueError, html_surface._check_mesh, 'fsaverage3')
    mesh.pop('pial_left')
    assert_raises(ValueError, html_surface._check_mesh, mesh)
    assert_raises(TypeError, html_surface._check_mesh,
                  surface.load_surf_mesh(mesh['pial_right']))


def test_one_mesh_info():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    info = html_surface.one_mesh_info(
        surf_map, fsaverage['pial_right'], '90%', black_bg=True,
        bg_map=fsaverage['sulc_right'])
    assert {'_x', '_y', '_z', '_i', '_j', '_k'}.issubset(
        info['inflated_left'].keys())
    assert len(decode(
        info['inflated_left']['_x'], '<f4')) == len(surf_map)
    assert len(info['vertexcolor_left']) == len(surf_map)
    cmax = np.max(np.abs(surf_map))
    assert (info['cmin'], info['cmax']) == (-cmax, cmax)
    assert type(info['cmax']) == float
    json.dumps(info)
    assert info['black_bg']
    assert not info['full_brain_mesh']
    check_colors(info['colorscale'])


def test_full_brain_info():
    fsaverage = fetch_surf_fsaverage()
    img = _get_img()
    info = html_surface.full_brain_info(img)
    check_colors(info['colorscale'])
    assert {'pial_left', 'pial_right',
            'inflated_left', 'inflated_right',
            'vertexcolor_left', 'vertexcolor_right'}.issubset(info.keys())
    assert info['cmin'] == - info['cmax']
    assert info['full_brain_mesh']
    assert not info['black_bg']
    assert type(info['cmax']) == float
    json.dumps(info)
    for hemi in ['left', 'right']:
        mesh = surface.load_surf_mesh(fsaverage['pial_{}'.format(hemi)])
        assert len(info['vertexcolor_{}'.format(hemi)]) == len(mesh[0])
        assert len(decode(
            info['inflated_{}'.format(hemi)]['_z'], '<f4')) == len(mesh[0])
        assert len(decode(
            info['pial_{}'.format(hemi)]['_j'], '<i4')) == len(mesh[1])


def test_fill_html_template():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    img = _get_img()
    info = html_surface.one_mesh_info(
        surf_map, fsaverage['pial_right'], '90%', black_bg=True,
        bg_map=fsaverage['sulc_right'])
    html = html_surface._fill_html_template(info, embed_js=False)
    check_html(html)
    assert "jquery.min.js" in html.html
    info = html_surface.full_brain_info(img)
    html = html_surface._fill_html_template(info)
    check_html(html)
    assert "* plotly.js (gl3d - minified) v1.38.3" in html.html


def test_view_surf():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    html = html_surface.view_surf(fsaverage['pial_right'], surf_map,
                                  fsaverage['sulc_right'], '90%')
    check_html(html)
    html = html_surface.view_surf(fsaverage['pial_right'], surf_map,
                                  fsaverage['sulc_right'], .3)
    check_html(html)
    html = html_surface.view_surf(fsaverage['pial_right'])
    check_html(html)
    destrieux = datasets.fetch_atlas_surf_destrieux()['map_left']
    html = html_surface.view_surf(
        fsaverage['pial_left'], destrieux, symmetric_cmap=False)
    check_html(html)
    assert_raises(ValueError, html_surface.view_surf, mesh, mesh[0][::2, 0])
    assert_raises(ValueError, html_surface.view_surf, mesh, mesh[0][:, 0],
                  bg_map=mesh[0][::2, 0])


def test_view_img_on_surf():
    img = _get_img()
    fsaverage = dict(fetch_surf_fsaverage())
    html = html_surface.view_img_on_surf(img, threshold='92.3%')
    check_html(html)
    html = html_surface.view_img_on_surf(img, threshold=0, surf_mesh=fsaverage)
    check_html(html)
    html = html_surface.view_img_on_surf(img, threshold=.4)
    check_html(html)
    html = html_surface.view_img_on_surf(
        img, threshold=.4, cmap='hot', black_bg=True)
    check_html(html)
    html = html_surface.view_img_on_surf(img, surf_mesh='fsaverage')
    check_html(html)
    assert_raises(DimensionError, html_surface.view_img_on_surf, [img, img])
    img_4d = image.new_img_like(img, img.get_data()[:, :, :, np.newaxis])
    assert len(img_4d.shape) == 4
    html = html_surface.view_img_on_surf(img, threshold='92.3%')
    check_html(html)
