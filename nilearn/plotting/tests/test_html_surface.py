import re
import json
import base64
import tempfile
import os

import numpy as np
try:
    from lxml import etree
    LXML_INSTALLED = True
except ImportError:
    LXML_INSTALLED = False

from nilearn import datasets, surface
from nilearn.plotting import html_surface
from nilearn.datasets import fetch_surf_fsaverage5 as fetch_surf_fsaverage


# Note: html output by view_surf and view_img_on_surf
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _get_img():
    return datasets.fetch_localizer_button_task()['tmaps'][0]


def _normalize_ws(text):
    return re.sub(r'\s+', ' ', text)


def test_add_js_lib():
    html = html_surface.HTML_TEMPLATE
    cdn = html_surface.add_js_lib(html, embed_js=False)
    assert _normalize_ws("""<script
    src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js">
    </script>
    <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
    """) in _normalize_ws(cdn)
    inline = _normalize_ws(html_surface.add_js_lib(html, embed_js=True))
    assert _normalize_ws("""/*! jQuery v3.3.1 | (c) JS Foundation and other
                            contributors | jquery.org/license */""") in inline
    assert _normalize_ws("""**
                            * plotly.js (gl3d - minified) v1.38.3
                            * Copyright 2012-2018, Plotly, Inc.
                            * All rights reserved.
                            * Licensed under the MIT license
                            */ """) in inline


def _check_colors(colors_json):
    colors = json.loads(colors_json)
    assert len(colors) == 100
    val, cstring = zip(*colors)
    assert np.allclose(np.linspace(0, 1, 100), val, atol=1e-3)
    assert val[0] == 0
    assert val[-1] == 1
    for cs in cstring:
        assert re.match(r'rgb\(\d+, \d+, \d+\)', cs)
    return val, cstring


def test_colorscale():
    cmap = 'jet'
    values = np.linspace(-13, -1.5, 20)
    threshold = None
    (colors_json, abs_max, new_cmap, norm, abs_threshold
     ) = html_surface.colorscale(cmap, values, threshold)
    _check_colors(colors_json)
    assert abs_max == 13
    assert new_cmap.N == 256
    assert (norm.vmax, norm.vmin) == (13, -13)
    assert abs_threshold is None
    threshold = 0
    (colors_json, abs_max, new_cmap, norm, abs_threshold
     ) = html_surface.colorscale(cmap, values, threshold)
    _check_colors(colors_json)
    assert abs_max == 13
    assert new_cmap.N == 256
    assert (norm.vmax, norm.vmin) == (13, -13)
    assert abs_threshold == 1.5
    threshold = 100
    (colors_json, abs_max, new_cmap, norm, abs_threshold
     ) = html_surface.colorscale(cmap, values, threshold)
    _check_colors(colors_json)
    assert abs_max == 13
    assert new_cmap.N == 256
    assert (norm.vmax, norm.vmin) == (13, -13)
    assert abs_threshold == 13
    threshold = 50
    (colors_json, abs_max, new_cmap, norm, abs_threshold
     ) = html_surface.colorscale(cmap, values, threshold)
    val, cstring = _check_colors(colors_json)
    assert cstring[50] == 'rgb(127, 127, 127)'
    assert abs_max == 13
    assert new_cmap.N == 256
    assert (norm.vmax, norm.vmin) == (13, -13)
    assert np.allclose(abs_threshold, 7.25)


def _test_encode():
    for dtype in ['<f4', '<i4', '>f4', '>i4']:
        a = np.arange(10, dtype=dtype)
        encoded = html_surface._encode(a)
        decoded = base64.b64decode(encoded.encode('utf-8'))
        b = np.frombuffer(decoded, dtype=dtype)
        assert np.allclose(html_surface._decode(encoded), b)
        assert np.allclose(a, b)


def test_to_plotly():
    fsaverage = fetch_surf_fsaverage()
    print(fsaverage)
    coord, triangles = surface.load_surf_mesh(fsaverage['pial_left'])
    plotly = html_surface.to_plotly(fsaverage['pial_left'])
    for i, key in enumerate(['_x', '_y', '_z']):
        assert np.allclose(
            html_surface._decode(plotly[key], '<f4'), coord[:, i])
    for i, key in enumerate(['_i', '_j', '_k']):
        assert np.allclose(
            html_surface._decode(plotly[key], '<i4'), triangles[:, i])


def test_to_color_strings():
    colors = [[0, 0, 1], [1, 0, 0], [.5, .5, .5]]
    as_str = html_surface._to_color_strings(colors)
    assert as_str == ['#0000ff', '#ff0000', '#7f7f7f']


def test_get_vertexcolor():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_left'])
    surf_map = np.arange(len(mesh[0]))
    colors, cmax, cmap, norm, abs_threshold = html_surface.colorscale(
        'jet', surf_map, 10)
    vertexcolors = html_surface._get_vertexcolor(
        surf_map, cmap, norm, abs_threshold, fsaverage['sulc_left'])
    assert len(vertexcolors) == len(mesh[0])
    vertexcolors = html_surface._get_vertexcolor(
        surf_map, cmap, norm, abs_threshold)
    assert len(vertexcolors) == len(mesh[0])


def test_one_mesh_info():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    info, col = html_surface.one_mesh_info(
        surf_map, fsaverage['pial_right'], 90, black_bg=True,
        bg_map=fsaverage['sulc_right'])
    assert {'_x', '_y', '_z', '_i', '_j', '_k'}.issubset(
        info['inflated_left'].keys())
    assert len(html_surface._decode(
        info['inflated_left']['_x'], '<f4')) == len(surf_map)
    assert len(info['vertexcolor_left']) == len(surf_map)
    cmax = np.max(np.abs(surf_map))
    assert (info['cmin'], info['cmax']) == (-cmax, cmax)
    assert type(info['cmax']) == float
    json.dumps(info)
    assert info['black_bg']
    assert not info['full_brain_mesh']
    _check_colors(col)


def test_full_brain_info():
    fsaverage = fetch_surf_fsaverage()
    img = _get_img()
    info, colors = html_surface.full_brain_info(img)
    _check_colors(colors)
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
        assert len(html_surface._decode(
            info['inflated_{}'.format(hemi)]['_z'], '<f4')) == len(mesh[0])
        assert len(html_surface._decode(
            info['pial_{}'.format(hemi)]['_j'], '<i4')) == len(mesh[1])


def _check_html(html):
    fd, tmpfile = tempfile.mkstemp()
    try:
        os.close(fd)
        html.save_as_html(tmpfile)
        with open(tmpfile) as f:
            saved = f.read()
        assert saved == html.standalone()
    finally:
        os.remove(tmpfile)
    assert "INSERT" not in html.html
    assert html.standalone() == html.html
    assert html._repr_html_() == html.iframe()
    assert str(html) == html.standalone()
    assert '<meta charset="UTF-8" />' in str(html)
    if not LXML_INSTALLED:
        return
    root = etree.HTML(html.html)
    head = root.find('head')
    assert len(head.findall('script')) == 4
    body = root.find('body')
    div = body.find('div')
    assert ('id', 'surface-plot') in div.items()
    selects = body.findall('select')
    assert len(selects) == 3
    hemi = selects[0]
    assert ('id', 'select-hemisphere') in hemi.items()
    assert len(hemi.findall('option')) == 2
    kind = selects[1]
    assert ('id', 'select-kind') in kind.items()
    assert len(kind.findall('option')) == 2
    view = selects[2]
    assert ('id', 'select-view') in view.items()
    assert len(view.findall('option')) == 7


def test_fill_html_template():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    img = _get_img()
    info, colors = html_surface.one_mesh_info(
        surf_map, fsaverage['pial_right'], 90, black_bg=True,
        bg_map=fsaverage['sulc_right'])
    html = html_surface._fill_html_template(info, colors, embed_js=False)
    _check_html(html)
    assert "jquery.min.js" in html.html
    info, colors = html_surface.full_brain_info(img)
    html = html_surface._fill_html_template(info, colors)
    _check_html(html)
    assert "* plotly.js (gl3d - minified) v1.38.3" in html.html


def test_view_surf():
    fsaverage = fetch_surf_fsaverage()
    mesh = surface.load_surf_mesh(fsaverage['pial_right'])
    surf_map = mesh[0][:, 0]
    html = html_surface.view_surf(fsaverage['pial_right'], surf_map,
                                  fsaverage['sulc_right'], 90)
    _check_html(html)


def test_view_img_on_surf():
    img = _get_img()
    html = html_surface.view_img_on_surf(img, threshold=95)
    _check_html(html)
