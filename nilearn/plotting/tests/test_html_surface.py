import re
import json
import base64

import numpy as np

from nilearn import datasets, surface
from nilearn.plotting import html_surface


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
    fsaverage = datasets.fetch_surf_fsaverage5()
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
