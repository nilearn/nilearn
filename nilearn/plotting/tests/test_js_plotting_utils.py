import os
import re
import base64
import webbrowser
import time
import tempfile

import numpy as np
import matplotlib
from numpy.testing import assert_warns, assert_no_warnings
try:
    from lxml import etree
    LXML_INSTALLED = True
except ImportError:
    LXML_INSTALLED = False

from nilearn.plotting import js_plotting_utils
from nilearn import surface
from nilearn.datasets import fetch_surf_fsaverage


# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _normalize_ws(text):
    return re.sub(r'\s+', ' ', text)


def test_add_js_lib():
    html = js_plotting_utils.get_html_template('surface_plot_template.html')
    cdn = js_plotting_utils.add_js_lib(html, embed_js=False)
    assert "decodeBase64" in cdn
    assert _normalize_ws("""<script
    src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js">
    </script>
    <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
    """) in _normalize_ws(cdn)
    inline = _normalize_ws(js_plotting_utils.add_js_lib(html, embed_js=True))
    assert _normalize_ws("""/*! jQuery v3.3.1 | (c) JS Foundation and other
                            contributors | jquery.org/license */""") in inline
    assert _normalize_ws("""**
                            * plotly.js (gl3d - minified) v1.38.3
                            * Copyright 2012-2018, Plotly, Inc.
                            * All rights reserved.
                            * Licensed under the MIT license
                            */ """) in inline
    assert "decodeBase64" in inline


def check_colors(colors):
    assert len(colors) == 100
    val, cstring = zip(*colors)
    assert np.allclose(np.linspace(0, 1, 100), val, atol=1e-3)
    assert val[0] == 0
    assert val[-1] == 1
    for cs in cstring:
        assert re.match(r'rgb\(\d+, \d+, \d+\)', cs)
    return val, cstring


def test_colorscale_no_threshold():
    cmap = 'jet'
    values = np.linspace(-13, -1.5, 20)
    threshold = None
    colors = js_plotting_utils.colorscale(cmap, values, threshold)
    check_colors(colors['colors'])
    assert (colors['vmin'], colors['vmax']) == (-13, 13)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (13, -13)
    assert colors['abs_threshold'] is None


def test_colorscale_threshold_0():
    cmap = 'jet'
    values = np.linspace(-13, -1.5, 20)
    threshold = '0%'
    colors = js_plotting_utils.colorscale(cmap, values, threshold)
    check_colors(colors['colors'])
    assert (colors['vmin'], colors['vmax']) == (-13, 13)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (13, -13)
    assert colors['abs_threshold'] == 1.5
    assert colors['symmetric_cmap']


def test_colorscale_threshold_99():
    cmap = 'jet'
    values = np.linspace(-13, -1.5, 20)
    threshold = '99%'
    colors = js_plotting_utils.colorscale(cmap, values, threshold)
    check_colors(colors['colors'])
    assert (colors['vmin'], colors['vmax']) == (-13, 13)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (13, -13)
    assert colors['abs_threshold'] == 13
    assert colors['symmetric_cmap']


def test_colorscale_threshold_50():
    cmap = 'jet'
    values = np.linspace(-13, -1.5, 20)
    threshold = '50%'
    colors = js_plotting_utils.colorscale(cmap, values, threshold)
    val, cstring = check_colors(colors['colors'])
    assert cstring[50] == 'rgb(127, 127, 127)'
    assert (colors['vmin'], colors['vmax']) == (-13, 13)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (13, -13)
    assert np.allclose(colors['abs_threshold'], 7.55, 2)
    assert colors['symmetric_cmap']


def test_colorscale_absolute_threshold():
    cmap = 'jet'
    values = np.linspace(-13, -1.5, 20)
    threshold = 7.25
    colors = js_plotting_utils.colorscale(cmap, values, threshold)
    val, cstring = check_colors(colors['colors'])
    assert cstring[50] == 'rgb(127, 127, 127)'
    assert (colors['vmin'], colors['vmax']) == (-13, 13)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (13, -13)
    assert np.allclose(colors['abs_threshold'], 7.25)
    assert colors['symmetric_cmap']


def test_colorscale_asymmetric_cmap():
    cmap = 'jet'
    values = np.arange(15)
    colors = js_plotting_utils.colorscale(cmap, values, symmetric_cmap=False)
    assert (colors['vmin'], colors['vmax']) == (0, 14)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (14, 0)
    assert not colors['symmetric_cmap']
    values = np.arange(15) + 3
    colors = js_plotting_utils.colorscale(cmap, values, symmetric_cmap=False)
    assert (colors['vmin'], colors['vmax']) == (3, 17)
    assert (colors['norm'].vmax, colors['norm'].vmin) == (17, 3)


def test_colorscale_vmin_vmax():
    cmap = 'jet'
    values = np.arange(15)
    colors = js_plotting_utils.colorscale(cmap, values, vmax=7)
    assert (colors['vmin'], colors['vmax']) == (-7, 7)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (7, -7)
    assert colors['symmetric_cmap']
    colors = js_plotting_utils.colorscale(
        cmap, values, vmax=7, vmin=-5)
    assert (colors['vmin'], colors['vmax']) == (-7, 7)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (7, -7)
    assert colors['symmetric_cmap']


def test_colorscale_asymmetric_cmap_vmax():
    cmap = 'jet'
    values = np.arange(15)
    colors = js_plotting_utils.colorscale(cmap, values, vmax=7,
                                          symmetric_cmap=False)
    assert (colors['vmin'], colors['vmax']) == (0, 7)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (7, 0)
    assert not colors['symmetric_cmap']
    values = np.arange(15) + 3
    colors = js_plotting_utils.colorscale(cmap, values, vmax=7,
                                          symmetric_cmap=False)
    assert (colors['vmin'], colors['vmax']) == (3, 7)
    assert (colors['norm'].vmax, colors['norm'].vmin) == (7, 3)
    colors = js_plotting_utils.colorscale(
        cmap, values, vmax=7, symmetric_cmap=False, vmin=1)
    assert (colors['vmin'], colors['vmax']) == (1, 7)
    assert (colors['norm'].vmax, colors['norm'].vmin) == (7, 1)
    colors = js_plotting_utils.colorscale(
        cmap, values, vmax=10, symmetric_cmap=False, vmin=6, threshold=5)
    assert (colors['vmin'], colors['vmax']) == (0, 10)
    assert (colors['norm'].vmax, colors['norm'].vmin) == (10, 0)
    colors = js_plotting_utils.colorscale(
        cmap, values, vmax=10, symmetric_cmap=False, vmin=None, threshold=5)
    assert (colors['vmin'], colors['vmax']) == (0, 10)
    assert (colors['norm'].vmax, colors['norm'].vmin) == (10, 0)


def test_colorscale_asymmetric_cmap_negative_values():
    cmap = 'jet'
    values = np.linspace(-15, 4)
    assert_warns(UserWarning, js_plotting_utils.colorscale, cmap,
                 values, symmetric_cmap=False)

    colors = js_plotting_utils.colorscale(cmap, values, vmax=7,
                                          symmetric_cmap=False)
    assert (colors['vmin'], colors['vmax']) == (-7, 7)
    assert colors['cmap'].N == 256
    assert (colors['norm'].vmax, colors['norm'].vmin) == (7, -7)
    assert colors['symmetric_cmap']


def test_encode():
    for dtype in ['<f4', '<i4', '>f4', '>i4']:
        a = np.arange(10, dtype=dtype)
        encoded = js_plotting_utils.encode(a)
        decoded = base64.b64decode(encoded.encode('utf-8'))
        b = np.frombuffer(decoded, dtype=dtype)
        assert np.allclose(js_plotting_utils.decode(encoded, dtype=dtype), b)
        assert np.allclose(a, b)


def test_mesh_to_plotly():
    fsaverage = fetch_surf_fsaverage()
    coord, triangles = surface.load_surf_mesh(fsaverage['pial_left'])
    plotly = js_plotting_utils.mesh_to_plotly(fsaverage['pial_left'])
    for i, key in enumerate(['_x', '_y', '_z']):
        assert np.allclose(
            js_plotting_utils.decode(plotly[key], '<f4'), coord[:, i])
    for i, key in enumerate(['_i', '_j', '_k']):
        assert np.allclose(
            js_plotting_utils.decode(plotly[key], '<i4'), triangles[:, i])


def check_html(html, check_selects=True, plot_div_id='surface-plot'):
    fd, tmpfile = tempfile.mkstemp()
    try:
        os.close(fd)
        html.save_as_html(tmpfile)
        with open(tmpfile) as f:
            saved = f.read()
        assert saved == html.get_standalone()
    finally:
        os.remove(tmpfile)
    assert "INSERT" not in html.html
    assert html.get_standalone() == html.html
    assert html._repr_html_() == html.get_iframe()
    assert str(html) == html.get_standalone()
    assert '<meta charset="UTF-8" />' in str(html)
    _check_open_in_browser(html)
    resized = html.resize(3, 17)
    assert resized is html
    assert (html.width, html.height) == (3, 17)
    assert "width=3 height=17" in html.get_iframe()
    assert "width=33 height=37" in html.get_iframe(33, 37)
    if not LXML_INSTALLED:
        return
    root = etree.HTML(html.html.encode('utf-8'),
                      parser=etree.HTMLParser(huge_tree=True))
    head = root.find('head')
    assert len(head.findall('script')) == 5
    body = root.find('body')
    div = body.find('div')
    assert ('id', plot_div_id) in div.items()
    if not check_selects:
        return
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


def _open_mock(f):
    print('opened {}'.format(f))


def _check_open_in_browser(html):
    wb_open = webbrowser.open
    webbrowser.open = _open_mock
    try:
        html.open_in_browser(temp_file_lifetime=None)
        temp_file = html._temp_file
        assert html._temp_file is not None
        assert os.path.isfile(temp_file)
        html.remove_temp_file()
        assert html._temp_file is None
        assert not os.path.isfile(temp_file)
        html.remove_temp_file()
        html._temp_file = 'aaaaaaaaaaaaaaaaaaaaaa'
        html.remove_temp_file()
    finally:
        webbrowser.open = wb_open
        try:
            os.remove(temp_file)
        except Exception:
            pass


def test_temp_file_removing():
    html = js_plotting_utils.HTMLDocument('hello')
    wb_open = webbrowser.open
    webbrowser.open = _open_mock
    try:
        html.open_in_browser(temp_file_lifetime=.5)
        assert os.path.isfile(html._temp_file)
        time.sleep(1.5)
        assert not os.path.isfile(html._temp_file)
        html.open_in_browser(temp_file_lifetime=None)
        assert os.path.isfile(html._temp_file)
        time.sleep(1.5)
        assert os.path.isfile(html._temp_file)
    finally:
        webbrowser.open = wb_open
        try:
            os.remove(html._temp_file)
        except Exception:
            pass


def _open_views():
    return [js_plotting_utils.HTMLDocument('') for i in range(12)]


def _open_one_view():
    for i in range(12):
        v = js_plotting_utils.HTMLDocument('')
    return v


def test_open_view_warning():
    # opening many views (without deleting the SurfaceView objects)
    # should raise a warning about memory usage
    assert_warns(UserWarning, _open_views)
    assert_no_warnings(_open_one_view)


def test_to_color_strings():
    colors = [[0, 0, 1], [1, 0, 0], [.5, .5, .5]]
    as_str = js_plotting_utils.to_color_strings(colors)
    assert as_str == ['#0000ff', '#ff0000', '#7f7f7f']

    colors = [[0, 0, 1, 1], [1, 0, 0, 1], [.5, .5, .5, 0]]
    as_str = js_plotting_utils.to_color_strings(colors)
    assert as_str == ['#0000ff', '#ff0000', '#7f7f7f']

    colors = ['#0000ff', '#ff0000', '#7f7f7f']
    as_str = js_plotting_utils.to_color_strings(colors)
    assert as_str == ['#0000ff', '#ff0000', '#7f7f7f']

    colors = [[0, 0, 1, 1], [1, 0, 0, 1], [.5, .5, .5, 0]]
    as_str = js_plotting_utils.to_color_strings(colors)
    assert as_str == ['#0000ff', '#ff0000', '#7f7f7f']

    colors = ['r', 'green', 'black', 'white']
    as_str = js_plotting_utils.to_color_strings(colors)
    assert as_str == ['#ff0000', '#008000', '#000000', '#ffffff']

    if matplotlib.__version__ < '2':
        return

    colors = ['#0000ffff', '#ff0000ab', '#7f7f7f00']
    as_str = js_plotting_utils.to_color_strings(colors)
    assert as_str == ['#0000ff', '#ff0000', '#7f7f7f']
