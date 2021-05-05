"""
Helper functions for views, i.e. interactive plots from html_surface and
html_connectome.
"""

import os
import base64
import warnings
from string import Template

import matplotlib as mpl
import numpy as np
from matplotlib import cm as mpl_cm

# included here for backward compatibility
from nilearn.plotting.html_document import (
    HTMLDocument, set_max_img_views_before_warning,)  # noqa
from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .. import surface

MAX_IMG_VIEWS_BEFORE_WARNING = 10


def add_js_lib(html, embed_js=True):
    """Add javascript libraries to html template.

    If embed_js is True, jquery and plotly are embedded in resulting page.
    otherwise, they are loaded via CDNs.

    """
    js_dir = os.path.join(os.path.dirname(__file__), 'data', 'js')
    with open(os.path.join(js_dir, 'surface-plot-utils.js')) as f:
        js_utils = f.read()
    if not embed_js:
        js_lib = """
        <script
        src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js">
        </script>
        <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
        <script>
        {}
        </script>
        """.format(js_utils)
    else:
        with open(os.path.join(js_dir, 'jquery.min.js')) as f:
            jquery = f.read()
        with open(os.path.join(js_dir, 'plotly-gl3d-latest.min.js')) as f:
            plotly = f.read()
        js_lib = """
        <script>{}</script>
        <script>{}</script>
        <script>
        {}
        </script>
        """.format(jquery, plotly, js_utils)
    if not isinstance(html, Template):
        html = Template(html)
    return html.safe_substitute({'INSERT_JS_LIBRARIES_HERE': js_lib})


def get_html_template(template_name):
    """Get an HTML file from package data"""
    template_path = os.path.join(
        os.path.dirname(__file__), 'data', 'html', template_name)
    with open(template_path, 'rb') as f:
        return Template(f.read().decode('utf-8'))


def colorscale(cmap, values, threshold=None, symmetric_cmap=True,
               vmax=None, vmin=None):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    cmap = mpl_cm.get_cmap(cmap)
    abs_values = np.abs(values)
    if not symmetric_cmap and (values.min() < 0):
        warnings.warn('you have specified symmetric_cmap=False '
                      'but the map contains negative values; '
                      'setting symmetric_cmap to True')
        symmetric_cmap = True
    if symmetric_cmap and vmin is not None:
        warnings.warn('vmin cannot be chosen when cmap is symmetric')
        vmin = None
    if threshold is not None:
        if vmin is not None:
            warnings.warn('choosing both vmin and a threshold is not allowed; '
                          'setting vmin to 0')
        vmin = 0
    if vmax is None:
        vmax = abs_values.max()
    # cast to float to avoid TypeError if vmax is a numpy boolean
    vmax = float(vmax)
    if symmetric_cmap:
        vmin = - vmax
    if vmin is None:
        vmin = values.min()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    abs_threshold = None
    if threshold is not None:
        abs_threshold = check_threshold(threshold, values, fast_abs_percentile)
        istart = int(norm(-abs_threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(abs_threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)  # just an average gray color
    our_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    x = np.linspace(0, 1, 100)
    rgb = our_cmap(x, bytes=True)[:, :3]
    rgb = np.array(rgb, dtype=int)
    colors = []
    for i, col in zip(x, rgb):
        colors.append([np.round(i, 3), "rgb({}, {}, {})".format(*col)])
    return {
        'colors': colors, 'vmin': vmin, 'vmax': vmax, 'cmap': our_cmap,
        'norm': norm, 'abs_threshold': abs_threshold,
        'symmetric_cmap': symmetric_cmap
    }


def encode(a):
    """Base64 encode a numpy array"""
    try:
        data = a.tobytes()
    except AttributeError:
        # np < 1.9
        data = a.tostring()
    return base64.b64encode(data).decode('utf-8')


def decode(b, dtype):
    """Decode a numpy array encoded as Base64"""
    return np.frombuffer(base64.b64decode(b.encode('utf-8')), dtype)


def mesh_to_plotly(mesh):
    mesh = surface.load_surf_mesh(mesh)
    x, y, z = map(encode, np.asarray(mesh[0].T, dtype='<f4'))
    i, j, k = map(encode, np.asarray(mesh[1].T, dtype='<i4'))
    info = {
        "_x": x,
        "_y": y,
        "_z": z,
        "_i": i,
        "_j": j,
        "_k": k,
    }
    return info


def to_color_strings(colors):
    cmap = mpl.colors.ListedColormap(colors)
    colors = cmap(np.arange(cmap.N))[:, :3]
    colors = np.asarray(colors * 255, dtype='uint8')
    colors = ['#{:02x}{:02x}{:02x}'.format(*row) for row in colors]
    return colors
