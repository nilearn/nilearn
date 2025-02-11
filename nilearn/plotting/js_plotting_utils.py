"""Helps for views, i.e. interactive plots from html_surface and \
html_connectome.
"""

import base64
import warnings
from pathlib import Path
from string import Template

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nilearn.plotting.html_document import (  # noqa: F401
    HTMLDocument,
    set_max_img_views_before_warning,
)
from nilearn.surface import load_surf_mesh

from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold

MAX_IMG_VIEWS_BEFORE_WARNING = 10


def add_js_lib(html, embed_js=True):
    """Add javascript libraries to html template.

    If embed_js is True, jquery and plotly are embedded in resulting page.
    otherwise, they are loaded via CDNs.

    """
    js_dir = Path(__file__).parent / "data" / "js"
    with (js_dir / "surface-plot-utils.js").open() as f:
        js_utils = f.read()
    if not embed_js:
        js_lib = f"""
        <script
        src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js">
        </script>
        <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
        <script>
        {js_utils}
        </script>
        """
    else:
        with (js_dir / "jquery.min.js").open() as f:
            jquery = f.read()
        with (js_dir / "plotly-gl3d-latest.min.js").open() as f:
            plotly = f.read()
        js_lib = f"""
        <script>{jquery}</script>
        <script>{plotly}</script>
        <script>
        {js_utils}
        </script>
        """
    if not isinstance(html, Template):
        html = Template(html)
    return html.safe_substitute({"INSERT_JS_LIBRARIES_HERE": js_lib})


def get_html_template(template_name):
    """Get an HTML file from package data."""
    template_path = Path(__file__).parent / "data" / "html" / template_name

    with template_path.open("rb") as f:
        return Template(f.read().decode("utf-8"))


def colorscale(
    cmap, values, threshold=None, symmetric_cmap=True, vmax=None, vmin=None
):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    cmap = plt.get_cmap(cmap)
    abs_values = np.abs(values)
    if not symmetric_cmap and (values.min() < 0):
        warnings.warn(
            "you have specified symmetric_cmap=False "
            "but the map contains negative values; "
            "setting symmetric_cmap to True",
            stacklevel=3,
        )
        symmetric_cmap = True
    if symmetric_cmap and vmin is not None:
        warnings.warn(
            "vmin cannot be chosen when cmap is symmetric", stacklevel=3
        )
        vmin = None
    if vmax is None:
        vmax = abs_values.max()
    # cast to float to avoid TypeError if vmax is a numpy boolean
    vmax = float(vmax)
    if symmetric_cmap:
        vmin = -vmax
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
            cmaplist[i] = (0.5, 0.5, 0.5, 1.0)  # just an average gray color
    our_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )
    x = np.linspace(0, 1, 100)
    rgb = our_cmap(x, bytes=True)[:, :3]
    rgb = np.array(rgb, dtype=int)
    colors = [
        [np.round(i, 3), f"rgb({col[0]}, {col[1]}, {col[2]})"]
        for i, col in zip(x, rgb)
    ]
    return {
        "colors": colors,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": our_cmap,
        "norm": norm,
        "abs_threshold": abs_threshold,
        "symmetric_cmap": symmetric_cmap,
    }


def encode(a):
    """Base64 encode a numpy array."""
    try:
        data = a.tobytes()
    except AttributeError:
        # np < 1.9
        data = a.tostring()
    return base64.b64encode(data).decode("utf-8")


def decode(b, dtype):
    """Decode a numpy array encoded as Base64."""
    return np.frombuffer(base64.b64decode(b.encode("utf-8")), dtype)


def mesh_to_plotly(mesh):
    """Convert a :term:`mesh` to plotly format."""
    mesh = load_surf_mesh(mesh)
    x, y, z = map(encode, np.asarray(mesh.coordinates.T, dtype="<f4"))
    i, j, k = map(encode, np.asarray(mesh.faces.T, dtype="<i4"))
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
    """Return a list of colors as hex strings."""
    cmap = mpl.colors.ListedColormap(colors)
    colors = cmap(np.arange(cmap.N))[:, :3]
    colors = np.asarray(colors * 255, dtype="uint8")
    colors = [
        f"#{int(row[0]):02x}{int(row[1]):02x}{int(row[2]):02x}"
        for row in colors
    ]
    return colors
