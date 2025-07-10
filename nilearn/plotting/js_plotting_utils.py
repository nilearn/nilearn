"""Helps for views, i.e. interactive plots from html_surface and \
html_connectome.
"""

import base64
from pathlib import Path
from string import Template

import numpy as np

from nilearn._utils.html_document import (  # noqa: F401
    HTMLDocument,
    set_max_img_views_before_warning,
)
from nilearn.surface import load_surf_mesh

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
