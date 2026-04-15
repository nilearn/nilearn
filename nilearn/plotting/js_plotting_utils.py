"""Helps for views, i.e. interactive plots from html_surface and \
html_connectome.
"""

import base64

import numpy as np

from nilearn.surface import load_surf_mesh

MAX_IMG_VIEWS_BEFORE_WARNING = 10

NIIVUE_VERSION = "0.68.1"


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
