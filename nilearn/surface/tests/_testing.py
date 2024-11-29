"""Helper functions for testing surface code."""

import numpy as np
from scipy.spatial import Delaunay

from nilearn.conftest import _rng
from nilearn.surface import InMemoryMesh


def generate_surf():
    """
    Generate random surface object for testing input output functions.

    This does not generate meaningful surfaces.
    """
    rng = _rng()
    coords = rng.random((20, 3))
    faces = rng.integers(coords.shape[0], size=(30, 3))
    return InMemoryMesh(coordinates=coords, faces=faces)


def flat_mesh(x_s, y_s, z=0):
    """Create a flat horizontal mesh."""
    x, y = np.mgrid[:x_s, :y_s]
    x, y = x.ravel(), y.ravel()
    z = np.ones(len(x)) * z
    vertices = np.asarray([x, y, z]).T
    triangulation = Delaunay(vertices[:, :2]).simplices
    return InMemoryMesh(coordinates=vertices, faces=triangulation)


def z_const_img(x_s, y_s, z_s):
    """Create an image that is constant in z direction."""
    hslice = np.arange(x_s * y_s).reshape((x_s, y_s))
    return np.ones((x_s, y_s, z_s)) * hslice[:, :, np.newaxis]
