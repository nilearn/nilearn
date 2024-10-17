"""Fixtures for experimental module."""

import numpy as np
import pytest

from nilearn.experimental.surface import (
    InMemoryMesh,
    PolyMesh,
    SurfaceImage,
)


def _make_mesh():
    """Create a sample mesh with two parts: left and right, and total of
    9 vertices and 10 faces.

    The left part is a tetrahedron with four vertices and four faces.
    The right part is a pyramid with five vertices and six faces.
    """
    left_coords = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    left_faces = np.asarray([[1, 0, 2], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    right_coords = (
        np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
        + 2.0
    )
    right_faces = np.asarray(
        [
            [0, 1, 4],
            [0, 3, 1],
            [1, 3, 2],
            [1, 2, 4],
            [2, 3, 4],
            [0, 4, 3],
        ]
    )
    return PolyMesh(
        left=InMemoryMesh(left_coords, left_faces),
        right=InMemoryMesh(right_coords, right_faces),
    )


@pytest.fixture()
def make_mesh():
    """Return _make_mesh as a function allowing it to be used as a fixture."""
    return _make_mesh


@pytest.fixture
def make_surface_img():
    """Create a sample surface image using the sample mesh.
    This will add some random data to the vertices of the mesh.
    The shape of the data will be (n_samples, n_vertices).
    n_samples by default is 1.
    """

    def _make_surface_img(n_samples=(1,)):
        mesh = _make_mesh()
        data = {}
        for i, (key, val) in enumerate(mesh.parts.items()):
            data_shape = (*tuple(n_samples), val.n_vertices)
            data_part = (
                np.arange(np.prod(data_shape)).reshape(data_shape) + 1.0
            ) * 10**i
            data[key] = data_part
        return SurfaceImage(mesh, data)

    return _make_surface_img


@pytest.fixture
def make_surface_mask():
    """Create a sample surface mask using the sample mesh.
    This will create a mask with n_zeros zeros (default is 4) and the
    rest ones. If empty is True, the mask will be None, required for
    tests for html reports.
    """

    def _make_surface_mask(n_zeros=4, empty=False):
        if empty:
            return None
        else:
            mesh = _make_mesh()
            data = {}
            for key, val in mesh.parts.items():
                data_part = np.ones(val.n_vertices, dtype=int)
                for i in range(n_zeros // 2):
                    data_part[..., i] = 0
                data_part = data_part.astype(bool)
                data[key] = data_part
            return SurfaceImage(mesh, data)

    return _make_surface_mask


@pytest.fixture
def surface_label_img():
    """Return a sample surface label image using the sample mesh.
    Has two regions with values 0 and 1 respectively.
    """

    def _surface_label_img():
        mesh = _make_mesh()
        data = {
            "left": np.asarray([0, 0, 1, 1]),
            "right": np.asarray([1, 1, 0, 0, 0]),
        }

        return SurfaceImage(mesh, data)

    return _surface_label_img


@pytest.fixture
def flip():
    """Flip hemispheres of a surface image data or mesh."""

    def f(poly_obj):
        keys = list(poly_obj.parts.keys())
        keys = [keys[-1]] + keys[:-1]
        return dict(zip(keys, poly_obj.parts.values()))

    return f


@pytest.fixture
def flip_img(flip):
    """Flip hemispheres of a surface image."""

    def f(img):
        return SurfaceImage(flip(img.mesh), flip(img.data))

    return f


@pytest.fixture
def assert_img_equal():
    """Check that 2 SurfaceImages are equal."""

    def f(img_1, img_2):
        assert set(img_1.data.parts.keys()) == set(img_2.data.parts.keys())
        for key in img_1.data.parts:
            assert np.array_equal(img_1.data.parts[key], img_2.data.parts[key])

    return f


@pytest.fixture
def drop_img_part():
    """Remove one hemisphere from a SurfaceImage."""

    def f(img, part_name="right"):
        mesh_parts = img.mesh.parts.copy()
        mesh_parts.pop(part_name)
        data_parts = img.data.parts.copy()
        data_parts.pop(part_name)
        return SurfaceImage(mesh_parts, data_parts)

    return f
