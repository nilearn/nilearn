"""Fixtures for experimental module."""

import numpy as np
import pytest

from nilearn.experimental.surface import (
    InMemoryMesh,
    PolyMesh,
    SurfaceImage,
)


@pytest.fixture()
def make_mesh():
    """Create a sample mesh with two parts: left and right, and total of
    9 vertices and 10 faces.

    The left part is a tetrahedron with four vertices and four faces.
    The right part is a pyramid with five vertices and six faces.
    """

    def _make_mesh():
        left_coords = np.asarray(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        left_faces = np.asarray([[1, 0, 2], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        right_coords = (
            np.asarray(
                [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]
            )
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

    return _make_mesh


@pytest.fixture
def make_surface_img(make_mesh, request):
    """Create a sample surface image using the sample mesh.
    This will just add some random data to the vertices of the mesh.
    The shape of the data will be (n_samples, n_vertices).
    n_samples is the parameter of the fixture, default is 50.
    """
    n_samples = getattr(request, "param", 50)
    mesh = make_mesh()
    data = {}
    for i, (key, val) in enumerate(mesh.parts.items()):
        data_shape = (n_samples, val.n_vertices)
        data_part = (
            np.arange(np.prod(data_shape)).reshape(data_shape) + 1.0
        ) * 10**i
        data[key] = data_part
    return SurfaceImage(mesh, data)


@pytest.fixture
def make_surface_mask(make_mesh, request):
    """Create a sample surface mask using the sample mesh.
    This will create a mask with n_zeros zeros and the rest ones.
    n_zeros is the parameter of the fixture, default is 4.
    """
    n_zeros = getattr(request, "param", 4)
    mesh = make_mesh()
    data = {}
    for key, val in mesh.parts.items():
        data_part = np.ones(val.n_vertices, dtype=int)
        for i in range(n_zeros // 2):
            data_part[..., i] = 0
        data_part = data_part.astype(bool)
        data[key] = data_part
    return SurfaceImage(mesh, data)


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
