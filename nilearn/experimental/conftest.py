"""Fxitures for experimental module."""

from typing import Callable

import numpy as np
import pytest

from nilearn.experimental.surface import (
    InMemoryMesh,
    PolyMesh,
    SurfaceImage,
    load_fsaverage,
)


@pytest.fixture
def mini_mesh() -> PolyMesh:
    """Small mesh for tests with 2 parts with different numbers of vertices."""
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


@pytest.fixture
def make_mini_img(mini_mesh) -> Callable:
    """Small surface image for tests."""

    def f(shape=()):
        data = {}
        for i, (key, val) in enumerate(mini_mesh.parts.items()):
            data_shape = tuple(shape) + (val.n_vertices,)
            data_part = (
                np.arange(np.prod(data_shape)).reshape(data_shape) + 1.0
            ) * 10**i
            data[key] = data_part
        return SurfaceImage(mini_mesh, data)

    return f


@pytest.fixture
def mini_mask(mini_img) -> SurfaceImage:
    """Raturn small surface mask."""
    data = {k: (v > v.ravel()[0]) for k, v in mini_img.data.parts.items()}
    return SurfaceImage(mini_img.mesh, data)


@pytest.fixture
def mini_img(make_mini_img) -> SurfaceImage:
    """Raturn small surface image."""
    return make_mini_img()


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
def pial_surface_mesh():
    """Get fsaverage mesh for testing."""
    return load_fsaverage()["pial"]


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
