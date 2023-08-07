from typing import Callable
import numpy as np
import pytest

from nilearn.experimental.surface import (
    InMemoryMesh,
    SurfaceImage,
    load_fsaverage,
    PolyMesh,
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
    return {
        "left_hemisphere": InMemoryMesh(left_coords, left_faces),
        "right_hemisphere": InMemoryMesh(right_coords, right_faces),
    }


def _flip(parts):
    return {
        "right_hemisphere": parts["left_hemisphere"],
        "left_hemisphere": parts["right_hemisphere"],
    }


@pytest.fixture
def flipped_mini_mesh(mini_mesh) -> PolyMesh:
    return _flip(mini_mesh)


@pytest.fixture
def make_mini_img(mini_mesh) -> Callable:
    """Small surface image for tests"""

    def f(shape=()):
        data = {}
        for i, (key, val) in enumerate(mini_mesh.items()):
            data_shape = tuple(shape) + (val.n_vertices,)
            data_part = (
                np.arange(np.prod(data_shape)).reshape(data_shape) + 1.0
            ) * 10**i
            data[key] = data_part
        return SurfaceImage(mini_mesh, data)

    return f


@pytest.fixture
def make_flipped_mini_img(make_mini_img) -> Callable:
    def f(shape=()):
        img = make_mini_img(shape)
        return SurfaceImage(_flip(img.mesh), _flip(img.data))

    return f


@pytest.fixture
def mini_img(make_mini_img) -> SurfaceImage:
    return make_mini_img()


@pytest.fixture
def flipped_mini_img(make_flipped_mini_img) -> SurfaceImage:
    return make_flipped_mini_img()


@pytest.fixture
def pial_surface_mesh():
    """Get fsaverage mesh for testing."""
    return load_fsaverage()["pial"]
