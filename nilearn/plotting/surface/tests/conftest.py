"""Test fixtures used by nilearn.plotting.surface tests."""

import numpy as np
import pytest

from nilearn.surface import SurfaceImage


@pytest.fixture
def bg_map(rng, in_memory_mesh):
    """Return a background map with positive value."""
    return np.abs(rng.standard_normal(size=in_memory_mesh.n_vertices))


@pytest.fixture
def surface_image_roi(surf_mask_1d):
    """SurfaceImage for plotting."""
    return surf_mask_1d


@pytest.fixture
def parcellation(in_memory_mesh):
    parcellation = np.zeros((in_memory_mesh.n_vertices,))
    parcellation[in_memory_mesh.faces[3]] = 1
    parcellation[in_memory_mesh.faces[5]] = 2
    return parcellation


@pytest.fixture
def surface_image_parcellation(rng, in_memory_mesh):
    data = rng.integers(100, size=(in_memory_mesh.n_vertices, 1)).astype(float)
    parcellation = SurfaceImage(
        mesh={"left": in_memory_mesh, "right": in_memory_mesh},
        data={"left": data, "right": data},
    )
    return parcellation
