import pytest

import numpy as np


@pytest.fixture
def bg_map(rng, in_memory_mesh):
    """Return a background map with posive value."""
    return np.abs(rng.standard_normal(size=in_memory_mesh.n_vertices))


@pytest.fixture
def surface_image_roi(surf_mask_1d):
    """SurfaceImage for plotting."""
    return surf_mask_1d
