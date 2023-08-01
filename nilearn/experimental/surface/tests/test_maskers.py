import numpy as np
import pytest

from nilearn.experimental.surface import (
    SurfaceImage,
    SurfaceMasker,
    load_fsaverage,
)


@pytest.fixture
def pial_surface_mesh():
    """Get fsaverage mesh for testing."""
    return load_fsaverage()["pial"]


def test_SurfaceMasker(pial_surface_mesh):
    """Test fit_transform method"""
    masker = SurfaceMasker()
    data_array = np.arange(1, 5 * 10242 + 1).reshape((5, 10242))
    surf_img = SurfaceImage(
        data={"left_hemisphere": data_array, "right_hemisphere": data_array},
        mesh=pial_surface_mesh,
    )
    masked_data = masker.fit_transform(surf_img)
    assert masked_data.ndim == 2
    assert masked_data.shape == (5, 20484)
