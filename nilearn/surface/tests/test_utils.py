import numpy as np
import pytest

from nilearn.surface import PolyData
from nilearn.surface.utils import (
    assert_polydata_equal,
    assert_surface_mesh_equal,
)


def test_assert_polydata_equal():
    """Check that PolyData with different keys are detected."""
    data_1 = PolyData(left=np.ones((3, 2)), right=np.ones((3, 2)))
    data_2 = PolyData(left=np.ones((3, 2)), right=np.ones((3, 2)))

    assert_polydata_equal(data_1, data_2)

    data_2.parts["new_key"] = np.ones((3, 2))

    with pytest.raises(
        ValueError, match="PolyData do not have the same keys."
    ):
        assert_polydata_equal(data_1, data_2)


def test_assert_surface_mesh_equal(surf_mesh):
    """Check that Meshes with different faces/coords are detected."""
    assert_surface_mesh_equal(surf_mesh.parts["left"], surf_mesh.parts["left"])

    with pytest.raises(
        ValueError, match="Meshes do not have the same coordinates."
    ):
        assert_surface_mesh_equal(
            surf_mesh.parts["left"], surf_mesh.parts["right"]
        )

    surf_mesh.parts["right"].coordinates = surf_mesh.parts["left"].coordinates

    with pytest.raises(ValueError, match="Meshes do not have the same faces."):
        assert_surface_mesh_equal(
            surf_mesh.parts["left"], surf_mesh.parts["right"]
        )
