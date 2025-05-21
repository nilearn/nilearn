"""Utilities for surface mesh, data, images."""

from warnings import warn

import numpy as np

from nilearn._utils.exceptions import MeshDimensionError
from nilearn._utils.logger import find_stack_level


def assert_polydata_equal(data_1, data_2):
    """Check that 2 PolyData data are equal."""
    set_1 = set(data_1.parts.keys())
    set_2 = set(data_2.parts.keys())
    if set_1 != set_2:
        diff = set_1.symmetric_difference(set_2)
        raise ValueError(
            f"PolyData do not have the same keys. Offending keys: {diff}"
        )

    for key in data_1.parts:
        if not np.array_equal(data_1.parts[key], data_2.parts[key]):
            raise ValueError(
                f"Part '{key}' of PolyData instances are not equal."
            )


def assert_polymesh_equal(mesh_1, mesh_2) -> None:
    """Check that 2 PolyMeshes are equal."""
    assert_polymesh_have_same_keys(mesh_1, mesh_2)
    for key in mesh_1.parts:
        assert_surface_mesh_equal(mesh_1.parts[key], mesh_2.parts[key])


def assert_polymesh_have_same_keys(mesh_1, mesh_2):
    """Check that 2 polymeshes have the same keys."""
    set_1 = set(mesh_1.parts.keys())
    set_2 = set(mesh_2.parts.keys())
    if set_1 != set_2:
        diff = set_1.symmetric_difference(set_2)
        raise MeshDimensionError(
            f"PolyMeshes do not have the same keys. Offending keys: {diff}"
        )


def check_polymesh_equal(mesh_1, mesh_2) -> None:
    """Check polymesh at-least have same number of vertices if not equal."""
    try:
        assert_polymesh_equal(mesh_1, mesh_2)
    except MeshDimensionError:
        assert_polymesh_have_same_keys(mesh_1, mesh_2)
        for key in mesh_1.parts:
            assert_same_number_vertices(mesh_1.parts[key], mesh_2.parts[key])
        warn(
            "Meshes are not identical but have compatible number of vertices.",
            stacklevel=find_stack_level(),
        )


def assert_same_number_vertices(mesh_1, mesh_2):
    """Assert 2 meshes or polymeshes have the same number of vertices."""
    if mesh_1.n_vertices != mesh_2.n_vertices:
        raise MeshDimensionError(
            f"Number of vertices do not match for between meshes.\n"
            f"{mesh_1.n_vertices=} and {mesh_2.n_vertices=}"
        )


def assert_surface_mesh_equal(mesh_1, mesh_2) -> None:
    """Check that 2 SurfaceMeshes are equal."""
    if not np.array_equal(mesh_1.coordinates, mesh_2.coordinates):
        raise MeshDimensionError("Meshes do not have the same coordinates.")
    if not np.array_equal(mesh_1.faces, mesh_2.faces):
        raise MeshDimensionError("Meshes do not have the same faces.")


def assert_surface_image_equal(img_1, img_2) -> None:
    """Check that 2 SurfaceImages are equal."""
    assert_polymesh_equal(img_1.mesh, img_2.mesh)
    assert_polydata_equal(img_1.data, img_2.data)
