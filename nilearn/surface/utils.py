"""Utilities for surface mesh, data, images."""

import numpy as np


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


def assert_polymesh_equal(mesh_1, mesh_2):
    """Check that 2 PolyMeshes are equal."""
    set_1 = set(mesh_1.parts.keys())
    set_2 = set(mesh_2.parts.keys())
    if set_1 != set_2:
        diff = set_1.symmetric_difference(set_2)
        raise ValueError(
            f"PolyMeshes do not have the same keys. Offending keys: {diff}"
        )

    for key in mesh_1.parts:
        if mesh_1.parts[key].n_vertices != mesh_2.parts[key].n_vertices:
            raise ValueError(
                f"Number of vertices do not match for '{key}'."
                "number of vertices in mesh_1: "
                f"{mesh_1.parts[key].n_vertices}; "
                f"in mesh_2: {mesh_2.parts[key].n_vertices}"
            )

        assert_surface_mesh_equal(mesh_1.parts[key], mesh_2.parts[key])


def assert_surface_mesh_equal(mesh_1, mesh_2):
    """Check that 2 SurfaceMeshes are equal."""
    if not np.array_equal(mesh_1.coordinates, mesh_2.coordinates):
        raise ValueError("Meshes do not have the same coordinates.")
    if not np.array_equal(mesh_1.faces, mesh_2.faces):
        raise ValueError("Meshes do not have the same faces.")


def assert_surface_image_equal(img_1, img_2):
    """Check that 2 SurfaceImages are equal."""
    assert_polymesh_equal(img_1.mesh, img_2.mesh)
    assert_polydata_equal(img_1.data, img_2.data)
