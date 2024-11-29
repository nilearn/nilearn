"""Utilities to test surface mesh, data, images."""

import numpy as np


def assert_polydata_equal(data_1, data_2):
    """Check that 2 SurfaceImages data are equal."""
    assert set(data_1.parts.keys()) == set(data_2.parts.keys())
    for key in data_1.parts:
        assert np.array_equal(data_1.parts[key], data_2.parts[key])


def assert_polymesh_equal(mesh_1, mesh_2):
    """Check that 2 PolyMeshes are equal."""
    assert set(mesh_1.parts.keys()) == set(mesh_2.parts.keys())
    for key in mesh_1.parts:
        assert_surface_mesh_equal(mesh_1.parts[key], mesh_2.parts[key])


def assert_surface_mesh_equal(mesh_1, mesh_2):
    """Check that 2 SurfaceMeshes are equal."""
    assert np.array_equal(mesh_1.coordinates, mesh_2.coordinates)
    assert np.array_equal(mesh_1.faces, mesh_2.faces)


def assert_surface_image_equal(img_1, img_2):
    """Check that 2 SurfaceImages are equal."""
    assert_polymesh_equal(img_1.mesh, img_2.mesh)
    assert_polydata_equal(img_1.data, img_2.data)
