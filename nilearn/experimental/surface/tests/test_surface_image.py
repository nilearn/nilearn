import nibabel as nib
import numpy as np
import pytest

from nilearn.experimental.surface import (
    FileMesh,
    InMemoryMesh,
    Mesh,
    SurfaceImage,
)


@pytest.fixture
def fake_mesh_testing_data():
    """Create fake mesh data for testing."""
    mesh_data = {
        "coordinates": np.array(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype="uint8"
        ),
        "faces": np.array([[0, 1, 2]], dtype="uint8"),
    }
    fake_mesh = {
        "left": InMemoryMesh(mesh_data["coordinates"], mesh_data["faces"]),
        "right": InMemoryMesh(mesh_data["coordinates"], mesh_data["faces"]),
    }
    return mesh_data, fake_mesh


def test_mesh_repr():
    """Instantiate Mesh class and call its __repr__ method."""
    n_vertices = 100
    mesh = Mesh()
    mesh.n_vertices = n_vertices
    assert repr(mesh) == f"<Mesh with {n_vertices} vertices>"


def test_in_memory_mesh_coordinates_and_faces_attributes(
    fake_mesh_testing_data,
):
    """Test InMemoryMesh attribute values are as expected."""
    mesh_data, _ = fake_mesh_testing_data
    mesh = InMemoryMesh(mesh_data["coordinates"], mesh_data["faces"])
    assert np.array_equal(mesh.coordinates, mesh_data["coordinates"])
    assert np.array_equal(mesh.faces, mesh_data["faces"])


def test_file_mesh_coordinates_and_faces_properties(
    tmp_path, fake_mesh_testing_data
):
    """Test FileMesh property values are as expected."""
    mesh_data, _ = fake_mesh_testing_data
    gii_image = nib.gifti.GiftiImage()
    coords_array = nib.gifti.GiftiDataArray(
        mesh_data["coordinates"], intent="NIFTI_INTENT_POINTSET"
    )
    faces_array = nib.gifti.GiftiDataArray(
        mesh_data["faces"], intent="NIFTI_INTENT_TRIANGLE"
    )
    gii_image.add_gifti_data_array(coords_array)
    gii_image.add_gifti_data_array(faces_array)
    nib.save(gii_image, f"{tmp_path}test.gii")
    mesh = FileMesh(f"{tmp_path}test.gii")
    assert np.array_equal(mesh.coordinates, mesh_data["coordinates"])
    assert np.array_equal(mesh.faces, mesh_data["faces"])


def test_surface_image(fake_mesh_testing_data):
    """Smoke test for SurfaceImage."""
    fake_data = {
        "left": np.arange(1, 10).reshape(3, 3),
        "right": np.arange(1, 10).reshape(3, 3),
    }
    _, fake_mesh = fake_mesh_testing_data
    surface_image = SurfaceImage(data=fake_data, mesh=fake_mesh)
    assert surface_image.shape == (3, 6)
