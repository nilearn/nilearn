import numpy as np

from nilearn.experimental.surface import _io


def test_read_array():
    """Smoke test."""
    expected_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = _io.read_array(expected_data)
    assert np.all(result == expected_data)


def test_read_mesh():
    """Smoke test."""
    expected_coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_faces = np.array([[0, 1, 2]])
    result = _io.read_mesh([expected_coordinates, expected_faces])
    assert np.all(result["coordinates"] == expected_coordinates)
    assert np.all(result["faces"] == expected_faces)
    print(result["faces"].shape)
