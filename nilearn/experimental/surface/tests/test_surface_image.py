from pathlib import Path

import numpy as np
import pytest

from nilearn import datasets
from nilearn.experimental.surface import FileMesh, InMemoryMesh, SurfaceImage


def test_compare_file_and_inmemory_mesh(mini_mesh, tmp_path):
    left = mini_mesh.parts["left"]
    gifti_file = tmp_path / "left.gii"
    left.to_gifti(gifti_file)

    left_read = FileMesh(gifti_file)
    assert left.n_vertices == left_read.n_vertices
    assert np.array_equal(left.coordinates, left_read.coordinates)
    assert np.array_equal(left.faces, left_read.faces)

    left_loaded = left_read.loaded()
    assert isinstance(left_loaded, InMemoryMesh)
    assert left.n_vertices == left_loaded.n_vertices
    assert np.array_equal(left.coordinates, left_loaded.coordinates)
    assert np.array_equal(left.faces, left_loaded.faces)


def test_surface_image_shape(make_mini_img):
    img = make_mini_img()
    assert img.shape == (9,)
    img = make_mini_img((3,))
    assert img.shape == (3, 9)
    img = make_mini_img((7, 3))
    assert img.shape == (7, 3, 9)


def test_data_shape_not_matching_mesh(mini_img, flip):
    with pytest.raises(ValueError, match="shape.*vertices"):
        SurfaceImage(mini_img.mesh, flip(mini_img.data))


def test_data_shape_inconsistent(make_mini_img):
    img = make_mini_img((7,))
    bad_data = {
        "left": img.data.parts["left"],
        "right": img.data.parts["right"][:4],
    }
    with pytest.raises(ValueError, match="incompatible shapes"):
        SurfaceImage(img.mesh, bad_data)


def test_data_keys_not_matching_mesh(mini_img):
    with pytest.raises(ValueError, match="same keys"):
        SurfaceImage(
            {"left": mini_img.mesh.parts["left"]},
            mini_img.data,
        )


def test_load_surf_data_gii_gz():
    """Test the loader with gzipped fsaverage5 files."""
    mesh_right = datasets.fetch_surf_fsaverage().pial_right
    mesh_left = datasets.fetch_surf_fsaverage().pial_left
    data_right = datasets.fetch_surf_fsaverage().sulc_right
    data_left = datasets.fetch_surf_fsaverage().sulc_left

    SurfaceImage(
        mesh={"left": mesh_left, "right": mesh_right},
        data={"left": data_left, "right": data_right},
    )

    # use Path
    SurfaceImage(
        mesh={"left": Path(mesh_left), "right": Path(mesh_right)},
        data={"left": Path(data_left), "right": Path(data_right)},
    )
