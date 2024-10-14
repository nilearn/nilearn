from pathlib import Path

import numpy as np
import pytest

from nilearn import datasets
from nilearn.experimental.surface import FileMesh, InMemoryMesh, SurfaceImage
from nilearn.surface import load_surf_data, load_surf_mesh


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


@pytest.mark.parametrize("use_path", [True, False])
@pytest.mark.parametrize(
    "output_filename, expected_files, unexpected_files",
    [
        ("foo.gii", ["foo_hemi-L.gii", "foo_hemi-L.gii"], ["foo.gii"]),
        ("foo_hemi-L_T1w.gii", ["foo_hemi-L_T1w.gii"], ["foo_hemi-R_T1w.gii"]),
        ("foo_hemi-R_T1w.gii", ["foo_hemi-R_T1w.gii"], ["foo_hemi-L_T1w.gii"]),
    ],
)
def test_load_save_mesh(
    tmp_path, output_filename, expected_files, unexpected_files, use_path
):
    """Load fsaverage5 from filename or Path and save.

    Check that
    - the appropriate hemisphere information is added to the filename
    - only one hemisphere is saved if hemi- is in the filename
    - the roundtrip does not change the data
    """
    mesh_right = datasets.fetch_surf_fsaverage().pial_right
    mesh_left = datasets.fetch_surf_fsaverage().pial_left
    data_right = datasets.fetch_surf_fsaverage().sulc_right
    data_left = datasets.fetch_surf_fsaverage().sulc_left

    if use_path:
        img = SurfaceImage(
            mesh={"left": Path(mesh_left), "right": Path(mesh_right)},
            data={"left": Path(data_left), "right": Path(data_right)},
        )
    else:
        img = SurfaceImage(
            mesh={"left": mesh_left, "right": mesh_right},
            data={"left": data_left, "right": data_right},
        )

    if use_path:
        img.mesh.to_filename(tmp_path / output_filename)
    else:
        img.mesh.to_filename(str(tmp_path / output_filename))

    for file in unexpected_files:
        assert not (tmp_path / file).exists()

    for file in expected_files:
        assert (tmp_path / file).exists()

        mesh = load_surf_mesh(tmp_path / file)
        if "hemi-L" in file:
            expected_mesh = load_surf_mesh(mesh_left)
        elif "hemi-R" in file:
            expected_mesh = load_surf_mesh(mesh_right)
        assert np.array_equal(mesh.faces, expected_mesh.faces)
        assert np.array_equal(mesh.coordinates, expected_mesh.coordinates)


def test_save_mesh_default_suffix(tmp_path, mini_img):
    """Check default .gii extension is added."""
    mini_img.mesh.to_filename(
        tmp_path / "give_me_a_default_suffix_hemi-L_mesh"
    )
    assert (tmp_path / "give_me_a_default_suffix_hemi-L_mesh.gii").exists()


def test_save_mesh_error(tmp_path, mini_img):
    with pytest.raises(ValueError, match="cannot contain both"):
        mini_img.mesh.to_filename(
            tmp_path / "hemi-L_hemi-R_cannot_have_both.gii"
        )


def test_save_mesh_error_wrong_suffix(tmp_path, mini_img):
    with pytest.raises(ValueError, match="with the extension '.gii'"):
        mini_img.mesh.to_filename(
            tmp_path / "hemi-L_hemi-R_cannot_have_both.foo"
        )


@pytest.mark.parametrize("use_path", [True, False])
@pytest.mark.parametrize(
    "output_filename, expected_files, unexpected_files",
    [
        ("foo.gii", ["foo_hemi-L.gii", "foo_hemi-L.gii"], ["foo.gii"]),
        ("foo_hemi-L_T1w.gii", ["foo_hemi-L_T1w.gii"], ["foo_hemi-R_T1w.gii"]),
        ("foo_hemi-R_T1w.gii", ["foo_hemi-R_T1w.gii"], ["foo_hemi-L_T1w.gii"]),
    ],
)
def test_load_save_data(
    tmp_path, output_filename, expected_files, unexpected_files, use_path
):
    mesh_right = datasets.fetch_surf_fsaverage().pial_right
    mesh_left = datasets.fetch_surf_fsaverage().pial_left
    data_right = datasets.fetch_surf_fsaverage().sulc_right
    data_left = datasets.fetch_surf_fsaverage().sulc_left

    if use_path:
        img = SurfaceImage(
            mesh={"left": Path(mesh_left), "right": Path(mesh_right)},
            data={"left": Path(data_left), "right": Path(data_right)},
        )
    else:
        img = SurfaceImage(
            mesh={"left": mesh_left, "right": mesh_right},
            data={"left": data_left, "right": data_right},
        )

    if use_path:
        img.data.to_filename(tmp_path / output_filename)
    else:
        img.data.to_filename(str(tmp_path / output_filename))

    for file in unexpected_files:
        assert not (tmp_path / file).exists()

    for file in expected_files:
        assert (tmp_path / file).exists()

        data = load_surf_data(tmp_path / file)
        if "hemi-L" in file:
            expected_data = load_surf_data(data_left)
        elif "hemi-R" in file:
            expected_data = load_surf_data(data_right)
        assert np.array_equal(data, expected_data)


def test_load_from_volume_3D_nifti(img_3d_mni, mini_mesh, tmp_path):
    """Instantiate surface image with 3D Niftiimage object or file for data."""
    SurfaceImage.from_volume(mesh=mini_mesh, volume_img=img_3d_mni)

    img_3d_mni.to_filename(tmp_path / "tmp.nii.gz")

    SurfaceImage.from_volume(
        mesh=mini_mesh,
        volume_img=tmp_path / "tmp.nii.gz",
    )


def test_load_from_volume_4D_nifti(img_4d_mni, mini_mesh, tmp_path):
    """Instantiate surface image with 4D Niftiimage object or file for data."""
    img = SurfaceImage.from_volume(mesh=mini_mesh, volume_img=img_4d_mni)
    # check that we have the correct number of time points
    assert img.shape[0] == img_4d_mni.shape[3]

    img_4d_mni.to_filename(tmp_path / "tmp.nii.gz")

    SurfaceImage.from_volume(
        mesh=mini_mesh,
        volume_img=tmp_path / "tmp.nii.gz",
    )


def test_surface_image_error():
    """Instantiate surface image with Niftiimage object or file for data."""
    mesh_right = datasets.fetch_surf_fsaverage().pial_right
    mesh_left = datasets.fetch_surf_fsaverage().pial_left

    with pytest.raises(TypeError, match="[PolyData, dict]"):
        SurfaceImage(mesh={"left": mesh_left, "right": mesh_right}, data=3)
