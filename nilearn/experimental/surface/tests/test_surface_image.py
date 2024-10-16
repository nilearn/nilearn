from pathlib import Path

import numpy as np
import pytest

from nilearn import datasets
from nilearn.experimental.surface import FileMesh, InMemoryMesh, SurfaceImage
from nilearn.surface import load_surf_data, load_surf_mesh


def test_compare_file_and_inmemory_mesh(make_mesh, tmp_path):
    mesh = make_mesh()
    left = mesh.parts["left"]
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


@pytest.mark.parametrize("shape", [(1,), (3,), (7, 3)])
def test_surface_image_shape(make_surface_img, shape):
    img = make_surface_img(shape)
    assert img.shape == (*shape, 9)


def test_data_shape_not_matching_mesh(make_surface_img, flip):
    img = make_surface_img()
    with pytest.raises(ValueError, match="shape.*vertices"):
        SurfaceImage(img.mesh, flip(img.data))


def test_data_shape_inconsistent(make_surface_img):
    img = make_surface_img((7,))
    bad_data = {
        "left": img.data.parts["left"],
        "right": img.data.parts["right"][:4],
    }
    with pytest.raises(ValueError, match="incompatible shapes"):
        SurfaceImage(img.mesh, bad_data)


def test_data_keys_not_matching_mesh(make_surface_img):
    with pytest.raises(ValueError, match="same keys"):
        img = make_surface_img()
        SurfaceImage(
            {"left": img.mesh.parts["left"]},
            img.data,
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


def test_save_mesh_default_suffix(tmp_path, make_surface_img):
    """Check default .gii extension is added."""
    make_surface_img().mesh.to_filename(
        tmp_path / "give_me_a_default_suffix_hemi-L_mesh"
    )
    assert (tmp_path / "give_me_a_default_suffix_hemi-L_mesh.gii").exists()


def test_save_mesh_error(tmp_path, make_surface_img):
    with pytest.raises(ValueError, match="cannot contain both"):
        make_surface_img().mesh.to_filename(
            tmp_path / "hemi-L_hemi-R_cannot_have_both.gii"
        )


def test_save_mesh_error_wrong_suffix(tmp_path, make_surface_img):
    with pytest.raises(ValueError, match="with the extension '.gii'"):
        make_surface_img().mesh.to_filename(
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


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_save_dtype(make_surface_img, tmp_path, dtype):
    """Check saving several data type."""
    img = make_surface_img()
    img.data.parts["right"] = img.data.parts["right"].astype(dtype)
    img.data.to_filename(tmp_path / "data.gii")


def test_load_from_volume_3d_nifti(img_3d_mni, make_mesh, tmp_path):
    """Instantiate surface image with 3D Niftiimage object or file for data."""
    mesh = make_mesh()
    SurfaceImage.from_volume(mesh=mesh, volume_img=img_3d_mni)

    img_3d_mni.to_filename(tmp_path / "tmp.nii.gz")

    SurfaceImage.from_volume(
        mesh=mesh,
        volume_img=tmp_path / "tmp.nii.gz",
    )


def test_load_from_volume_4d_nifti(img_4d_mni, make_mesh, tmp_path):
    """Instantiate surface image with 4D Niftiimage object or file for data."""
    mesh = make_mesh()
    img = SurfaceImage.from_volume(mesh=mesh, volume_img=img_4d_mni)
    # check that we have the correct number of time points
    assert img.shape[0] == img_4d_mni.shape[3]

    img_4d_mni.to_filename(tmp_path / "tmp.nii.gz")

    SurfaceImage.from_volume(
        mesh=mesh,
        volume_img=tmp_path / "tmp.nii.gz",
    )


def test_surface_image_error():
    """Instantiate surface image with Niftiimage object or file for data."""
    mesh_right = datasets.fetch_surf_fsaverage().pial_right
    mesh_left = datasets.fetch_surf_fsaverage().pial_left

    with pytest.raises(TypeError, match="[PolyData, dict]"):
        SurfaceImage(mesh={"left": mesh_left, "right": mesh_right}, data=3)
