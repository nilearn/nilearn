# Tests for functions in surf_plotting.py

import warnings
from pathlib import Path

import numpy as np
import pytest
from nibabel import Nifti1Image, freesurfer, gifti, load, nifti1
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial import Delaunay
from scipy.stats import pearsonr
from sklearn.exceptions import EfficiencyWarning

from nilearn import datasets, image
from nilearn._utils import data_gen
from nilearn.image import resampling
from nilearn.surface import (
    FileMesh,
    InMemoryMesh,
    PolyData,
    PolyMesh,
    SurfaceImage,
    load_surf_data,
    load_surf_mesh,
    vol_to_surf,
)
from nilearn.surface.surface import (
    _choose_kind,
    _data_to_gifti,
    _gifti_img_to_mesh,
    _interpolation_sampling,
    _load_surf_files_gifti_gzip,
    _load_uniform_ball_cloud,
    _masked_indices,
    _mesh_to_gifti,
    _nearest_voxel_sampling,
    _projection_matrix,
    _sample_locations,
    _sample_locations_between_surfaces,
    _uniform_ball_cloud,
    _vertex_outer_normals,
    check_mesh_and_data,
    check_mesh_is_fsaverage,
)

datadir = Path(__file__).resolve().parent / "data"


def flat_mesh(x_s, y_s, z=0):
    """Create a flat horizontal mesh."""
    x, y = np.mgrid[:x_s, :y_s]
    x, y = x.ravel(), y.ravel()
    z = np.ones(len(x)) * z
    vertices = np.asarray([x, y, z]).T
    triangulation = Delaunay(vertices[:, :2]).simplices
    return InMemoryMesh(coordinates=vertices, faces=triangulation)


def z_const_img(x_s, y_s, z_s):
    """Create an image that is constant in z direction."""
    hslice = np.arange(x_s * y_s).reshape((x_s, y_s))
    return np.ones((x_s, y_s, z_s)) * hslice[:, :, np.newaxis]


def test_check_mesh():
    mesh = check_mesh_is_fsaverage("fsaverage5")
    assert mesh is check_mesh_is_fsaverage(mesh)
    with pytest.raises(ValueError):
        check_mesh_is_fsaverage("fsaverage2")
    mesh.pop("pial_left")
    with pytest.raises(ValueError):
        check_mesh_is_fsaverage(mesh)
    with pytest.raises(TypeError):
        check_mesh_is_fsaverage(load_surf_mesh(mesh["pial_right"]))


def test_check_mesh_and_data(rng, in_memory_mesh):
    data = in_memory_mesh.coordinates[:, 0]

    m, d = check_mesh_and_data(in_memory_mesh, data)
    assert (m[0] == in_memory_mesh.coordinates).all()
    assert (m[1] == in_memory_mesh.faces).all()
    assert (d == data).all()

    # Generate faces such that max index is larger than
    # the length of coordinates array.
    wrong_faces = rng.integers(in_memory_mesh.n_vertices + 1, size=(30, 3))
    wrong_mesh = InMemoryMesh(in_memory_mesh.coordinates, wrong_faces)

    # Check that check_mesh_and_data raises an error
    # with the resulting wrong mesh
    with pytest.raises(
        ValueError,
        match="Mismatch between .* indices of faces .* number of nodes.",
    ):
        check_mesh_and_data(wrong_mesh, data)

    # Alter the data and check that an error is raised
    data = in_memory_mesh.coordinates[::2, 0]
    with pytest.raises(
        ValueError, match="Mismatch between number of nodes in mesh"
    ):
        check_mesh_and_data(in_memory_mesh, data)


def test_load_surf_data_numpy_gt_1pt23():
    """Test loading fsaverage surface.

    Threw an error with numpy >=1.24.x
    but only a deprecaton warning with numpy <1.24.x.

    Regression test for
    https://github.com/nilearn/nilearn/issues/3638
    """
    fsaverage = datasets.fetch_surf_fsaverage()
    load_surf_data(fsaverage["pial_left"])


def test_load_surf_data_array():
    # test loading and squeezing data from numpy array
    data_flat = np.zeros((20,))
    data_squeeze = np.zeros((20, 1, 3))
    assert_array_equal(load_surf_data(data_flat), np.zeros((20,)))
    assert_array_equal(load_surf_data(data_squeeze), np.zeros((20, 3)))


def test_load_surf_data_from_gifti_file(tmp_path):
    filename_gii = tmp_path / "tmp.gii"
    darray = gifti.GiftiDataArray(
        data=np.zeros((20,)), datatype="NIFTI_TYPE_FLOAT32"
    )
    gii = gifti.GiftiImage(darrays=[darray])
    gii.to_filename(filename_gii)
    assert_array_equal(load_surf_data(filename_gii), np.zeros((20,)))


def test_load_surf_data_from_empty_gifti_file(tmp_path):
    filename_gii_empty = tmp_path / "tmp.gii"
    gii_empty = gifti.GiftiImage()
    gii_empty.to_filename(filename_gii_empty)
    with pytest.raises(
        ValueError, match="must contain at least one data array"
    ):
        load_surf_data(filename_gii_empty)


def test_load_surf_data_from_nifti_file(tmp_path):
    filename_nii = tmp_path / "tmp.nii"
    filename_niigz = tmp_path / "tmp.nii.gz"
    nii = Nifti1Image(np.zeros((20,)), affine=None)
    nii.to_filename(filename_nii)
    nii.to_filename(filename_niigz)
    assert_array_equal(load_surf_data(filename_nii), np.zeros((20,)))
    assert_array_equal(load_surf_data(filename_niigz), np.zeros((20,)))


def test_load_surf_data_gii_gz():
    # Test the loader `load_surf_data` with gzipped fsaverage5 files

    # surface data
    fsaverage = datasets.fetch_surf_fsaverage().sulc_left
    gii = _load_surf_files_gifti_gzip(fsaverage)
    assert isinstance(gii, gifti.GiftiImage)

    data = load_surf_data(fsaverage)
    assert isinstance(data, np.ndarray)

    # surface mesh
    fsaverage = datasets.fetch_surf_fsaverage().pial_left
    gii = _load_surf_files_gifti_gzip(fsaverage)
    assert isinstance(gii, gifti.GiftiImage)


def test_load_surf_data_file_freesurfer(tmp_path):
    # test loading of fake data from sulc and thickness files
    # using load_surf_data.
    # We test load_surf_data by creating fake data with function
    # 'write_morph_data' that works only if nibabel
    # version is recent with nibabel >= 2.1.0
    filename_area = tmp_path / "tmp.area"
    data = np.zeros((20,))
    freesurfer.io.write_morph_data(filename_area, data)
    assert_array_equal(load_surf_data(filename_area), np.zeros((20,)))

    filename_curv = tmp_path / "tmp.curv"
    freesurfer.io.write_morph_data(filename_curv, data)
    assert_array_equal(load_surf_data(filename_curv), np.zeros((20,)))

    filename_sulc = tmp_path / "tmp.sulc"
    freesurfer.io.write_morph_data(filename_sulc, data)
    assert_array_equal(load_surf_data(filename_sulc), np.zeros((20,)))

    filename_thick = tmp_path / "tmp.thickness"
    freesurfer.io.write_morph_data(filename_thick, data)
    assert_array_equal(load_surf_data(filename_thick), np.zeros((20,)))

    # test loading of data from real label and annot files
    label_start = np.array([5900, 5899, 5901, 5902, 2638])
    label_end = np.array([8756, 6241, 8757, 1896, 6243])
    label = load_surf_data(datadir / "test.label")
    assert_array_equal(label[:5], label_start)
    assert_array_equal(label[-5:], label_end)
    assert label.shape == (10,)
    del label, label_start, label_end

    annot_start = np.array([24, 29, 28, 27, 24, 31, 11, 25, 0, 12])
    annot_end = np.array([16, 16, 16, 16, 16, 16, 16, 16, 16, 16])
    annot = load_surf_data(datadir / "test.annot")
    assert_array_equal(annot[:10], annot_start)
    assert_array_equal(annot[-10:], annot_end)
    assert annot.shape == (10242,)
    del annot, annot_start, annot_end


@pytest.mark.parametrize("suffix", [".vtk", ".obj", ".mnc", ".txt"])
def test_load_surf_data_file_error(tmp_path, suffix):
    # test if files with unexpected suffixes raise errors
    data = np.zeros((20,))
    filename_wrong = tmp_path / f"tmp{suffix}"
    np.savetxt(filename_wrong, data)
    with pytest.raises(ValueError, match="input type is not recognized"):
        load_surf_data(filename_wrong)


def test_load_surf_mesh_list(in_memory_mesh):
    # test if correct list is returned
    assert hasattr(load_surf_mesh(in_memory_mesh), "coordinates")
    assert hasattr(load_surf_mesh(in_memory_mesh), "faces")
    assert_array_equal(
        load_surf_mesh(in_memory_mesh).coordinates, in_memory_mesh.coordinates
    )
    assert_array_equal(
        load_surf_mesh(in_memory_mesh).faces, in_memory_mesh.faces
    )

    # test if incorrect list, array or dict raises error
    with pytest.raises(ValueError, match="it must have two elements"):
        load_surf_mesh([])
    with pytest.raises(ValueError, match="it must have two elements"):
        load_surf_mesh([in_memory_mesh.coordinates])
    with pytest.raises(ValueError, match="it must have two elements"):
        load_surf_mesh(
            [
                in_memory_mesh.coordinates,
                in_memory_mesh.coordinates,
                in_memory_mesh.faces,
            ]
        )
    with pytest.raises(ValueError, match="input type is not recognized"):
        load_surf_mesh(in_memory_mesh.coordinates)
    with pytest.raises(ValueError, match="input type is not recognized"):
        load_surf_mesh({})


def test_gifti_img_to_mesh(in_memory_mesh):
    coord_array = gifti.GiftiDataArray(
        data=in_memory_mesh.coordinates, datatype="NIFTI_TYPE_FLOAT32"
    )
    coord_array.intent = nifti1.intent_codes["NIFTI_INTENT_POINTSET"]

    face_array = gifti.GiftiDataArray(
        data=in_memory_mesh.faces, datatype="NIFTI_TYPE_FLOAT32"
    )
    face_array.intent = nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]

    gii = gifti.GiftiImage(darrays=[coord_array, face_array])
    coords, faces = _gifti_img_to_mesh(gii)
    assert_array_equal(coords, in_memory_mesh.coordinates)
    assert_array_equal(faces, in_memory_mesh.faces)


def test_load_surf_mesh_file_gii_gz():
    # Test the loader `load_surf_mesh` with gzipped fsaverage5 files
    fsaverage = datasets.fetch_surf_fsaverage().pial_left
    mesh = load_surf_mesh(fsaverage)
    coords = mesh.coordinates
    faces = mesh.faces
    assert isinstance(coords, np.ndarray)
    assert isinstance(faces, np.ndarray)


def test_load_surf_mesh_file_gii(tmp_path, in_memory_mesh):
    # Test the loader `load_surf_mesh`
    # test if correct gii is loaded into correct list
    coord_array = gifti.GiftiDataArray(
        data=in_memory_mesh.coordinates,
        intent=nifti1.intent_codes["NIFTI_INTENT_POINTSET"],
        datatype="NIFTI_TYPE_FLOAT32",
    )
    face_array = gifti.GiftiDataArray(
        data=in_memory_mesh.faces,
        intent=nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"],
        datatype="NIFTI_TYPE_FLOAT32",
    )

    gii = gifti.GiftiImage(darrays=[coord_array, face_array])
    filename_gii_mesh = tmp_path / "tmp.gii"
    gii.to_filename(filename_gii_mesh)

    assert_array_almost_equal(
        load_surf_mesh(filename_gii_mesh).coordinates,
        in_memory_mesh.coordinates,
    )
    assert_array_almost_equal(
        load_surf_mesh(filename_gii_mesh).faces, in_memory_mesh.faces
    )


def test_load_surf_mesh_file_gii_error(tmp_path, in_memory_mesh):
    # test if incorrect gii raises error
    coord_array = gifti.GiftiDataArray(
        data=in_memory_mesh.coordinates,
        intent=nifti1.intent_codes["NIFTI_INTENT_POINTSET"],
        datatype="NIFTI_TYPE_FLOAT32",
    )
    face_array = gifti.GiftiDataArray(
        data=in_memory_mesh.faces,
        intent=nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"],
        datatype="NIFTI_TYPE_FLOAT32",
    )

    filename_gii_mesh_no_point = tmp_path / "tmp.gii"
    gii = gifti.GiftiImage(darrays=[face_array, face_array])
    gii.to_filename(filename_gii_mesh_no_point)

    with pytest.raises(ValueError, match="NIFTI_INTENT_POINTSET"):
        load_surf_mesh(filename_gii_mesh_no_point)

    filename_gii_mesh_no_face = tmp_path / "tmp.gii"
    gii = gifti.GiftiImage(darrays=[coord_array, coord_array])
    gii.to_filename(filename_gii_mesh_no_face)

    with pytest.raises(ValueError, match="NIFTI_INTENT_TRIANGLE"):
        load_surf_mesh(filename_gii_mesh_no_face)


@pytest.mark.parametrize(
    "suffix", [".pial", ".inflated", ".white", ".orig", "sphere"]
)
def test_load_surf_mesh_file_freesurfer(suffix, tmp_path, in_memory_mesh):
    filename_fs_mesh = tmp_path / f"tmp{suffix}"
    freesurfer.write_geometry(
        filename_fs_mesh, in_memory_mesh.coordinates, in_memory_mesh.faces
    )

    assert hasattr(load_surf_mesh(filename_fs_mesh), "coordinates")
    assert hasattr(load_surf_mesh(filename_fs_mesh), "faces")
    assert_array_almost_equal(
        load_surf_mesh(filename_fs_mesh).coordinates,
        in_memory_mesh.coordinates,
    )
    assert_array_almost_equal(
        load_surf_mesh(filename_fs_mesh).faces, in_memory_mesh.faces
    )


@pytest.mark.parametrize("suffix", [".vtk", ".obj", ".mnc", ".txt"])
def test_load_surf_mesh_file_error(suffix, tmp_path, in_memory_mesh):
    # test if files with unexpected suffixes raise errors
    filename_wrong = tmp_path / f"tmp{suffix}"
    freesurfer.write_geometry(
        filename_wrong, in_memory_mesh.coordinates, in_memory_mesh.faces
    )

    with pytest.raises(ValueError, match="input type is not recognized"):
        load_surf_mesh(filename_wrong)


def test_load_surf_mesh_file_glob(tmp_path, in_memory_mesh):
    fname1 = tmp_path / "tmp1.pial"
    freesurfer.write_geometry(
        fname1, in_memory_mesh.coordinates, in_memory_mesh.faces
    )

    fname2 = tmp_path / "tmp2.pial"
    freesurfer.write_geometry(
        fname2, in_memory_mesh.coordinates, in_memory_mesh.faces
    )

    with pytest.raises(ValueError, match="More than one file matching path"):
        load_surf_mesh(tmp_path / "*.pial")
    with pytest.raises(ValueError, match="No files matching path"):
        load_surf_mesh(tmp_path / "*.unlikelysuffix")
    assert hasattr(load_surf_mesh(fname1), "coordinates")
    assert hasattr(load_surf_mesh(fname1), "faces")
    assert_array_almost_equal(
        load_surf_mesh(fname1).coordinates, in_memory_mesh.coordinates
    )
    assert_array_almost_equal(
        load_surf_mesh(fname1).faces, in_memory_mesh.faces
    )


def test_load_surf_data_file_glob(tmp_path):
    data2D = np.ones((20, 3))
    fnames = []
    for f in range(3):
        filename = tmp_path / f"glob_{f}_tmp.gii"
        fnames.append(filename)
        data2D[:, f] *= f
        darray = gifti.GiftiDataArray(
            data=data2D[:, f], datatype="NIFTI_TYPE_FLOAT32"
        )
        gii = gifti.GiftiImage(darrays=[darray])
        gii.to_filename(fnames[f])

    assert_array_equal(
        load_surf_data(tmp_path / "glob*.gii"),
        data2D,
    )

    # make one more gii file that has more than one dimension
    filename = tmp_path / "glob_3_tmp.gii"
    fnames.append(filename)
    darray1 = gifti.GiftiDataArray(
        data=np.ones((20,)), datatype="NIFTI_TYPE_FLOAT32"
    )
    gii = gifti.GiftiImage(darrays=[darray1, darray1, darray1])
    gii.to_filename(fnames[-1])

    data2D = np.concatenate((data2D, np.ones((20, 3))), axis=1)
    assert_array_equal(
        load_surf_data(tmp_path / "glob*.gii"),
        data2D,
    )

    # make one more gii file that has a different shape in axis=0
    filename = tmp_path / "glob_4_tmp.gii"
    fnames.append(filename)
    darray = gifti.GiftiDataArray(
        data=np.ones((15, 1)), datatype="NIFTI_TYPE_FLOAT32"
    )
    gii = gifti.GiftiImage(darrays=[darray])
    gii.to_filename(fnames[-1])

    with pytest.raises(
        ValueError, match="files must contain data with the same shape"
    ):
        load_surf_data(tmp_path / "*.gii")


@pytest.mark.parametrize("xy", [(10, 7), (5, 5), (3, 2)])
def test_flat_mesh(xy):
    mesh = flat_mesh(xy[0], xy[1])
    points = mesh.coordinates
    triangles = mesh.faces
    a, b, c = points[triangles[0]]
    n = np.cross(b - a, c - a)
    assert np.allclose(n, [0.0, 0.0, 1.0])


def test_vertex_outer_normals():
    # compute normals for a flat horizontal mesh, they should all be (0, 0, 1)
    mesh = flat_mesh(5, 7)
    computed_normals = _vertex_outer_normals(mesh)
    true_normals = np.zeros((len(mesh.coordinates), 3))
    true_normals[:, 2] = 1
    assert_array_almost_equal(computed_normals, true_normals)


def test_load_uniform_ball_cloud():
    # Note: computed and shipped point clouds may differ since KMeans results
    # change after
    # https://github.com/scikit-learn/scikit-learn/pull/9288
    # but the exact position of the points does not matter as long as they are
    # well spread inside the unit ball
    for n_points in [10, 20, 40, 80, 160]:
        with warnings.catch_warnings(record=True) as w:
            points = _load_uniform_ball_cloud(n_points=n_points)
            assert_array_equal(points.shape, (n_points, 3))
            assert len(w) == 0
    with pytest.warns(EfficiencyWarning):
        _load_uniform_ball_cloud(n_points=3)
    for n_points in [3, 7]:
        computed = _uniform_ball_cloud(n_points)
        loaded = _load_uniform_ball_cloud(n_points)
        assert_array_almost_equal(computed, loaded)
        assert (np.std(computed, axis=0) > 0.1).all()
        assert (np.linalg.norm(computed, axis=1) <= 1).all()


def test_sample_locations():
    # check positions of samples on toy example, with an affine != identity
    # flat horizontal mesh
    mesh = flat_mesh(5, 7)
    affine = np.diagflat([10, 20, 30, 1])
    inv_affine = np.linalg.inv(affine)
    # transform vertices to world space
    vertices = np.asarray(
        resampling.coord_transform(*mesh.coordinates.T, affine=affine)
    ).T
    # compute by hand the true offsets in voxel space
    # (transformed by affine^-1)
    ball_offsets = _load_uniform_ball_cloud(10)
    ball_offsets = np.asarray(
        resampling.coord_transform(*ball_offsets.T, affine=inv_affine)
    ).T
    line_offsets = np.zeros((10, 3))
    line_offsets[:, 2] = np.linspace(1, -1, 10)
    line_offsets = np.asarray(
        resampling.coord_transform(*line_offsets.T, affine=inv_affine)
    ).T
    # check we get the same locations
    for kind, offsets in [("line", line_offsets), ("ball", ball_offsets)]:
        locations = _sample_locations(
            [vertices, mesh.faces], affine, 1.0, kind=kind, n_points=10
        )
        true_locations = np.asarray(
            [vertex + offsets for vertex in mesh.coordinates]
        )
        assert_array_equal(locations.shape, true_locations.shape)
        assert_array_almost_equal(true_locations, locations)
    with pytest.raises(ValueError):
        _sample_locations(mesh, affine, 1.0, kind="bad_kind")


@pytest.mark.parametrize("depth", [(0.0,), (-1.0,), (1.0,), (-1.0, 0.0, 0.5)])
@pytest.mark.parametrize("n_points", [None, 10])
def test_sample_locations_depth(depth, n_points, affine_eye):
    mesh = flat_mesh(5, 7)
    radius = 8.0
    locations = _sample_locations(
        mesh, affine_eye, radius, n_points=n_points, depth=depth
    )
    offsets = np.asarray([[0.0, 0.0, -z * radius] for z in depth])
    expected = np.asarray([vertex + offsets for vertex in mesh.coordinates])
    assert np.allclose(locations, expected)


@pytest.mark.parametrize(
    "depth,n_points",
    [
        (None, 1),
        (None, 7),
        ([0.0], 8),
        ([-1.0], 8),
        ([1.0], 8),
        ([-1.0, 0.0, 0.5], 8),
    ],
)
def test_sample_locations_between_surfaces(depth, n_points, affine_eye):
    inner = flat_mesh(5, 7)
    outer = InMemoryMesh(
        coordinates=inner.coordinates + np.asarray([0.0, 0.0, 1.0]),
        faces=inner.faces,
    )
    locations = _sample_locations_between_surfaces(
        outer, inner, affine_eye, n_points=n_points, depth=depth
    )

    if depth is None:
        expected = np.asarray(
            [
                np.linspace(b, a, n_points)
                for (a, b) in zip(
                    inner.coordinates.ravel(), outer.coordinates.ravel()
                )
            ]
        )
        expected = np.rollaxis(
            expected.reshape((*outer.coordinates.shape, n_points)), 2, 1
        )

    else:
        offsets = [[0.0, 0.0, -z] for z in depth]
        expected = np.asarray(
            [vertex + offsets for vertex in outer.coordinates]
        )

    assert np.allclose(locations, expected)


def test_depth_ball_sampling():
    img, *_ = data_gen.generate_mni_space_img()
    mesh = load_surf_mesh(datasets.fetch_surf_fsaverage()["pial_left"])
    with pytest.raises(ValueError, match=".*does not support.*"):
        vol_to_surf(img, mesh, kind="ball", depth=[0.5])


@pytest.mark.parametrize("kind", ["line", "ball"])
@pytest.mark.parametrize("n_scans", [1, 20])
@pytest.mark.parametrize("use_mask", [True, False])
def test_vol_to_surf(kind, n_scans, use_mask):
    img, mask_img = data_gen.generate_mni_space_img(n_scans)
    if not use_mask:
        mask_img = None
    if n_scans == 1:
        img = image.new_img_like(img, image.get_data(img).squeeze())
    fsaverage = datasets.fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_left"])
    inner_mesh = load_surf_mesh(fsaverage["white_left"])
    center_mesh = (
        np.mean([mesh.coordinates, inner_mesh.coordinates], axis=0),
        mesh.faces,
    )
    proj = vol_to_surf(
        img, mesh, kind="depth", inner_mesh=inner_mesh, mask_img=mask_img
    )
    other_proj = vol_to_surf(img, center_mesh, kind=kind, mask_img=mask_img)
    correlation = pearsonr(proj.ravel(), other_proj.ravel())[0]
    assert correlation > 0.99
    with pytest.raises(ValueError, match=".*interpolation.*"):
        vol_to_surf(img, mesh, interpolation="bad")


def test_masked_indices():
    mask = np.ones((4, 3, 8))
    mask[:, :, ::2] = 0
    locations = np.mgrid[:5, :3, :8].ravel().reshape((3, -1))
    masked = _masked_indices(locations.T, mask.shape, mask)
    # These elements are masked by the mask
    assert (masked[::2] == 1).all()
    # The last element of locations is one row beyond first image dimension
    assert (masked[-24:] == 1).all()
    # 4 * 3 * 8 / 2 elements should remain unmasked
    assert (1 - masked).sum() == 48


def test_projection_matrix(affine_eye):
    mesh = flat_mesh(5, 7, 4)
    img = z_const_img(5, 7, 13)
    proj = _projection_matrix(
        mesh, affine_eye, img.shape, radius=2.0, n_points=10
    )
    # proj matrix has shape (n_vertices, img_size)
    assert proj.shape == (5 * 7, 5 * 7 * 13)
    # proj.dot(img) should give the values of img at the vertices' locations
    values = proj.dot(img.ravel()).reshape((5, 7))
    assert_array_almost_equal(values, img[:, :, 0])
    mesh = flat_mesh(5, 7)
    proj = _projection_matrix(
        mesh, affine_eye, (5, 7, 1), radius=0.1, n_points=10
    )
    assert_array_almost_equal(proj.toarray(), np.eye(proj.shape[0]))
    mask = np.ones(img.shape, dtype=int)
    mask[0] = 0
    proj = _projection_matrix(
        mesh, affine_eye, img.shape, radius=2.0, n_points=10, mask=mask
    )
    proj = proj.toarray()
    # first row of the mesh is masked
    assert_array_almost_equal(proj.sum(axis=1)[:7], np.zeros(7))
    assert_array_almost_equal(proj.sum(axis=1)[7:], np.ones(proj.shape[0] - 7))
    # mask and img should have the same shape
    with pytest.raises(ValueError):
        _projection_matrix(
            mesh, affine_eye, img.shape, mask=np.ones((3, 3, 2))
        )


def test_sampling_affine(affine_eye):
    # check sampled (projected) values on a toy image
    img = np.ones((4, 4, 4))
    img[1, :, :] = 2
    nodes = [[1, 1, 2], [10, 10, 20], [30, 30, 30]]
    mesh = [np.asarray(nodes), None]
    affine = 10 * affine_eye
    affine[-1, -1] = 1
    texture = _nearest_voxel_sampling(
        [img], mesh, affine=affine, radius=1, kind="ball"
    )
    assert_array_almost_equal(texture[0], [1.0, 2.0, 1.0], decimal=15)
    texture = _interpolation_sampling(
        [img], mesh, affine=affine, radius=0, kind="ball"
    )
    assert_array_almost_equal(texture[0], [1.1, 2.0, 1.0], decimal=15)


@pytest.mark.parametrize("kind", ["auto", "line", "ball"])
@pytest.mark.parametrize("use_inner_mesh", [True, False])
@pytest.mark.parametrize("projection", ["linear", "nearest"])
def test_sampling(kind, use_inner_mesh, projection, affine_eye):
    mesh = flat_mesh(5, 7, 4)
    img = z_const_img(5, 7, 13)
    mask = np.ones(img.shape, dtype=int)
    mask[0] = 0
    projector = {
        "nearest": _nearest_voxel_sampling,
        "linear": _interpolation_sampling,
    }[projection]
    inner_mesh = mesh if use_inner_mesh else None
    projection = projector(
        [img], mesh, affine_eye, kind=kind, radius=0.0, inner_mesh=inner_mesh
    )
    assert_array_almost_equal(projection.ravel(), img[:, :, 0].ravel())
    projection = projector(
        [img],
        mesh,
        affine_eye,
        kind=kind,
        radius=0.0,
        mask=mask,
        inner_mesh=inner_mesh,
    )
    assert_array_almost_equal(projection.ravel()[7:], img[1:, :, 0].ravel())
    assert np.isnan(projection.ravel()[:7]).all()


@pytest.mark.parametrize("projection", ["linear", "nearest"])
def test_sampling_between_surfaces(projection, affine_eye):
    projector = {
        "nearest": _nearest_voxel_sampling,
        "linear": _interpolation_sampling,
    }[projection]
    mesh = flat_mesh(13, 7, 3.0)
    inner_mesh = flat_mesh(13, 7, 1)
    img = z_const_img(5, 7, 13).T
    projection = projector(
        [img],
        mesh,
        affine_eye,
        kind="auto",
        n_points=100,
        inner_mesh=inner_mesh,
    )
    assert_array_almost_equal(
        projection.ravel(), img[:, :, 1:4].mean(axis=-1).ravel()
    )


def test_choose_kind():
    kind = _choose_kind("abc", None)
    assert kind == "abc"
    kind = _choose_kind("abc", "mesh")
    assert kind == "abc"
    kind = _choose_kind("auto", None)
    assert kind == "line"
    kind = _choose_kind("auto", "mesh")
    assert kind == "depth"
    with pytest.raises(TypeError, match=".*sampling strategy"):
        kind = _choose_kind("depth", None)


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
def test_data_to_gifti(rng, tmp_path, dtype):
    """Check saving several data type to gifti.

    - check that strings and Path work
    - make sure files can be loaded with nibabel
    """
    data = rng.random((5, 6)).astype(dtype)
    _data_to_gifti(data=data, gifti_file=tmp_path / "data.gii")
    _data_to_gifti(data=data, gifti_file=str(tmp_path / "data.gii"))
    load(tmp_path / "data.gii")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_data_to_gifti_unsupported_dtype(rng, tmp_path, dtype):
    """Check saving unsupported data type raises an error."""
    data = rng.random((5, 6)).astype(dtype)
    with pytest.raises(ValueError, match="supports uint8, int32 and float32"):
        _data_to_gifti(data=data, gifti_file=tmp_path / "data.gii")


@pytest.mark.parametrize("shape", [(5,), (5, 1), (5, 2)])
def test_polydata_shape(shape):
    data = PolyData(left=np.ones(shape), right=np.ones(shape))

    assert len(data.shape) == len(shape)
    assert data.shape[0] == shape[0] * 2

    data = PolyData(left=np.ones(shape))

    assert len(data.shape) == len(shape)
    assert data.shape[0] == shape[0]


def test_polydata_1d_check_parts():
    """Smoke test for check parts.

    - passing a 1D array at instantiation is fine:
      they are convertd to 2D
    - _check_parts can be used make sure parts are fine
      if assigned after instantiation.
    """
    data = PolyData(left=np.ones((7,)), right=np.ones((5,)))

    data.parts["left"] = np.ones((1,))

    data._check_parts()


def test_mesh_to_gifti(single_mesh, tmp_path):
    """Check saving mesh to gifti.

    - check that strings and Path work
    - make sure files can be loaded with nibabel
    """
    coordinates, faces = single_mesh
    _mesh_to_gifti(
        coordinates=coordinates, faces=faces, gifti_file=tmp_path / "mesh.gii"
    )
    _mesh_to_gifti(
        coordinates=coordinates,
        faces=faces,
        gifti_file=str(tmp_path / "mesh.gii"),
    )
    load(tmp_path / "mesh.gii")


def test_compare_file_and_inmemory_mesh(surf_mesh, tmp_path):
    mesh = surf_mesh()
    left = mesh.parts["left"]
    gifti_file = tmp_path / "left.gii"
    left.to_gifti(gifti_file)

    left_read = FileMesh(gifti_file)
    left_read.__repr__()  # for coverage
    assert left.n_vertices == left_read.n_vertices
    assert np.array_equal(left.coordinates, left_read.coordinates)
    assert np.array_equal(left.faces, left_read.faces)

    left_loaded = left_read.loaded()
    assert isinstance(left_loaded, InMemoryMesh)
    assert left.n_vertices == left_loaded.n_vertices
    assert np.array_equal(left.coordinates, left_loaded.coordinates)
    assert np.array_equal(left.faces, left_loaded.faces)


@pytest.mark.parametrize("shape", [1, 3])
def test_surface_image_shape(surf_img_2d, shape):
    assert surf_img_2d(shape).shape == (9, shape)


def test_data_shape_not_matching_mesh(surf_img_1d, flip_surf_img_parts):
    with pytest.raises(ValueError, match="shape.*vertices"):
        SurfaceImage(surf_img_1d.mesh, flip_surf_img_parts(surf_img_1d.data))


def test_data_shape_inconsistent(surf_img_2d):
    bad_data = {
        "left": surf_img_2d(7).data.parts["left"],
        "right": surf_img_2d(7).data.parts["right"][0][:4],
    }
    with pytest.raises(ValueError, match="incompatible shapes"):
        SurfaceImage(surf_img_2d(7).mesh, bad_data)


def test_data_keys_not_matching_mesh(surf_img_1d):
    with pytest.raises(ValueError, match="same keys"):
        SurfaceImage(
            {"left": surf_img_1d.mesh.parts["left"]},
            surf_img_1d.data,
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


def test_save_mesh_default_suffix(tmp_path, surf_img_1d):
    """Check default .gii extension is added."""
    surf_img_1d.mesh.to_filename(
        tmp_path / "give_me_a_default_suffix_hemi-L_mesh"
    )
    assert (tmp_path / "give_me_a_default_suffix_hemi-L_mesh.gii").exists()


def test_save_mesh_error(tmp_path, surf_img_1d):
    with pytest.raises(ValueError, match="cannot contain both"):
        surf_img_1d.mesh.to_filename(
            tmp_path / "hemi-L_hemi-R_cannot_have_both.gii"
        )


def test_save_mesh_error_wrong_suffix(tmp_path, surf_img_1d):
    with pytest.raises(ValueError, match="with the extension '.gii'"):
        surf_img_1d.mesh.to_filename(
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
    """Load and save gifti leaves them unchanged."""
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


def test_load_save_data_1d(rng, tmp_path, surf_mesh):
    """Load and save 1D gifti leaves them unchanged."""
    data = {}
    for hemi in ["left", "right"]:
        size = (surf_mesh().parts[hemi].n_vertices,)
        data[hemi] = rng.random(size=size).astype(np.uint8)
        darray = gifti.GiftiDataArray(
            data=data[hemi], datatype="NIFTI_TYPE_UINT8"
        )
        gii = gifti.GiftiImage(darrays=[darray])
        gii.to_filename(tmp_path / f"original_{hemi}.gii")

    img = SurfaceImage(
        mesh=surf_mesh(),
        data={
            "left": tmp_path / "original_left.gii",
            "right": tmp_path / "original_right.gii",
        },
    )

    img.data.to_filename(tmp_path / "nilearn.gii")

    for hemi in ["left", "right"]:
        original_surf_img = load(tmp_path / f"original_{hemi}.gii")
        nilearn_surf_img = load(
            tmp_path / f"nilearn_hemi-{hemi[0].upper()}.gii"
        )
        assert_array_equal(
            original_surf_img.agg_data(), nilearn_surf_img.agg_data()
        )


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
def test_save_dtype(surf_img_1d, tmp_path, dtype):
    """Check saving several data type."""
    surf_img_1d.data.parts["right"] = surf_img_1d.data.parts["right"].astype(
        dtype
    )
    surf_img_1d.data.to_filename(tmp_path / "data.gii")


def test_load_from_volume_3d_nifti(img_3d_mni, surf_mesh, tmp_path):
    """Instantiate surface image with 3D Niftiimage object or file for data."""
    mesh = surf_mesh()
    SurfaceImage.from_volume(mesh=mesh, volume_img=img_3d_mni)

    img_3d_mni.to_filename(tmp_path / "tmp.nii.gz")

    SurfaceImage.from_volume(
        mesh=mesh,
        volume_img=tmp_path / "tmp.nii.gz",
    )


def test_load_from_volume_4d_nifti(img_4d_mni, surf_mesh, tmp_path):
    """Instantiate surface image with 4D Niftiimage object or file for data."""
    img = SurfaceImage.from_volume(mesh=surf_mesh(), volume_img=img_4d_mni)
    # check that we have the correct number of time points
    assert img.shape[1] == img_4d_mni.shape[3]

    img_4d_mni.to_filename(tmp_path / "tmp.nii.gz")

    SurfaceImage.from_volume(
        mesh=surf_mesh(),
        volume_img=tmp_path / "tmp.nii.gz",
    )


def test_surface_image_error():
    """Instantiate surface image with Niftiimage object or file for data."""
    mesh_right = datasets.fetch_surf_fsaverage().pial_right
    mesh_left = datasets.fetch_surf_fsaverage().pial_left

    with pytest.raises(TypeError, match="[PolyData, dict]"):
        SurfaceImage(mesh={"left": mesh_left, "right": mesh_right}, data=3)


def test_polydata_error():
    with pytest.raises(ValueError, match="Either left or right"):
        PolyData(left=None, right=None)


def test_polymesh_error():
    with pytest.raises(ValueError, match="Either left or right"):
        PolyMesh(left=None, right=None)


def test_inmemorymesh_index_error(in_memory_mesh):
    with pytest.raises(
        IndexError, match="Use 0 for coordinates and 1 for faces"
    ):
        in_memory_mesh[2]
