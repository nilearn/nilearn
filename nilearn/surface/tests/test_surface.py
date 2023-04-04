# Tests for functions in surf_plotting.py

import os
import tempfile
import warnings

from collections import namedtuple

import nibabel as nb
import numpy as np
import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.spatial import Delaunay
from scipy.stats import pearsonr

from nibabel import gifti

from nilearn import datasets
from nilearn import image
from nilearn.image import resampling
from nilearn.image.tests.test_resampling import rotation
from nilearn.surface import Mesh, Surface
from nilearn.surface import surface
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf
from nilearn.surface.surface import (_gifti_img_to_mesh,
                                     _load_surf_files_gifti_gzip)
from nilearn.surface.testing_utils import (generate_surf, flat_mesh,
                                           z_const_img)
from nilearn._utils import data_gen

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


class MeshLikeObject:
    """Class with attributes coordinates and
    faces to be used for testing purposes.
    """
    def __init__(self, coordinates, faces):
        self._coordinates = coordinates
        self._faces = faces
    @property
    def coordinates(self):
        return self._coordinates
    @property
    def faces(self):
        return self._faces


class SurfaceLikeObject:
    """Class with attributes mesh and
    data to be used for testing purposes.
    """
    def __init__(self, mesh, data):
        self._mesh = mesh
        self._data = data
    @classmethod
    def fromarrays(cls, coordinates, faces, data):
        return cls(MeshLikeObject(coordinates, faces), data)
    @property
    def mesh(self):
        return self._mesh
    @property
    def data(self):
        return self._data


def test_load_surf_data_array():
    # test loading and squeezing data from numpy array
    data_flat = np.zeros((20, ))
    data_squeeze = np.zeros((20, 1, 3))
    assert_array_equal(load_surf_data(data_flat), np.zeros((20, )))
    assert_array_equal(load_surf_data(data_squeeze), np.zeros((20, 3)))


def test_load_surf_data_file_nii_gii(tmp_path):
    # test loading of fake data from gifti file
    fd_gii, filename_gii = tempfile.mkstemp(suffix='.gii',
                                            dir=str(tmp_path))
    os.close(fd_gii)
    darray = gifti.GiftiDataArray(
        data=np.zeros((20, )), datatype='NIFTI_TYPE_FLOAT32'
    )
    gii = gifti.GiftiImage(darrays=[darray])
    nb.save(gii, filename_gii)
    assert_array_equal(load_surf_data(filename_gii), np.zeros((20, )))
    os.remove(filename_gii)

    # test loading of data from empty gifti file
    fd_empty, filename_gii_empty = tempfile.mkstemp(suffix='.gii',
                                                    dir=str(tmp_path))
    os.close(fd_empty)
    gii_empty = gifti.GiftiImage()
    nb.save(gii_empty, filename_gii_empty)
    with pytest.raises(ValueError,
                       match='must contain at least one data array'
                       ):
        load_surf_data(filename_gii_empty)
    os.remove(filename_gii_empty)

    # test loading of fake data from nifti file
    fd_gii2, filename_nii = tempfile.mkstemp(suffix='.nii',
                                             dir=str(tmp_path))
    os.close(fd_gii2)
    fd_niigz, filename_niigz = tempfile.mkstemp(suffix='.nii.gz',
                                                dir=str(tmp_path))
    os.close(fd_niigz)
    nii = nb.Nifti1Image(np.zeros((20, )), affine=None)
    nb.save(nii, filename_nii)
    nb.save(nii, filename_niigz)
    assert_array_equal(load_surf_data(filename_nii), np.zeros((20, )))
    assert_array_equal(load_surf_data(filename_niigz), np.zeros((20, )))
    os.remove(filename_nii)
    os.remove(filename_niigz)


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
    data = np.zeros((20, ))
    fs_area, filename_area = tempfile.mkstemp(suffix='.area',
                                              dir=str(tmp_path))
    os.close(fs_area)
    nb.freesurfer.io.write_morph_data(filename_area, data)
    assert_array_equal(load_surf_data(filename_area), np.zeros((20, )))
    os.remove(filename_area)

    fs_curv, filename_curv = tempfile.mkstemp(suffix='.curv',
                                              dir=str(tmp_path))
    os.close(fs_curv)
    nb.freesurfer.io.write_morph_data(filename_curv, data)
    assert_array_equal(load_surf_data(filename_curv), np.zeros((20, )))
    os.remove(filename_curv)

    fd_sulc, filename_sulc = tempfile.mkstemp(suffix='.sulc',
                                              dir=str(tmp_path))
    os.close(fd_sulc)
    nb.freesurfer.io.write_morph_data(filename_sulc, data)
    assert_array_equal(load_surf_data(filename_sulc), np.zeros((20, )))
    os.remove(filename_sulc)

    fd_thick, filename_thick = tempfile.mkstemp(suffix='.thickness',
                                                dir=str(tmp_path))
    os.close(fd_thick)
    nb.freesurfer.io.write_morph_data(filename_thick, data)
    assert_array_equal(load_surf_data(filename_thick), np.zeros((20, )))
    os.remove(filename_thick)

    # test loading of data from real label and annot files
    label_start = np.array([5900, 5899, 5901, 5902, 2638])
    label_end = np.array([8756, 6241, 8757, 1896, 6243])
    label = load_surf_data(os.path.join(datadir, 'test.label'))
    assert_array_equal(label[:5], label_start)
    assert_array_equal(label[-5:], label_end)
    assert label.shape == (10, )
    del label, label_start, label_end

    annot_start = np.array([24, 29, 28, 27, 24, 31, 11, 25, 0, 12])
    annot_end = np.array([16, 16, 16, 16, 16, 16, 16, 16, 16, 16])
    annot = load_surf_data(os.path.join(datadir, 'test.annot'))
    assert_array_equal(annot[:10], annot_start)
    assert_array_equal(annot[-10:], annot_end)
    assert annot.shape == (10242, )
    del annot, annot_start, annot_end


def test_load_surf_data_file_error(tmp_path):
    # test if files with unexpected suffixes raise errors
    data = np.zeros((20, ))
    wrong_suff = ['.vtk', '.obj', '.mnc', '.txt']
    for suff in wrong_suff:
        fd, filename_wrong = tempfile.mkstemp(suffix=suff,
                                              dir=str(tmp_path))
        os.close(fd)
        np.savetxt(filename_wrong, data)
        with pytest.raises(ValueError,
                           match='input type is not recognized'
                           ):
            load_surf_data(filename_wrong)
        os.remove(filename_wrong)


def test_load_surf_mesh():
    coords, faces = generate_surf()
    mesh = Mesh(coords, faces)
    assert_array_equal(mesh.coordinates, coords)
    assert_array_equal(mesh.faces, faces)
    # Call load_surf_mesh with a Mesh as argument
    loaded_mesh = load_surf_mesh(mesh)
    assert isinstance(loaded_mesh, Mesh)
    assert_array_equal(mesh.coordinates, loaded_mesh.coordinates)
    assert_array_equal(mesh.faces, loaded_mesh.faces)

    mesh_like = MeshLikeObject(coords, faces)
    assert_array_equal(mesh_like.coordinates, coords)
    assert_array_equal(mesh_like.faces, faces)
    # Call load_surf_mesh with an object having
    # coordinates and faces attributes
    loaded_mesh = load_surf_mesh(mesh_like)
    assert isinstance(loaded_mesh, Mesh)
    assert_array_equal(mesh_like.coordinates, loaded_mesh.coordinates)
    assert_array_equal(mesh_like.faces, loaded_mesh.faces)


def test_load_surface():
    coords, faces = generate_surf()
    mesh = Mesh(coords, faces)
    data = mesh[0][:,0]
    surf = Surface(mesh, data)
    surf_like_obj = SurfaceLikeObject(mesh, data)
    # Load the surface from:
    #   - Surface-like objects having the right attributes
    #   - a list of length 2 (mesh, data)
    for loadings in [surf,
                     surf_like_obj,
                     [mesh, data]]:
        s = surface.load_surface(loadings)
        assert_array_equal(s.data, data)
        assert_array_equal(s.data, surf.data)
        assert_array_equal(s.mesh.coordinates, coords)
        assert_array_equal(s.mesh.coordinates, surf.mesh.coordinates)
        assert_array_equal(s.mesh.faces, surf.mesh.faces)
    # Giving an iterable of length other than 2 will raise an error
    # Length 3
    with pytest.raises(ValueError,
                       match="`load_surface` accepts iterables of length 2"):
        s = surface.load_surface([coords, faces, data])
    # Length 1
    with pytest.raises(ValueError,
                       match="`load_surface` accepts iterables of length 2"):
        s = surface.load_surface([coords])
    # Giving other objects will raise an error
    with pytest.raises(ValueError,
                       match="Wrong parameter `surface` in `load_surface`"):
        s = surface.load_surface("foo")


def test_load_surf_mesh_list():
    # test if correct list is returned
    mesh = generate_surf()
    assert len(load_surf_mesh(mesh)) == 2
    assert_array_equal(load_surf_mesh(mesh)[0], mesh[0])
    assert_array_equal(load_surf_mesh(mesh)[1], mesh[1])
    # test if incorrect list, array or dict raises error
    with pytest.raises(ValueError, match='it must have two elements'):
        load_surf_mesh([])
    with pytest.raises(ValueError, match='it must have two elements'):
        load_surf_mesh([mesh[0]])
    with pytest.raises(ValueError, match='it must have two elements'):
        load_surf_mesh([mesh[0], mesh[1], mesh[1]])
    with pytest.raises(ValueError, match='input type is not recognized'):
        load_surf_mesh(mesh[0])
    with pytest.raises(ValueError, match='input type is not recognized'):
        load_surf_mesh(dict())
    del mesh


def test_gifti_img_to_mesh():
    mesh = generate_surf()

    coord_array = gifti.GiftiDataArray(
        data=mesh[0], datatype='NIFTI_TYPE_FLOAT32'
    )
    coord_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET']

    face_array = gifti.GiftiDataArray(
        data=mesh[1], datatype='NIFTI_TYPE_FLOAT32'
    )
    face_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']

    gii = gifti.GiftiImage(darrays=[coord_array, face_array])
    coords, faces = _gifti_img_to_mesh(gii)
    assert_array_equal(coords, mesh[0])
    assert_array_equal(faces, mesh[1])


def test_load_surf_mesh_file_gii_gz():
    # Test the loader `load_surf_mesh` with gzipped fsaverage5 files

    fsaverage = datasets.fetch_surf_fsaverage().pial_left
    coords, faces = load_surf_mesh(fsaverage)
    assert isinstance(coords, np.ndarray)
    assert isinstance(faces, np.ndarray)


def test_load_surf_mesh_file_gii(tmp_path):
    # Test the loader `load_surf_mesh`
    mesh = generate_surf()

    # test if correct gii is loaded into correct list
    fd_mesh, filename_gii_mesh = tempfile.mkstemp(suffix='.gii',
                                                  dir=str(tmp_path))
    os.close(fd_mesh)
    coord_array = gifti.GiftiDataArray(
        data=mesh[0],
        intent=nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET'],
        datatype='NIFTI_TYPE_FLOAT32'
    )
    face_array = gifti.GiftiDataArray(
        data=mesh[1],
        intent=nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'],
        datatype='NIFTI_TYPE_FLOAT32'
    )

    gii = gifti.GiftiImage(darrays=[coord_array, face_array])
    nb.save(gii, filename_gii_mesh)
    assert_array_almost_equal(load_surf_mesh(filename_gii_mesh)[0], mesh[0])
    assert_array_almost_equal(load_surf_mesh(filename_gii_mesh)[1], mesh[1])
    os.remove(filename_gii_mesh)

    # test if incorrect gii raises error
    fd_no, filename_gii_mesh_no_point = tempfile.mkstemp(suffix='.gii',
                                                         dir=str(tmp_path))
    os.close(fd_no)
    nb.save(gifti.GiftiImage(darrays=[face_array, face_array]),
                filename_gii_mesh_no_point)
    with pytest.raises(ValueError, match='NIFTI_INTENT_POINTSET'):
        load_surf_mesh(filename_gii_mesh_no_point)
    os.remove(filename_gii_mesh_no_point)

    fd_face, filename_gii_mesh_no_face = tempfile.mkstemp(suffix='.gii',
                                                          dir=str(tmp_path))
    os.close(fd_face)
    nb.save(gifti.GiftiImage(darrays=[coord_array, coord_array]),
                filename_gii_mesh_no_face)
    with pytest.raises(ValueError, match='NIFTI_INTENT_TRIANGLE'):
        load_surf_mesh(filename_gii_mesh_no_face)
    os.remove(filename_gii_mesh_no_face)


def test_load_surf_mesh_file_freesurfer(tmp_path):
    mesh = generate_surf()
    for suff in ['.pial', '.inflated', '.white', '.orig', 'sphere']:
        fd, filename_fs_mesh = tempfile.mkstemp(suffix=suff,
                                                dir=str(tmp_path))
        os.close(fd)
        nb.freesurfer.write_geometry(filename_fs_mesh, mesh[0], mesh[1])
        assert len(load_surf_mesh(filename_fs_mesh)) == 2
        assert_array_almost_equal(load_surf_mesh(filename_fs_mesh)[0],
                                  mesh[0])
        assert_array_almost_equal(load_surf_mesh(filename_fs_mesh)[1],
                                  mesh[1])
        os.remove(filename_fs_mesh)


def test_load_surf_mesh_file_error(tmp_path):
    # test if files with unexpected suffixes raise errors
    mesh = generate_surf()
    wrong_suff = ['.vtk', '.obj', '.mnc', '.txt']
    for suff in wrong_suff:
        fd, filename_wrong = tempfile.mkstemp(suffix=suff,
                                              dir=str(tmp_path))
        os.close(fd)
        nb.freesurfer.write_geometry(filename_wrong, mesh[0], mesh[1])
        with pytest.raises(ValueError,
                           match='input type is not recognized'
                           ):
            load_surf_mesh(filename_wrong)
        os.remove(filename_wrong)


def test_load_surf_mesh_file_glob(tmp_path):
    mesh = generate_surf()
    fd1, fname1 = tempfile.mkstemp(suffix='.pial',
                                   dir=str(tmp_path))
    os.close(fd1)
    nb.freesurfer.write_geometry(fname1, mesh[0], mesh[1])
    fd2, fname2 = tempfile.mkstemp(suffix='.pial',
                                   dir=str(tmp_path))
    os.close(fd2)
    nb.freesurfer.write_geometry(fname2, mesh[0], mesh[1])

    with pytest.raises(ValueError, match='More than one file matching path'):
        load_surf_mesh(os.path.join(os.path.dirname(fname1), "*.pial"))
    with pytest.raises(ValueError, match='No files matching path'):
        load_surf_mesh(os.path.join(os.path.dirname(fname1),
                                    "*.unlikelysuffix")
                       )
    assert len(load_surf_mesh(fname1)) == 2
    assert_array_almost_equal(load_surf_mesh(fname1)[0], mesh[0])
    assert_array_almost_equal(load_surf_mesh(fname1)[1], mesh[1])
    os.remove(fname1)
    os.remove(fname2)


def test_load_surf_data_file_glob(tmp_path):

    data2D = np.ones((20, 3))
    fnames = []
    for f in range(3):
        fd, filename = tempfile.mkstemp(prefix='glob_%s_' % f,
                                        suffix='.gii',
                                        dir=str(tmp_path))
        os.close(fd)
        fnames.append(filename)
        data2D[:, f] *= f
        darray = gifti.GiftiDataArray(
            data=data2D[:, f], datatype='NIFTI_TYPE_FLOAT32'
        )
        gii = gifti.GiftiImage(darrays=[darray])
        nb.save(gii, fnames[f])

    assert_array_equal(load_surf_data(
        os.path.join(os.path.dirname(fnames[0]), "glob*.gii")),
        data2D
    )

    # make one more gii file that has more than one dimension
    fd, filename = tempfile.mkstemp(prefix='glob_3_',
                                    suffix='.gii',
                                    dir=str(tmp_path))
    os.close(fd)
    fnames.append(filename)
    darray1 = gifti.GiftiDataArray(
        data=np.ones((20, )), datatype='NIFTI_TYPE_FLOAT32'
    )
    gii = gifti.GiftiImage(darrays=[darray1, darray1, darray1])
    nb.save(gii, fnames[-1])

    data2D = np.concatenate((data2D, np.ones((20, 3))), axis=1)
    assert_array_equal(load_surf_data(os.path.join(os.path.dirname(fnames[0]),
                                                   "glob*.gii")), data2D)

    # make one more gii file that has a different shape in axis=0
    fd, filename = tempfile.mkstemp(prefix='glob_4_',
                                    suffix='.gii',
                                    dir=str(tmp_path))
    os.close(fd)
    fnames.append(filename)
    darray = gifti.GiftiDataArray(
        data=np.ones((15, 1)), datatype='NIFTI_TYPE_FLOAT32'
    )
    gii = gifti.GiftiImage(darrays=[darray])
    nb.save(gii, fnames[-1])

    with pytest.raises(ValueError,
                       match='files must contain data with the same shape'
                       ):
        load_surf_data(os.path.join(os.path.dirname(fnames[0]), "*.gii"))
    for f in fnames:
        os.remove(f)


def _flat_mesh(x_s, y_s, z=0):
    # outer normals point upwards ie [0, 0, 1]
    x, y = np.mgrid[:x_s, :y_s]
    x, y = x.ravel(), y.ravel()
    z = np.ones(len(x)) * z
    vertices = np.asarray([x, y, z]).T
    triangulation = Delaunay(vertices[:, :2]).simplices
    mesh = [vertices, triangulation]
    return mesh


@pytest.mark.parametrize("xy", [(10, 7), (5, 5), (3, 2)])
def test_flat_mesh(xy):
    points, triangles = _flat_mesh(xy[0], xy[1])
    a, b, c = points[triangles[0]]
    n = np.cross(b - a, c - a)
    assert np.allclose(n, [0., 0., 1.])


def _z_const_img(x_s, y_s, z_s):
    hslice = np.arange(x_s * y_s).reshape((x_s, y_s))
    return np.ones((x_s, y_s, z_s)) * hslice[:, :, np.newaxis]


def test_vertex_outer_normals():
    # compute normals for a flat horizontal mesh, they should all be (0, 0, 1)
    mesh = flat_mesh(5, 7)
    computed_normals = surface._vertex_outer_normals(mesh)
    true_normals = np.zeros((len(mesh[0]), 3))
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
            points = surface._load_uniform_ball_cloud(n_points=n_points)
            assert_array_equal(points.shape, (n_points, 3))
            assert len(w) == 0
    with pytest.warns(surface.EfficiencyWarning):
        surface._load_uniform_ball_cloud(n_points=3)
    for n_points in [3, 7]:
        computed = surface._uniform_ball_cloud(n_points)
        loaded = surface._load_uniform_ball_cloud(n_points)
        assert_array_almost_equal(computed, loaded)
        assert (np.std(computed, axis=0) > .1).all()
        assert (np.linalg.norm(computed, axis=1) <= 1).all()


def test_sample_locations():
    # check positions of samples on toy example, with an affine != identity
    # flat horizontal mesh
    mesh = flat_mesh(5, 7)
    affine = np.diagflat([10, 20, 30, 1])
    inv_affine = np.linalg.inv(affine)
    # transform vertices to world space
    vertices = np.asarray(
        resampling.coord_transform(*mesh[0].T, affine=affine)).T
    # compute by hand the true offsets in voxel space
    # (transformed by affine^-1)
    ball_offsets = surface._load_uniform_ball_cloud(10)
    ball_offsets = np.asarray(
        resampling.coord_transform(*ball_offsets.T, affine=inv_affine)).T
    line_offsets = np.zeros((10, 3))
    line_offsets[:, 2] = np.linspace(1, -1, 10)
    line_offsets = np.asarray(
        resampling.coord_transform(*line_offsets.T, affine=inv_affine)).T
    # check we get the same locations
    for kind, offsets in [('line', line_offsets), ('ball', ball_offsets)]:
        locations = surface._sample_locations(
            [vertices, mesh[1]], affine, 1., kind=kind, n_points=10)
        true_locations = np.asarray([vertex + offsets for vertex in mesh[0]])
        assert_array_equal(locations.shape, true_locations.shape)
        assert_array_almost_equal(true_locations, locations)
    pytest.raises(ValueError, surface._sample_locations,
                  mesh, affine, 1., kind='bad_kind')


@pytest.mark.parametrize("depth", [(0.,), (-1.,), (1.,), (-1., 0., .5)])
@pytest.mark.parametrize("n_points", [None, 10])
def test_sample_locations_depth(depth, n_points):
    mesh = flat_mesh(5, 7)
    radius = 8.
    locations = surface._sample_locations(
        mesh, np.eye(4), radius, n_points=n_points, depth=depth)
    offsets = np.asarray([[0., 0., - z * radius] for z in depth])
    expected = np.asarray([vertex + offsets for vertex in mesh[0]])
    assert np.allclose(locations, expected)


@pytest.mark.parametrize(
    "depth,n_points",
    [(None, 1), (None, 7), ([0.], 8), ([-1.], 8),
     ([1.], 8), ([-1., 0., .5], 8)])
def test_sample_locations_between_surfaces(depth, n_points):
    inner = flat_mesh(5, 7)
    outer = inner[0] + [0., 0., 1.], inner[1]
    locations = surface._sample_locations_between_surfaces(
        outer, inner, np.eye(4), n_points=n_points, depth=depth)
    if depth is None:
        # can be simplified when we drop support for np 1.15
        # (broadcasting linspace)
        expected = np.asarray(
            [np.linspace(b, a, n_points)
             for (a, b) in zip(inner[0].ravel(), outer[0].ravel())])
        expected = np.rollaxis(
            expected.reshape((*outer[0].shape, n_points)), 2, 1)
    else:
        offsets = ([[0., 0., - z] for z in depth])
        expected = np.asarray([vertex + offsets for vertex in outer[0]])
    assert np.allclose(locations, expected)


def test_depth_ball_sampling():
    img, *_ = data_gen.generate_mni_space_img()
    mesh = surface.load_surf_mesh(datasets.fetch_surf_fsaverage()["pial_left"])
    with pytest.raises(ValueError, match=".*does not support.*"):
        surface.vol_to_surf(img, mesh, kind="ball", depth=[.5])


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
    mesh = surface.load_surf_mesh(fsaverage["pial_left"])
    inner_mesh = surface.load_surf_mesh(fsaverage["white_left"])
    center_mesh = np.mean([mesh[0], inner_mesh[0]], axis=0), mesh[1]
    proj = surface.vol_to_surf(
        img, mesh, kind="depth", inner_mesh=inner_mesh, mask_img=mask_img)
    other_proj = surface.vol_to_surf(
        img, center_mesh, kind=kind, mask_img=mask_img)
    correlation = pearsonr(proj.ravel(), other_proj.ravel())[0]
    assert correlation > .99
    with pytest.raises(ValueError, match=".*interpolation.*"):
        surface.vol_to_surf(img, mesh, interpolation="bad")


def test_masked_indices():
    mask = np.ones((4, 3, 8))
    mask[:, :, ::2] = 0
    locations = np.mgrid[:5, :3, :8].ravel().reshape((3, -1))
    masked = surface._masked_indices(locations.T, mask.shape, mask)
    # These elements are masked by the mask
    assert (masked[::2] == 1).all()
    # The last element of locations is one row beyond first image dimension
    assert (masked[-24:] == 1).all()
    # 4 * 3 * 8 / 2 elements should remain unmasked
    assert (1 - masked).sum() == 48


def test_projection_matrix():
    mesh = flat_mesh(5, 7, 4)
    img = z_const_img(5, 7, 13)
    proj = surface._projection_matrix(
        mesh, np.eye(4), img.shape, radius=2., n_points=10)
    # proj matrix has shape (n_vertices, img_size)
    assert proj.shape == (5 * 7, 5 * 7 * 13)
    # proj.dot(img) should give the values of img at the vertices' locations
    values = proj.dot(img.ravel()).reshape((5, 7))
    assert_array_almost_equal(values, img[:, :, 0])
    mesh = flat_mesh(5, 7)
    proj = surface._projection_matrix(
        mesh, np.eye(4), (5, 7, 1), radius=.1, n_points=10)
    assert_array_almost_equal(proj.toarray(), np.eye(proj.shape[0]))
    mask = np.ones(img.shape, dtype=int)
    mask[0] = 0
    proj = surface._projection_matrix(
        mesh, np.eye(4), img.shape, radius=2., n_points=10, mask=mask)
    proj = proj.toarray()
    # first row of the mesh is masked
    assert_array_almost_equal(proj.sum(axis=1)[:7], np.zeros(7))
    assert_array_almost_equal(proj.sum(axis=1)[7:], np.ones(proj.shape[0] - 7))
    # mask and img should have the same shape
    pytest.raises(ValueError, surface._projection_matrix,
                  mesh, np.eye(4), img.shape, mask=np.ones((3, 3, 2)))


def test_sampling_affine():
    # check sampled (projected) values on a toy image
    img = np.ones((4, 4, 4))
    img[1, :, :] = 2
    nodes = [[1, 1, 2], [10, 10, 20], [30, 30, 30]]
    mesh = [np.asarray(nodes), None]
    affine = 10 * np.eye(4)
    affine[-1, -1] = 1
    texture = surface._nearest_voxel_sampling(
        [img], mesh, affine=affine, radius=1, kind='ball')
    assert_array_equal(texture[0], [1., 2., 1.])
    texture = surface._interpolation_sampling(
        [img], mesh, affine=affine, radius=0, kind='ball')
    assert_array_almost_equal(texture[0], [1.1, 2., 1.])


@pytest.mark.parametrize("kind", ["auto", "line", "ball"])
@pytest.mark.parametrize("use_inner_mesh", [True, False])
@pytest.mark.parametrize("projection", ["linear", "nearest"])
def test_sampling(kind, use_inner_mesh, projection):
    mesh = flat_mesh(5, 7, 4)
    img = z_const_img(5, 7, 13)
    mask = np.ones(img.shape, dtype=int)
    mask[0] = 0
    projector = {"nearest": surface._nearest_voxel_sampling,
                 "linear": surface._interpolation_sampling}[projection]
    inner_mesh = mesh if use_inner_mesh else None
    projection = projector([img], mesh, np.eye(4),
                           kind=kind, radius=0., inner_mesh=inner_mesh)
    assert_array_almost_equal(projection.ravel(), img[:, :, 0].ravel())
    projection = projector([img], mesh, np.eye(4),
                           kind=kind, radius=0., mask=mask,
                           inner_mesh=inner_mesh)
    assert_array_almost_equal(projection.ravel()[7:],
                              img[1:, :, 0].ravel())
    assert np.isnan(projection.ravel()[:7]).all()


@pytest.mark.parametrize("projection", ["linear", "nearest"])
def test_sampling_between_surfaces(projection):
    projector = {"nearest": surface._nearest_voxel_sampling,
                 "linear": surface._interpolation_sampling}[projection]
    mesh = flat_mesh(13, 7, 3.)
    inner_mesh = flat_mesh(13, 7, 1)
    img = z_const_img(5, 7, 13).T
    projection = projector(
        [img], mesh, np.eye(4),
        kind="auto", n_points=100, inner_mesh=inner_mesh)
    assert_array_almost_equal(
        projection.ravel(), img[:, :, 1:4].mean(axis=-1).ravel())


def test_choose_kind():
    kind = surface._choose_kind("abc", None)
    assert kind == "abc"
    kind = surface._choose_kind("abc", "mesh")
    assert kind == "abc"
    kind = surface._choose_kind("auto", None)
    assert kind == "line"
    kind = surface._choose_kind("auto", "mesh")
    assert kind == "depth"
    with pytest.raises(TypeError, match=".*sampling strategy"):
        kind = surface._choose_kind("depth", None)


def test_check_mesh():
    mesh = surface._check_mesh('fsaverage5')
    assert mesh is surface._check_mesh(mesh)
    with pytest.raises(ValueError):
        surface._check_mesh('fsaverage2')
    mesh.pop('pial_left')
    with pytest.raises(ValueError):
        surface._check_mesh(mesh)
    with pytest.raises(TypeError):
        surface._check_mesh(surface.load_surf_mesh(mesh['pial_right']))


def test_check_mesh_and_data():
    coords, faces = generate_surf()
    mesh = Mesh(coords, faces)
    data = mesh[0][:, 0]
    m, d = surface.check_mesh_and_data(mesh, data)
    assert (m[0] == mesh[0]).all()
    assert (m[1] == mesh[1]).all()
    assert (d == data).all()
    # Generate faces such that max index is larger than
    # the length of coordinates array.
    rng = np.random.RandomState(42)
    wrong_faces = rng.randint(coords.shape[0] + 1, size=(30, 3))
    wrong_mesh = Mesh(coords, wrong_faces)
    # Check that check_mesh_and_data raises an error with the resulting wrong mesh
    with pytest.raises(ValueError,
                       match="Mismatch between the indices of faces and the number of nodes."):
        surface.check_mesh_and_data(wrong_mesh, data)
    # Alter the data and check that an error is raised
    data = mesh[0][::2, 0]
    with pytest.raises(ValueError,
                       match="Mismatch between number of nodes in mesh"):
        surface.check_mesh_and_data(mesh, data)


def test_check_surface():
    coords, faces = generate_surf()
    mesh = Mesh(coords, faces)
    data = mesh[0][:,0]
    surf = Surface(mesh, data)
    s = surface.check_surface(surf)
    assert_array_equal(s.data, data)
    assert_array_equal(s.data, surf.data)
    assert_array_equal(s.mesh.coordinates, coords)
    assert_array_equal(s.mesh.coordinates, mesh.coordinates)
    assert_array_equal(s.mesh.faces, faces)
    assert_array_equal(s.mesh.faces, mesh.faces)
    # Generate faces such that max index is larger than
    # the length of coordinates array.
    rng = np.random.RandomState(42)
    wrong_faces = rng.randint(coords.shape[0] + 1, size=(30, 3))
    wrong_mesh = Mesh(coords, wrong_faces)
    wrong_surface = Surface(wrong_mesh, data)
    # Check that check_mesh_and_data raises an error with the resulting wrong mesh
    with pytest.raises(ValueError,
                       match="Mismatch between the indices of faces and the number of nodes."):
        surface.check_surface(wrong_surface)
    # Alter the data and check that an error is raised
    wrong_data = mesh[0][::2, 0]
    wrong_surface = Surface(mesh, wrong_data)
    with pytest.raises(ValueError,
                       match="Mismatch between number of nodes in mesh"):
        surface.check_surface(wrong_surface)
