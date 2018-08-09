# Tests for functions in surf_plotting.py

import os
import tempfile
import warnings
import itertools

from distutils.version import LooseVersion
from nose import SkipTest
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal)
from nose.tools import assert_true, assert_raises
from nilearn._utils.testing import assert_raises_regex, assert_warns

import numpy as np
from scipy.spatial import Delaunay
import sklearn

import nibabel as nb
from nibabel import gifti

from nilearn import datasets
from nilearn import image
from nilearn.image import resampling
from nilearn.image.tests.test_resampling import rotation
from nilearn.surface import surface
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf
from nilearn.surface.surface import (_gifti_img_to_mesh,
                                     _load_surf_files_gifti_gzip)

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def _generate_surf():
    rng = np.random.RandomState(42)
    coords = rng.rand(20, 3)
    faces = rng.randint(coords.shape[0], size=(30, 3))
    return [coords, faces]


def test_load_surf_data_array():
    # test loading and squeezing data from numpy array
    data_flat = np.zeros((20, ))
    data_squeeze = np.zeros((20, 1, 3))
    assert_array_equal(load_surf_data(data_flat), np.zeros((20, )))
    assert_array_equal(load_surf_data(data_squeeze), np.zeros((20, 3)))


def test_load_surf_data_file_nii_gii():
    # test loading of fake data from gifti file
    filename_gii = tempfile.mktemp(suffix='.gii')
    if LooseVersion(nb.__version__) > LooseVersion('2.0.2'):
        darray = gifti.GiftiDataArray(data=np.zeros((20, )))
    else:
        # Avoid a bug in nibabel 1.2.0 where GiftiDataArray were not
        # initialized properly:
        darray = gifti.GiftiDataArray.from_array(np.zeros((20, )),
                                                 intent='t test')
    gii = gifti.GiftiImage(darrays=[darray])
    gifti.write(gii, filename_gii)
    assert_array_equal(load_surf_data(filename_gii), np.zeros((20, )))
    os.remove(filename_gii)

    # test loading of data from empty gifti file
    filename_gii_empty = tempfile.mktemp(suffix='.gii')
    gii_empty = gifti.GiftiImage()
    gifti.write(gii_empty, filename_gii_empty)
    assert_raises_regex(ValueError,
                        'must contain at least one data array',
                        load_surf_data, filename_gii_empty)
    os.remove(filename_gii_empty)

    # test loading of fake data from nifti file
    filename_nii = tempfile.mktemp(suffix='.nii')
    filename_niigz = tempfile.mktemp(suffix='.nii.gz')
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
    assert_true(isinstance(gii, gifti.GiftiImage))

    data = load_surf_data(fsaverage)
    assert_true(isinstance(data, np.ndarray))

    # surface mesh
    fsaverage = datasets.fetch_surf_fsaverage().pial_left
    gii = _load_surf_files_gifti_gzip(fsaverage)
    assert_true(isinstance(gii, gifti.GiftiImage))


def test_load_surf_data_file_freesurfer():
    # test loading of fake data from sulc and thickness files
    # using load_surf_data.
    # We test load_surf_data by creating fake data with function
    # 'write_morph_data' that works only if nibabel
    # version is recent with nibabel >= 2.1.0
    if LooseVersion(nb.__version__) >= LooseVersion('2.1.0'):
        data = np.zeros((20, ))
        filename_sulc = tempfile.mktemp(suffix='.sulc')
        nb.freesurfer.io.write_morph_data(filename_sulc, data)
        assert_array_equal(load_surf_data(filename_sulc), np.zeros((20, )))
        os.remove(filename_sulc)

        filename_thick = tempfile.mktemp(suffix='.thickness')
        nb.freesurfer.io.write_morph_data(filename_thick, data)
        assert_array_equal(load_surf_data(filename_thick), np.zeros((20, )))
        os.remove(filename_thick)

    # test loading of data from real label and annot files
    label_start = np.array([5900, 5899, 5901, 5902, 2638])
    label_end = np.array([8756, 6241, 8757, 1896, 6243])
    label = load_surf_data(os.path.join(datadir, 'test.label'))
    assert_array_equal(label[:5], label_start)
    assert_array_equal(label[-5:], label_end)
    assert_equal(label.shape, (10, ))
    del label, label_start, label_end

    annot_start = np.array([24, 29, 28, 27, 24, 31, 11, 25, 0, 12])
    annot_end = np.array([16, 16, 16, 16, 16, 16, 16, 16, 16, 16])
    annot = load_surf_data(os.path.join(datadir, 'test.annot'))
    assert_array_equal(annot[:10], annot_start)
    assert_array_equal(annot[-10:], annot_end)
    assert_equal(annot.shape, (10242, ))
    del annot, annot_start, annot_end


def test_load_surf_data_file_error():
    # test if files with unexpected suffixes raise errors
    data = np.zeros((20, ))
    wrong_suff = ['.vtk', '.obj', '.mnc', '.txt']
    for suff in wrong_suff:
        filename_wrong = tempfile.mktemp(suffix=suff)
        np.savetxt(filename_wrong, data)
        assert_raises_regex(ValueError,
                            'input type is not recognized',
                            load_surf_data, filename_wrong)
        os.remove(filename_wrong)


def test_load_surf_mesh_list():
    # test if correct list is returned
    mesh = _generate_surf()
    assert_equal(len(load_surf_mesh(mesh)), 2)
    assert_array_equal(load_surf_mesh(mesh)[0], mesh[0])
    assert_array_equal(load_surf_mesh(mesh)[1], mesh[1])
    # test if incorrect list, array or dict raises error
    assert_raises_regex(ValueError, 'it must have two elements',
                        load_surf_mesh, [])
    assert_raises_regex(ValueError, 'it must have two elements',
                        load_surf_mesh, [mesh[0]])
    assert_raises_regex(ValueError, 'it must have two elements',
                        load_surf_mesh, [mesh[0], mesh[1], mesh[1]])
    assert_raises_regex(ValueError, 'input type is not recognized',
                        load_surf_mesh, mesh[0])
    assert_raises_regex(ValueError, 'input type is not recognized',
                        load_surf_mesh, dict())
    del mesh


def test_gifti_img_to_mesh():
    mesh = _generate_surf()

    coord_array = gifti.GiftiDataArray(data=mesh[0])
    coord_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET']

    face_array = gifti.GiftiDataArray(data=mesh[1])
    face_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']

    gii = gifti.GiftiImage(darrays=[coord_array, face_array])
    coords, faces = _gifti_img_to_mesh(gii)
    assert_array_equal(coords, mesh[0])
    assert_array_equal(faces, mesh[1])


def test_load_surf_mesh_file_gii_gz():
    # Test the loader `load_surf_mesh` with gzipped fsaverage5 files

    fsaverage = datasets.fetch_surf_fsaverage().pial_left
    coords, faces = load_surf_mesh(fsaverage)
    assert_true(isinstance(coords, np.ndarray))
    assert_true(isinstance(faces, np.ndarray))


def test_load_surf_mesh_file_gii():
    # Test the loader `load_surf_mesh`

    # If nibabel is of older version we skip tests as nibabel does not
    # support intent argument and intent codes are not handled properly with
    # older versions

    if not LooseVersion(nb.__version__) >= LooseVersion('2.1.0'):
        raise SkipTest

    mesh = _generate_surf()

    # test if correct gii is loaded into correct list
    filename_gii_mesh = tempfile.mktemp(suffix='.gii')

    coord_array = gifti.GiftiDataArray(data=mesh[0],
                                       intent=nb.nifti1.intent_codes[
                                           'NIFTI_INTENT_POINTSET'])
    face_array = gifti.GiftiDataArray(data=mesh[1],
                                      intent=nb.nifti1.intent_codes[
                                          'NIFTI_INTENT_TRIANGLE'])

    gii = gifti.GiftiImage(darrays=[coord_array, face_array])
    gifti.write(gii, filename_gii_mesh)
    assert_array_equal(load_surf_mesh(filename_gii_mesh)[0], mesh[0])
    assert_array_equal(load_surf_mesh(filename_gii_mesh)[1], mesh[1])
    os.remove(filename_gii_mesh)

    # test if incorrect gii raises error
    filename_gii_mesh_no_point = tempfile.mktemp(suffix='.gii')
    gifti.write(gifti.GiftiImage(darrays=[face_array, face_array]),
                filename_gii_mesh_no_point)
    assert_raises_regex(ValueError, 'NIFTI_INTENT_POINTSET',
                        load_surf_mesh, filename_gii_mesh_no_point)
    os.remove(filename_gii_mesh_no_point)

    filename_gii_mesh_no_face = tempfile.mktemp(suffix='.gii')
    gifti.write(gifti.GiftiImage(darrays=[coord_array, coord_array]),
                filename_gii_mesh_no_face)
    assert_raises_regex(ValueError, 'NIFTI_INTENT_TRIANGLE',
                        load_surf_mesh, filename_gii_mesh_no_face)
    os.remove(filename_gii_mesh_no_face)


def test_load_surf_mesh_file_freesurfer():
    # Older nibabel versions does not support 'write_geometry'
    if LooseVersion(nb.__version__) <= LooseVersion('1.2.0'):
        raise SkipTest

    mesh = _generate_surf()
    for suff in ['.pial', '.inflated', '.white', '.orig', 'sphere']:
        filename_fs_mesh = tempfile.mktemp(suffix=suff)
        nb.freesurfer.write_geometry(filename_fs_mesh, mesh[0], mesh[1])
        assert_equal(len(load_surf_mesh(filename_fs_mesh)), 2)
        assert_array_almost_equal(load_surf_mesh(filename_fs_mesh)[0],
                                  mesh[0])
        assert_array_almost_equal(load_surf_mesh(filename_fs_mesh)[1],
                                  mesh[1])
        os.remove(filename_fs_mesh)


def test_load_surf_mesh_file_error():
    if LooseVersion(nb.__version__) <= LooseVersion('1.2.0'):
        raise SkipTest

    # test if files with unexpected suffixes raise errors
    mesh = _generate_surf()
    wrong_suff = ['.vtk', '.obj', '.mnc', '.txt']
    for suff in wrong_suff:
        filename_wrong = tempfile.mktemp(suffix=suff)
        nb.freesurfer.write_geometry(filename_wrong, mesh[0], mesh[1])
        assert_raises_regex(ValueError,
                            'input type is not recognized',
                            load_surf_data, filename_wrong)
        os.remove(filename_wrong)


def _flat_mesh(x_s, y_s, z=0):
    x, y = np.mgrid[:x_s, :y_s]
    x, y = x.ravel(), y.ravel()
    z = np.ones(len(x)) * z
    vertices = np.asarray([x, y, z]).T
    triangulation = Delaunay(vertices[:, :2]).simplices
    mesh = [vertices, triangulation]
    return mesh


def _z_const_img(x_s, y_s, z_s):
    hslice = np.arange(x_s * y_s).reshape((x_s, y_s))
    return np.ones((x_s, y_s, z_s)) * hslice[:, :, np.newaxis]


def test_vertex_outer_normals():
    # compute normals for a flat horizontal mesh, they should all be (0, 0, 1)
    mesh = _flat_mesh(5, 7)
    computed_normals = surface._vertex_outer_normals(mesh)
    true_normals = np.zeros((len(mesh[0]), 3))
    true_normals[:, 2] = 1
    assert_array_almost_equal(computed_normals, true_normals)


def test_load_uniform_ball_cloud():
    for n_points in [10, 20, 40, 80, 160]:
        with warnings.catch_warnings(record=True) as w:
            points = surface._load_uniform_ball_cloud(n_points=n_points)
            assert_array_equal(points.shape, (n_points, 3))
            assert_equal(len(w), 0)
    assert_warns(surface.EfficiencyWarning,
                 surface._load_uniform_ball_cloud, n_points=3)
    for n_points in [3, 10, 20]:
        computed = surface._uniform_ball_cloud(n_points)
        loaded = surface._load_uniform_ball_cloud(n_points)
        assert_array_almost_equal(computed, loaded)


def test_sample_locations():
    # check positions of samples on toy example, with an affine != identity
    # flat horizontal mesh
    mesh = _flat_mesh(5, 7)
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
    line_offsets[:, 2] = np.linspace(-1, 1, 10)
    line_offsets = np.asarray(
        resampling.coord_transform(*line_offsets.T, affine=inv_affine)).T
    # check we get the same locations
    for kind, offsets in [('line', line_offsets), ('ball', ball_offsets)]:
        locations = surface._sample_locations(
            [vertices, mesh[1]], affine, 1., kind=kind, n_points=10)
        true_locations = np.asarray([vertex + offsets for vertex in mesh[0]])
        assert_array_equal(locations.shape, true_locations.shape)
        assert_array_almost_equal(true_locations, locations)
    assert_raises(ValueError, surface._sample_locations,
                  mesh, affine, 1., kind='bad_kind')


def test_masked_indices():
    mask = np.ones((4, 3, 8))
    mask[:, :, ::2] = 0
    locations = np.mgrid[:5, :3, :8].ravel().reshape((3, -1))
    masked = surface._masked_indices(locations.T, mask.shape, mask)
    # These elements are masked by the mask
    assert_true((masked[::2] == 1).all())
    # The last element of locations is one row beyond first image dimension
    assert_true((masked[-24:] == 1).all())
    # 4 * 3 * 8 / 2 elements should remain unmasked
    assert_true((1 - masked).sum() == 48)


def test_projection_matrix():
    mesh = _flat_mesh(5, 7, 4)
    img = _z_const_img(5, 7, 13)
    proj = surface._projection_matrix(
        mesh, np.eye(4), img.shape, radius=2., n_points=10)
    # proj matrix has shape (n_vertices, img_size)
    assert_equal(proj.shape, (5 * 7, 5 * 7 * 13))
    # proj.dot(img) should give the values of img at the vertices' locations
    values = proj.dot(img.ravel()).reshape((5, 7))
    assert_array_almost_equal(values, img[:, :, 0])
    mesh = _flat_mesh(5, 7)
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
    assert_raises(ValueError, surface._projection_matrix,
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


def test_sampling():
    mesh = _flat_mesh(5, 7, 4)
    img = _z_const_img(5, 7, 13)
    mask = np.ones(img.shape, dtype=int)
    mask[0] = 0
    projectors = [surface._nearest_voxel_sampling,
                  surface._interpolation_sampling]
    for kind in ('line', 'ball'):
        for projector in projectors:
            projection = projector([img], mesh, np.eye(4),
                                   kind=kind, radius=0.)
            assert_array_almost_equal(projection.ravel(), img[:, :, 0].ravel())
            projection = projector([img], mesh, np.eye(4),
                                   kind=kind, radius=0., mask=mask)
            assert_array_almost_equal(projection.ravel()[7:],
                                      img[1:, :, 0].ravel())
            assert_true(np.isnan(projection.ravel()[:7]).all())


def test_vol_to_surf():
    # test 3d niimg to cortical surface projection and invariance to a change
    # of affine
    mni = datasets.load_mni152_template()
    mesh = _generate_surf()
    _check_vol_to_surf_results(mni, mesh)
    fsaverage = datasets.fetch_surf_fsaverage5().pial_left
    _check_vol_to_surf_results(mni, fsaverage)


def _check_vol_to_surf_results(img, mesh):
    mni_mask = datasets.load_mni152_brain_mask()
    for kind, interpolation, mask_img in itertools.product(
            ['ball', 'line'], ['linear', 'nearest'], [mni_mask, None]):
        proj_1 = vol_to_surf(
            img, mesh, kind=kind, interpolation=interpolation,
            mask_img=mask_img)
        assert_true(proj_1.ndim == 1)
        img_rot = image.resample_img(
            img, target_affine=rotation(np.pi / 3., np.pi / 4.))
        proj_2 = vol_to_surf(
            img_rot, mesh, kind=kind, interpolation=interpolation,
            mask_img=mask_img)
        # The projection values for the rotated image should be close
        # to the projection for the original image
        diff = np.abs(proj_1 - proj_2) / np.abs(proj_1)
        assert_true(np.mean(diff[diff < np.inf]) < .03)
        img_4d = image.concat_imgs([img, img])
        proj_4d = vol_to_surf(
            img_4d, mesh, kind=kind, interpolation=interpolation,
            mask_img=mask_img)
        nodes, _ = surface.load_surf_mesh(mesh)
        assert_array_equal(proj_4d.shape, [nodes.shape[0], 2])
        assert_array_almost_equal(proj_4d[:, 0], proj_1, 3)


def test_check_mesh_and_data():
    mesh = _generate_surf()
    data = mesh[0][:, 0]
    m, d = surface.check_mesh_and_data(mesh, data)
    assert (m[0] == mesh[0]).all()
    assert (m[1] == mesh[1]).all()
    assert (d == data).all()
    data = mesh[0][::2, 0]
    assert_raises(ValueError, surface.check_mesh_and_data, mesh, data)
