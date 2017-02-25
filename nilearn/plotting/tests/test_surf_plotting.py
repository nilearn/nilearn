# Tests for functions in surf_plotting.py

import os
import tempfile

from distutils.version import LooseVersion
from nose import SkipTest

import matplotlib
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

from nibabel import gifti
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal)

from nilearn._utils.testing import assert_raises_regex
from nilearn.plotting.surf_plotting import (load_surf_data, load_surf_mesh,
                                            plot_surf, plot_surf_stat_map,
                                            plot_surf_roi)

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


def test_plot_surf():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest

    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )

    # Plot mesh only
    plot_surf(mesh)

    # Plot mesh with background
    plot_surf(mesh, bg_map=bg)
    plot_surf(mesh, bg_map=bg, darkness=0.5)
    plot_surf(mesh, bg_map=bg, alpha=0.5)

    # Plot different views
    plot_surf(mesh, bg_map=bg, hemi='right')
    plot_surf(mesh, bg_map=bg, view='medial')
    plot_surf(mesh, bg_map=bg, hemi='right', view='medial')

    # Save execution time and memory
    plt.close()


def test_plot_surf_error():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest
    mesh = _generate_surf()
    rng = np.random.RandomState(0)

    # Wrong inputs for view or hemi
    assert_raises_regex(ValueError, 'view must be one of',
                        plot_surf, mesh, view='middle')
    assert_raises_regex(ValueError, 'hemi must be one of',
                        plot_surf, mesh, hemi='lft')

    # Wrong size of background image
    assert_raises_regex(ValueError,
                        'bg_map does not have the same number of vertices',
                        plot_surf, mesh,
                        bg_map=rng.randn(mesh[0].shape[0] - 1, ))

    # Wrong size of surface data
    assert_raises_regex(ValueError,
                        'surf_map does not have the same number of vertices',
                        plot_surf, mesh,
                        surf_map=rng.randn(mesh[0].shape[0] + 1, ))

    assert_raises_regex(ValueError,
                        'surf_map can only have one dimension', plot_surf,
                        mesh, surf_map=rng.randn(mesh[0].shape[0], 2))


def test_plot_surf_stat_map():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest

    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Plot mesh with stat map
    plot_surf_stat_map(mesh, stat_map=data)
    plot_surf_stat_map(mesh, stat_map=data, alpha=1)

    # Plot mesh with background and stat map
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_stat=True, darkness=0.5)

    # Apply threshold
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_stat=True, darkness=0.5,
                       threshold=0.3)

    # Change vmax
    plot_surf_stat_map(mesh, stat_map=data, vmax=5)

    # Change colormap
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix')

    # Save execution time and memory
    plt.close()


def test_plot_surf_stat_map_error():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Try to input vmin
    assert_raises_regex(ValueError,
                        'this function does not accept a "vmin" argument',
                        plot_surf_stat_map, mesh, stat_map=data, vmin=0)

    # Wrong size of stat map data
    assert_raises_regex(ValueError,
                        'surf_map does not have the same number of vertices',
                        plot_surf_stat_map, mesh,
                        stat_map=np.hstack((data, data)))

    assert_raises_regex(ValueError,
                        'surf_map can only have one dimension',
                        plot_surf_stat_map, mesh,
                        stat_map=np.vstack((data, data)).T)


def test_plot_surf_roi():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    roi1 = rng.randint(0, mesh[0].shape[0], size=5)
    roi2 = rng.randint(0, mesh[0].shape[0], size=10)
    parcellation = rng.rand(mesh[0].shape[0])

    # plot roi
    plot_surf_roi(mesh, roi_map=roi1)

    # plot parcellation
    plot_surf_roi(mesh, roi_map=parcellation)

    # plot roi list
    plot_surf_roi(mesh, roi_map=[roi1, roi2])

    # Save execution time and memory
    plt.close()


def test_plot_surf_roi_error():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    roi1 = rng.randint(0, mesh[0].shape[0], size=5)
    roi2 = rng.randint(0, mesh[0].shape[0], size=10)

    # Wrong input
    assert_raises_regex(ValueError,
                        'Invalid input for roi_map',
                        plot_surf_roi, mesh,
                        roi_map={'roi1': roi1, 'roi2': roi2})
