"""
Test the niimg_conversions

This test file is in nilearn/tests because nosetests seems to ignore modules whose
name starts with an underscore
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD


import os
import tempfile

import nose
from nose.tools import assert_raises, assert_equal, assert_true

import numpy as np
from numpy.testing import assert_array_equal

from tempfile import mktemp

import nibabel
from nibabel import Nifti1Image

from nilearn import _utils
from nilearn._utils import NiftiGenerator, testing


class PhonyNiimage:

    def __init__(self):
        self.data = np.ones((9, 9, 9, 9))
        self.affine = np.ones((4, 4))

    def get_data(self):
        return self.data

    def get_affine(self):
        return self.affine


def test_check_niimg():
    with assert_raises(TypeError) as cm:
        _utils.check_niimg(0)
    assert_true('image' in cm.exception.message
                or 'affine' in cm.exception.message)

    with assert_raises(TypeError) as cm:
        _utils.check_niimg([])
    assert_true('image' in cm.exception.message
                or 'affine' in cm.exception.message)

    # Test ensure_3d
    with assert_raises(TypeError) as cm:
        _utils.check_niimg(['test.nii', ], ensure_3d=True)
    assert_true('3D' in cm.exception.message)

    # Check that a filename does not raise an error
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True) as filename:
        _utils.check_niimg(filename)

    # Test ensure_3d with a in-memory object
    with assert_raises(TypeError) as cm:
        _utils.check_niimg(data, ensure_3d=True)
    assert_true('3D' in cm.exception.message)

    # Test ensure_3d with a non 3D image
    with assert_raises(TypeError) as cm:
        _utils.check_niimg(data_img, ensure_3d=True)
    assert_true('3D' in cm.exception.message)

    # Test ensure_3d with a 4D image with a length 1 4th dim
    data = np.zeros((40, 40, 40, 1))
    data_img = Nifti1Image(data, np.eye(4))
    _utils.check_niimg(data_img, ensure_3d=True)


def test_check_niimgs():
    with assert_raises(TypeError) as cm:
        _utils.check_niimgs(0)
    assert_true('image' in cm.exception.message
                or 'affine' in cm.exception.message)

    with assert_raises(TypeError) as cm:
        _utils.check_niimgs([])
    assert_true('image' in cm.exception.message
                or 'affine' in cm.exception.message)

    affine = np.eye(4)
    niimg = Nifti1Image(np.ones((10, 10, 10)), affine)

    _utils.check_niimgs([niimg, niimg])
    with assert_raises(TypeError) as cm:
        # This should raise an error: a 3D niimg is given and we want a 4D
        _utils.check_niimgs(niimg)
    assert_true('image' in cm.exception.message)
    # This shouldn't raise an error
    _utils.check_niimgs(niimg, accept_3d=True)

    # Test a Niimage that does not hold a shape attribute
    phony_niimg = PhonyNiimage()
    _utils.check_niimgs(phony_niimg)


def test_repr_niimgs():
    # Test with file path
    assert_equal(_utils._repr_niimgs("test"), "test")
    assert_equal(_utils._repr_niimgs(["test", "retest"]), "[test, retest]")
    # Create phony Niimg with filename
    affine = np.eye(4)
    shape = (10, 10, 10)
    niimg1 = Nifti1Image(np.ones(shape), affine)
    assert_equal(
            _utils._repr_niimgs(niimg1),
            ("%s(\nshape=%s,\naffine=%s\n)" % (niimg1.__class__.__name__,
                            repr(shape), repr(affine))))
    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    nibabel.save(niimg1, tmpimg1)
    assert_equal(
            _utils._repr_niimgs(niimg1),
            ("%s('%s')" % (niimg1.__class__.__name__, niimg1.get_filename())))


def _remove_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


def test_concat_niimgs():
    shape = (10, 11, 12)
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones(shape), affine)
    niimg2 = Nifti1Image(np.ones(shape), 2 * affine)
    niimg3 = Nifti1Image(np.zeros(shape), affine)
    niimg4d = Nifti1Image(np.ones(shape + (2, )), affine)

    concatenated = _utils.concat_niimgs((niimg1, niimg3, niimg1))
    concatenate_true = np.ones(shape + (3,))
    concatenate_true[..., 1] = 0
    np.testing.assert_almost_equal(concatenated.get_data(), concatenate_true)

    assert_raises(ValueError, _utils.concat_niimgs, [niimg1, niimg2])

    # Smoke-test the accept_4d
    assert_raises(ValueError, _utils.concat_niimgs, [niimg1, niimg4d])
    concatenated = _utils.concat_niimgs([niimg1, niimg4d], accept_4d=True)
    assert_equal(concatenated.shape[3], 3)

    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    _, tmpimg2 = tempfile.mkstemp(suffix='.nii')
    try:
        nibabel.save(niimg1, tmpimg1)
        nibabel.save(niimg2, tmpimg2)
        nose.tools.assert_raises(ValueError, _utils.concat_niimgs,
                                 [tmpimg1, tmpimg2])
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)

def test_NiftiGenerator():
    # create 3 random nifti images in dedicated temporary folder of the OS
    tmp_files = []
    n_test_niimgs = 3
    for ifile in range(n_test_niimgs):
        cur_tmp_file_path = mktemp()
        cur_tmp_file_path += '.nii'
        nibabel.Nifti1Image(
            np.random.randn(40, 50, 60),
            np.eye(4)).to_filename(cur_tmp_file_path)
        tmp_files.append(cur_tmp_file_path)

    # just retrieve 3 flavors of random nifti sets with same shape and affine
    paths_list = tmp_files
    img3d_list = [nibabel.load(path) for path in paths_list]
    data4d = np.concatenate(
        [img3d_list[i].get_data()[..., np.newaxis] 
        for i in range(n_test_niimgs)], axis=3)
    img4d = nibabel.Nifti1Image(data4d, img3d_list[0].get_affine())

    # iterate through them in 3 ways comparing between them at each iteration
    for (img1, data1, affine1), (img2, data2, affine2), \
        (img3, data3, affine3) in zip(NiftiGenerator(paths_list),
            NiftiGenerator(img3d_list), NiftiGenerator(img4d)):
        # test numpy arrays
        assert_array_equal(data1, data2)
        assert_array_equal(data1, data3)
        assert_array_equal(affine1, affine2)
        assert_array_equal(affine1, affine3)

        # idem, but retrieve pointers from Nift1Image object
        assert_true(isinstance(img1, nibabel.Nifti1Image))
        assert_true(isinstance(img2, nibabel.Nifti1Image))
        assert_true(isinstance(img3, nibabel.Nifti1Image))
        assert_array_equal(img1.get_data(), img2.get_data())
        assert_array_equal(img1.get_data(), img3.get_data())
        assert_array_equal(img1.get_affine(), img2.get_affine())
        assert_array_equal(img1.get_affine(), img3.get_affine())

    # clean-up
    for i in range(n_test_niimgs):
        os.remove(tmp_files[i])
