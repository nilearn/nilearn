"""
Test the niimg_conversions

This test file is in nilearn/tests because nosetests seems to ignore modules whose
name starts with an underscore
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os
import tempfile
import itertools

from nose.tools import assert_equal, assert_true

import numpy as np
from numpy.testing import assert_array_equal

import nibabel
from nibabel import Nifti1Image

from nilearn import _utils
from nilearn._utils import testing
from nilearn._utils.testing import assert_raises_regexp


class PhonyNiimage(object):

    def __init__(self):
        self.data = np.ones((9, 9, 9, 9))
        self.affine = np.ones((4, 4))

    def get_data(self):
        return self.data

    def get_affine(self):
        return self.affine


def test_check_niimg():
    # check error for non-forced but necessary resampling
    assert_raises_regexp(TypeError, 'image',
                         _utils.check_niimg, 0)

    # check error for non-forced but necessary resampling
    assert_raises_regexp(TypeError, 'image',
                         _utils.check_niimg, [])

    # Test ensure_3d
    # check error for non-forced but necessary resampling
    assert_raises_regexp(TypeError, '3D',
                         _utils.check_niimg, ['test.nii', ], ensure_3d=True)

    # Check that a filename does not raise an error
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True) as filename:
        _utils.check_niimg(filename)

    # Test ensure_3d with a in-memory object
    assert_raises_regexp(TypeError, '3D',
                         _utils.check_niimg, data, ensure_3d=True)

    # Test ensure_3d with a non 3D image
    assert_raises_regexp(TypeError, '3D',
                         _utils.check_niimg, data_img, ensure_3d=True)

    # Test ensure_3d with a 4D image with a length 1 4th dim
    data = np.zeros((40, 40, 40, 1))
    data_img = Nifti1Image(data, np.eye(4))
    _utils.check_niimg(data_img, ensure_3d=True)


def test_check_niimgs():
    assert_raises_regexp(TypeError, 'image',
                         _utils.check_niimgs, 0)

    assert_raises_regexp(TypeError, 'image',
                         _utils.check_niimgs, [])

    affine = np.eye(4)
    img_3d = Nifti1Image(np.ones((10, 10, 10)), affine)

    # Tests with return_iterator=False
    img_4d_1 = _utils.check_niimgs([img_3d, img_3d])
    assert_true(img_4d_1.get_data().shape == (10, 10, 10, 2))
    assert_array_equal(img_4d_1.get_affine(), affine)

    img_4d_2 = _utils.check_niimgs(img_4d_1)
    assert_array_equal(img_4d_2.get_data(), img_4d_2.get_data())
    assert_array_equal(img_4d_2.get_affine(), img_4d_2.get_affine())

    # Tests with return_iterator=True
    img_3d_iterator = _utils.check_niimgs([img_3d, img_3d],
                                          return_iterator=True)
    img_3d_iterator_length = sum(1 for _ in img_3d_iterator)
    assert_true(img_3d_iterator_length == 2)

    img_3d_iterator_1 = _utils.check_niimgs([img_3d, img_3d],
                                            return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimgs(img_3d_iterator_1,
                                            return_iterator=True)
    for img_1, img_2 in itertools.izip(img_3d_iterator_1, img_3d_iterator_2):
        assert_true(img_1.get_data().shape == (10, 10, 10))
        assert_array_equal(img_1.get_data(), img_2.get_data())
        assert_array_equal(img_1.get_affine(), img_2.get_affine())

    img_3d_iterator_1 = _utils.check_niimgs([img_3d, img_3d],
                                            return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimgs(img_4d_1,
                                            return_iterator=True)
    for img_1, img_2 in itertools.izip(img_3d_iterator_1, img_3d_iterator_2):
        assert_true(img_1.get_data().shape == (10, 10, 10))
        assert_array_equal(img_1.get_data(), img_2.get_data())
        assert_array_equal(img_1.get_affine(), img_2.get_affine())

    # This should raise an error: a 3D img is given and we want a 4D
    assert_raises_regexp(TypeError, 'image',
                         _utils.check_niimgs, img_3d)
    # This shouldn't raise an error
    _utils.check_niimgs(img_3d, accept_3d=True)

    # Test a Niimg-like object that does not hold a shape attribute
    phony_img = PhonyNiimage()
    _utils.check_niimgs(phony_img)


def test_repr_niimgs():
    # Test with file path
    assert_equal(_utils._repr_niimgs("test"), "test")
    assert_equal(_utils._repr_niimgs(["test", "retest"]), "[test, retest]")
    # Create phony Niimg with filename
    affine = np.eye(4)
    shape = (10, 10, 10)
    img1 = Nifti1Image(np.ones(shape), affine)
    assert_equal(
        _utils._repr_niimgs(img1),
        ("%s(\nshape=%s,\naffine=%s\n)" %
            (img1.__class__.__name__,
             repr(shape), repr(affine))))
    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    nibabel.save(img1, tmpimg1)
    assert_equal(
        _utils._repr_niimgs(img1),
        ("%s('%s')" % (img1.__class__.__name__, img1.get_filename())))


def _remove_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


def test_concat_niimgs():
    # create images different in affine and 3D/4D shape
    shape = (10, 11, 12)
    affine = np.eye(4)
    img1 = Nifti1Image(np.ones(shape), affine)
    img2 = Nifti1Image(np.ones(shape), 2 * affine)
    img3 = Nifti1Image(np.zeros(shape), affine)
    img4d = Nifti1Image(np.ones(shape + (2, )), affine)

    shape2 = (12, 11, 10)
    img1b = Nifti1Image(np.ones(shape2), affine)

    shape3 = (11, 22, 33)
    img1c = Nifti1Image(np.ones(shape2), affine)

    # check basic concatenation with equal shape/affine
    concatenated = _utils.concat_niimgs((img1, img3, img1),
                                        accept_4d=False)
    concatenate_true = np.ones(shape + (3,))

    # Smoke-test the accept_4d
    assert_raises_regexp(ValueError, 'image',
                         _utils.concat_niimgs, [img1, img4d])
    concatenated = _utils.concat_niimgs([img1, img4d], accept_4d=True)
    np.testing.assert_equal(concatenated.get_data(), concatenate_true,
                            verbose=0)

    # smoke-test auto_resample
    concatenated = _utils.concat_niimgs((img1, img1b, img1c), accept_4d=False,
        auto_resample=True)
    assert_true(concatenated.shape == img1.shape + (3, ))

    # check error for non-forced but necessary resampling
    assert_raises_regexp(ValueError, 'different from reference affine',
                         _utils.concat_niimgs, [img1, img2],
                         accept_4d=False)

    # Smoke-test the 4d parsing
    concatenated = _utils.concat_niimgs([img1, img4d], accept_4d=True)
    assert_equal(concatenated.shape[3], 3)

    # test list of 4D niimgs as input
    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    _, tmpimg2 = tempfile.mkstemp(suffix='.nii')
    try:
        nibabel.save(img1, tmpimg1)
        nibabel.save(img3, tmpimg2)
        concatenated = _utils.concat_niimgs([tmpimg1, tmpimg2],
                                            accept_4d=False)
        assert_array_equal(
            concatenated.get_data()[..., 0], img1.get_data())
        assert_array_equal(
            concatenated.get_data()[..., 1], img3.get_data())
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)
