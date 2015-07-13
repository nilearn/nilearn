"""
Test the niimg_conversions

This test file is in nilearn/tests because nosetests seems to ignore modules
whose name starts with an underscore
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os
import tempfile

from nose.tools import assert_equal, assert_true

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel
from nibabel import Nifti1Image

from nilearn import _utils, image
from nilearn._utils.exceptions import DimensionError
from nilearn._utils import testing, niimg_conversions
from nilearn._utils.testing import assert_raises_regex


class PhonyNiimage(nibabel.spatialimages.SpatialImage):

    def __init__(self):
        self.data = np.ones((9, 9, 9, 9))
        self.my_affine = np.ones((4, 4))

    def get_data(self):
        return self.data

    def get_affine(self):
        return self.my_affine

    @property
    def shape(self):
        return self.data.shape


def test_check_same_fov():

    affine_a = np.eye(4)
    affine_b = np.eye(4) * 2

    shape_a = (2, 2, 2)
    shape_b = (3, 3, 3)

    shape_a_affine_a = nibabel.Nifti1Image(np.empty(shape_a), affine_a)
    shape_a_affine_a_2 = nibabel.Nifti1Image(np.empty(shape_a), affine_a)
    shape_a_affine_b = nibabel.Nifti1Image(np.empty(shape_a), affine_b)
    shape_b_affine_a = nibabel.Nifti1Image(np.empty(shape_b), affine_a)
    shape_b_affine_b = nibabel.Nifti1Image(np.empty(shape_b), affine_b)

    niimg_conversions._check_same_fov(a=shape_a_affine_a,
                                      b=shape_a_affine_a_2,
                                      raise_error=True)

    assert_raises_regex(ValueError,
                        '[ac] and [ac] do not have the same affine',
                        niimg_conversions._check_same_fov,
                        a=shape_a_affine_a, b=shape_a_affine_a_2,
                        c=shape_a_affine_b, raise_error=True)

    assert_raises_regex(ValueError,
                        '[ab] and [ab] do not have the same shape',
                        niimg_conversions._check_same_fov,
                        a=shape_a_affine_a, b=shape_b_affine_a,
                        raise_error=True)

    assert_raises_regex(ValueError,
                        '[ab] and [ab] do not have the same affine',
                        niimg_conversions._check_same_fov,
                        a=shape_b_affine_b, b=shape_a_affine_a,
                        raise_error=True)

    assert_raises_regex(ValueError,
                        '[ab] and [ab] do not have the same shape',
                        niimg_conversions._check_same_fov,
                        a=shape_b_affine_b, b=shape_a_affine_a,
                        raise_error=True)


def test_check_niimg_3d():
    # check error for non-forced but necessary resampling
    assert_raises_regex(TypeError, 'nibabel format',
                        _utils.check_niimg, 0)

    # check error for non-forced but necessary resampling
    assert_raises_regex(TypeError, 'empty object',
                        _utils.check_niimg, [])

    # Test dimensionality error
    img = Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    assert_raises_regex(TypeError, 'Data must be a 3D',
                        _utils.check_niimg_3d, [img, img])

    # Check that a filename does not raise an error
    data = np.zeros((40, 40, 40, 1))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True) as filename:
        _utils.check_niimg_3d(filename)


def test_check_niimg_4d():
    assert_raises_regex(TypeError, 'nibabel format',
                        _utils.check_niimg_4d, 0)

    assert_raises_regex(TypeError, 'empty object',
                        _utils.check_niimg_4d, [])

    affine = np.eye(4)
    img_3d = Nifti1Image(np.ones((10, 10, 10)), affine)

    # Tests with return_iterator=False
    img_4d_1 = _utils.check_niimg_4d([img_3d, img_3d])
    assert_true(img_4d_1.get_data().shape == (10, 10, 10, 2))
    assert_array_equal(img_4d_1.get_affine(), affine)

    img_4d_2 = _utils.check_niimg_4d(img_4d_1)
    assert_array_equal(img_4d_2.get_data(), img_4d_2.get_data())
    assert_array_equal(img_4d_2.get_affine(), img_4d_2.get_affine())

    # Tests with return_iterator=True
    img_3d_iterator = _utils.check_niimg_4d([img_3d, img_3d],
                                          return_iterator=True)
    img_3d_iterator_length = sum(1 for _ in img_3d_iterator)
    assert_true(img_3d_iterator_length == 2)

    img_3d_iterator_1 = _utils.check_niimg_4d([img_3d, img_3d],
                                            return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimg_4d(img_3d_iterator_1,
                                            return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert_true(img_1.get_data().shape == (10, 10, 10))
        assert_array_equal(img_1.get_data(), img_2.get_data())
        assert_array_equal(img_1.get_affine(), img_2.get_affine())

    img_3d_iterator_1 = _utils.check_niimg_4d([img_3d, img_3d],
                                            return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimg_4d(img_4d_1,
                                            return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert_true(img_1.get_data().shape == (10, 10, 10))
        assert_array_equal(img_1.get_data(), img_2.get_data())
        assert_array_equal(img_1.get_affine(), img_2.get_affine())

    # This should raise an error: a 3D img is given and we want a 4D
    assert_raises_regex(DimensionError, 'Data must be a 4D Niimg-like object but '
                        'you provided',
                        _utils.check_niimg_4d, img_3d)

    # Test a Niimg-like object that does not hold a shape attribute
    phony_img = PhonyNiimage()
    _utils.check_niimg_4d(phony_img)

    a = nibabel.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    b = np.zeros((10, 10, 10))
    c = _utils.check_niimg_4d([a, b], return_iterator=True)
    assert_raises_regex(TypeError, 'Error encountered while loading image #1',
                        list, c)

    b = nibabel.Nifti1Image(np.zeros((10, 20, 10)), np.eye(4))
    c = _utils.check_niimg_4d([a, b], return_iterator=True)
    assert_raises_regex(
        ValueError,
        'Field of view of image #1 is different from reference FOV',
        list, c)


def test_check_niimg():
    affine = np.eye(4)
    img_3d = Nifti1Image(np.ones((10, 10, 10)), affine)
    img_4d = Nifti1Image(np.ones((10, 10, 10, 4)), affine)
    img_3_3d = [[[img_3d, img_3d]]]
    img_2_4d = [[img_4d, img_4d]]

    assert_raises_regex(
        DimensionError,
        'Data must be a 2D Niimg-like object but you provided a list of list '
        'of list of 3D images.', _utils.check_niimg, img_3_3d, ensure_ndim=2)

    assert_raises_regex(
        DimensionError,
        'Data must be a 4D Niimg-like object but you provided a list of list '
        'of 4D images.', _utils.check_niimg, img_2_4d, ensure_ndim=4)


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
    img1c = Nifti1Image(np.ones(shape3), affine)

    # Regression test for #601. Dimensionality of first image was not checked
    # properly
    assert_raises_regex(DimensionError, 'Data must be a 4D Niimg-like object but '
                        'you provided',
                        _utils.concat_niimgs, [img4d], ensure_ndim=4)

    # check basic concatenation with equal shape/affine
    concatenated = _utils.concat_niimgs((img1, img3, img1))

    assert_raises_regex(DimensionError, 'Data must be a 4D Niimg-like object but '
                        'you provided',
                        _utils.concat_niimgs, [img1, img4d])

    # smoke-test auto_resample
    concatenated = _utils.concat_niimgs((img1, img1b, img1c),
        auto_resample=True)
    assert_true(concatenated.shape == img1.shape + (3, ))

    # check error for non-forced but necessary resampling
    assert_raises_regex(ValueError, 'Field of view of image',
                        _utils.concat_niimgs, [img1, img2],
                        auto_resample=False)

    # test list of 4D niimgs as input
    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    _, tmpimg2 = tempfile.mkstemp(suffix='.nii')
    try:
        nibabel.save(img1, tmpimg1)
        nibabel.save(img3, tmpimg2)
        concatenated = _utils.concat_niimgs([tmpimg1, tmpimg2])
        assert_array_equal(
            concatenated.get_data()[..., 0], img1.get_data())
        assert_array_equal(
            concatenated.get_data()[..., 1], img3.get_data())
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)


def nifti_generator(buffer):
    for i in range(10):
        buffer.append(Nifti1Image(np.random.random((10, 10, 10)), np.eye(4)))
        yield buffer[-1]


def test_iterator_generator():
    # Create a list of random images
    l = [Nifti1Image(np.random.random((10, 10, 10)), np.eye(4))
         for i in range(10)]
    cc = _utils.concat_niimgs(l)
    assert_equal(cc.shape[-1], 10)
    assert_array_almost_equal(cc.get_data()[..., 0], l[0].get_data())

    # Same with iteration
    i = image.iter_img(l)
    cc = _utils.concat_niimgs(i)
    assert_equal(cc.shape[-1], 10)
    assert_array_almost_equal(cc.get_data()[..., 0], l[0].get_data())

    # Now, a generator
    b = []
    g = nifti_generator(b)
    cc = _utils.concat_niimgs(g)
    assert_equal(cc.shape[-1], 10)
    assert_equal(len(b), 10)
