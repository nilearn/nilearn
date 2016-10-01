"""
Test the niimg_conversions

This test file is in nilearn/tests because nosetests seems to ignore modules
whose name starts with an underscore
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os
import re
import tempfile

from nose.tools import assert_equal, assert_true

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel
from nibabel import Nifti1Image

import nilearn as ni
from nilearn import _utils, image
from nilearn._utils.exceptions import DimensionError
from nilearn._utils import testing, niimg_conversions
from nilearn._utils.testing import assert_raises_regex
from nilearn._utils.testing import with_memory_profiler
from nilearn._utils.testing import assert_memory_less_than
from nilearn._utils.niimg_conversions import _iter_check_niimg
from nilearn._utils.compat import get_affine as _get_affine


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
    assert_raises_regex(TypeError,
                        "Input data has incompatible dimensionality: "
                        "Expected dimension is 3D and you provided a list "
                        "of 3D images \(4D\).",
                        _utils.check_niimg_3d, [img, img])

    # Check that a filename does not raise an error
    data = np.zeros((40, 40, 40, 1))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True) as filename:
        _utils.check_niimg_3d(filename)

    # check data dtype equal with dtype='auto'
    img_check = _utils.check_niimg_3d(img, dtype='auto')
    assert_equal(img.get_data().dtype.kind, img_check.get_data().dtype.kind)


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
    assert_array_equal(_get_affine(img_4d_1), affine)

    img_4d_2 = _utils.check_niimg_4d(img_4d_1)
    assert_array_equal(img_4d_2.get_data(), img_4d_2.get_data())
    assert_array_equal(_get_affine(img_4d_2), _get_affine(img_4d_2))

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
        assert_array_equal(_get_affine(img_1), _get_affine(img_2))

    img_3d_iterator_1 = _utils.check_niimg_4d([img_3d, img_3d],
                                              return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimg_4d(img_4d_1,
                                              return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert_true(img_1.get_data().shape == (10, 10, 10))
        assert_array_equal(img_1.get_data(), img_2.get_data())
        assert_array_equal(_get_affine(img_1), _get_affine(img_2))

    # This should raise an error: a 3D img is given and we want a 4D
    assert_raises_regex(DimensionError,
                        "Input data has incompatible dimensionality: "
                        "Expected dimension is 4D and you provided a "
                        "3D image.",
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
        "Input data has incompatible dimensionality: "
        "Expected dimension is 2D and you provided "
        "a list of list of list of 3D images \(6D\)",
        _utils.check_niimg, img_3_3d, ensure_ndim=2)

    assert_raises_regex(
        DimensionError,
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a list of list of 4D images \(6D\)",
        _utils.check_niimg, img_2_4d, ensure_ndim=4)

    # check data dtype equal with dtype='auto'
    img_3d_check = _utils.check_niimg(img_3d, dtype='auto')
    assert_equal(img_3d.get_data().dtype.kind, img_3d_check.get_data().dtype.kind)

    img_4d_check = _utils.check_niimg(img_4d, dtype='auto')
    assert_equal(img_4d.get_data().dtype.kind, img_4d_check.get_data().dtype.kind)


def test_check_niimg_wildcards():
    tmp_dir = tempfile.tempdir + os.sep
    nofile_path = "/tmp/nofile"
    nofile_path_wildcards = "/tmp/no*file"
    wildcards_msg = ("No files matching the entered niimg expression: "
                     "'%s'.\n You may have left wildcards usage "
                     "activated: please set the global constant "
                     "'nilearn.EXPAND_PATH_WILDCARDS' to False to "
                     "deactivate this behavior.")

    file_not_found_msg = "File not found: '%s'"

    assert_equal(ni.EXPAND_PATH_WILDCARDS, True)
    # Check bad filename
    # Non existing file (with no magic) raise a ValueError exception
    assert_raises_regex(ValueError, file_not_found_msg % nofile_path,
                        _utils.check_niimg, nofile_path)
    # Non matching wildcard raises a ValueError exception
    assert_raises_regex(ValueError,
                        wildcards_msg % re.escape(nofile_path_wildcards),
                        _utils.check_niimg, nofile_path_wildcards)

    # First create some testing data
    data_3d = np.zeros((40, 40, 40))
    data_3d[20, 20, 20] = 1
    img_3d = Nifti1Image(data_3d, np.eye(4))

    data_4d = np.zeros((40, 40, 40, 3))
    data_4d[20, 20, 20] = 1
    img_4d = Nifti1Image(data_4d, np.eye(4))

    #######
    # Testing with an existing filename
    with testing.write_tmp_imgs(img_3d, create_files=True) as filename:
        assert_array_equal(_utils.check_niimg(filename).get_data(),
                           img_3d.get_data())
    # No globbing behavior
    with testing.write_tmp_imgs(img_3d, create_files=True) as filename:
        assert_array_equal(_utils.check_niimg(filename,
                                              wildcards=False).get_data(),
                           img_3d.get_data())

    #######
    # Testing with an existing filename
    with testing.write_tmp_imgs(img_4d, create_files=True) as filename:
        assert_array_equal(_utils.check_niimg(filename).get_data(),
                           img_4d.get_data())
    # No globbing behavior
    with testing.write_tmp_imgs(img_4d, create_files=True) as filename:
        assert_array_equal(_utils.check_niimg(filename,
                                              wildcards=False).get_data(),
                           img_4d.get_data())

    #######
    # Testing with a glob matching exactly one filename
    # Using a glob matching one file containing a 3d image returns a 4d image
    # with 1 as last dimension.
    with testing.write_tmp_imgs(img_3d,
                                create_files=True,
                                use_wildcards=True) as globs:
        glob_input = tmp_dir + globs
        assert_array_equal(_utils.check_niimg(glob_input).get_data()[..., 0],
                           img_3d.get_data())
    # Disabled globbing behavior should raise an ValueError exception
    with testing.write_tmp_imgs(img_3d,
                                create_files=True,
                                use_wildcards=True) as globs:
        glob_input = tmp_dir + globs
        assert_raises_regex(ValueError,
                            file_not_found_msg % re.escape(glob_input),
                            _utils.check_niimg,
                            glob_input,
                            wildcards=False)

    #######
    # Testing with a glob matching multiple filenames
    img_4d = _utils.check_niimg_4d((img_3d, img_3d))
    with testing.write_tmp_imgs(img_3d, img_3d,
                                create_files=True,
                                use_wildcards=True) as globs:
        assert_array_equal(_utils.check_niimg(glob_input).get_data(),
                           img_4d.get_data())

    #######
    # Test when global variable is set to False => no globbing allowed
    ni.EXPAND_PATH_WILDCARDS = False

    # Non existing filename (/tmp/nofile) could match an existing one through
    # globbing but global wildcards variable overrides this feature => raises
    # a ValueError
    assert_raises_regex(ValueError,
                        file_not_found_msg % nofile_path,
                        _utils.check_niimg, nofile_path)

    # Verify wildcards function parameter has no effect
    assert_raises_regex(ValueError,
                        file_not_found_msg % nofile_path,
                        _utils.check_niimg, nofile_path, wildcards=False)

    # Testing with an exact filename matching (3d case)
    with testing.write_tmp_imgs(img_3d, create_files=True) as filename:
        assert_array_equal(_utils.check_niimg(filename).get_data(),
                           img_3d.get_data())

    # Testing with an exact filename matching (4d case)
    with testing.write_tmp_imgs(img_4d, create_files=True) as filename:
        assert_array_equal(_utils.check_niimg(filename).get_data(),
                           img_4d.get_data())

    # Reverting to default behavior
    ni.EXPAND_PATH_WILDCARDS = True


def test_iter_check_niimgs():
    no_file_matching = "No files matching path: %s"
    affine = np.eye(4)
    img_4d = Nifti1Image(np.ones((10, 10, 10, 4)), affine)
    img_2_4d = [[img_4d, img_4d]]

    for empty in ((), [], (i for i in ()), [i for i in ()]):
        assert_raises_regex(ValueError,
                            "Input niimgs list is empty.",
                            list, _iter_check_niimg(empty))

    nofile_path = "/tmp/nofile"
    assert_raises_regex(ValueError,
                        no_file_matching % nofile_path,
                        list, _iter_check_niimg(nofile_path))

    # Create a test file
    filename = tempfile.mktemp(prefix="nilearn_test",
                               suffix=".nii",
                               dir=None)
    img_4d.to_filename(filename)
    niimgs = list(_iter_check_niimg([filename]))
    assert_array_equal(niimgs[0].get_data(),
                       _utils.check_niimg(img_4d).get_data())
    del img_4d
    del niimgs
    os.remove(filename)

    # Regular case
    niimgs = list(_iter_check_niimg(img_2_4d))
    assert_array_equal(niimgs[0].get_data(),
                       _utils.check_niimg(img_2_4d).get_data())


def _check_memory(list_img_3d):
    # We intentionally add an offset of memory usage to avoid non trustable
    # measures with memory_profiler.
    mem_offset = b'a' * 100 * 1024 ** 2
    list(_iter_check_niimg(list_img_3d))
    return mem_offset


@with_memory_profiler
def test_iter_check_niimgs_memory():
    # Verify that iterating over a list of images doesn't consume extra
    # memory.
    assert_memory_less_than(100, 0.1, _check_memory,
                            [Nifti1Image(np.ones((100, 100, 200)), np.eye(4))
                             for i in range(10)])


def test_repr_niimgs():
    # Test with file path
    assert_equal(_utils._repr_niimgs("test"), "test")
    assert_equal(_utils._repr_niimgs(["test", "retest"]), "[test, retest]")
    # Create phony Niimg with filename
    affine = np.eye(4)
    shape = (10, 10, 10)
    img1 = Nifti1Image(np.ones(shape), affine)
    assert_equal(
        _utils._repr_niimgs(img1).replace("10L","10"),
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
    _dimension_error_msg = ("Input data has incompatible dimensionality: "
                            "Expected dimension is 4D and you provided "
                            "a list of 4D images \(5D\)")
    assert_raises_regex(DimensionError, _dimension_error_msg,
                        _utils.concat_niimgs, [img4d], ensure_ndim=4)

    # check basic concatenation with equal shape/affine
    concatenated = _utils.concat_niimgs((img1, img3, img1))

    assert_raises_regex(DimensionError, _dimension_error_msg,
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
    tempdir = tempfile.mkdtemp()
    tmpimg1 = os.path.join(tempdir, '1.nii')
    tmpimg2 = os.path.join(tempdir, '2.nii')
    try:
        nibabel.save(img1, tmpimg1)
        nibabel.save(img3, tmpimg2)
        concatenated = _utils.concat_niimgs(os.path.join(tempdir, '*'))
        assert_array_equal(
            concatenated.get_data()[..., 0], img1.get_data())
        assert_array_equal(
            concatenated.get_data()[..., 1], img3.get_data())
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)
        if os.path.exists(tempdir):
            os.removedirs(tempdir)

    img5d = Nifti1Image(np.ones((2, 2, 2, 2, 2)), affine)
    assert_raises_regex(TypeError, 'Concatenated images must be 3D or 4D. '
                        'You gave a list of 5D images', _utils.concat_niimgs,
                        [img5d, img5d])


def test_concat_niimg_dtype():
    shape = [2, 3, 4]
    vols = [nibabel.Nifti1Image(
        np.zeros(shape + [n_scans]).astype(np.int16), np.eye(4))
            for n_scans in [1, 5]]
    nimg = _utils.concat_niimgs(vols)
    assert_equal(nimg.get_data().dtype, np.float32)
    nimg = _utils.concat_niimgs(vols, dtype=None)
    assert_equal(nimg.get_data().dtype, np.int16)


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
