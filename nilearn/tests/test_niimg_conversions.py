"""
Test the niimg_conversions

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel
from nibabel import Nifti1Image

import nilearn as ni
from nilearn import _utils, image
from nilearn._utils.exceptions import DimensionError
from nilearn._utils import testing, niimg_conversions
from nilearn._utils.testing import with_memory_profiler
from nilearn._utils.testing import assert_memory_less_than
from nilearn._utils.niimg_conversions import _iter_check_niimg
from nilearn.image import get_data


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

    @property
    def _data_cache(self):
        return self.data

    @property
    def _dataobj(self):
        return self.data


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

    with pytest.raises(ValueError,
                       match='[ac] and [ac] do not have the same affine'
                       ):
        niimg_conversions._check_same_fov(
            a=shape_a_affine_a, b=shape_a_affine_a_2,
            c=shape_a_affine_b, raise_error=True
        )
    with pytest.raises(ValueError,
                       match='[ab] and [ab] do not have the same shape'
                       ):
        niimg_conversions._check_same_fov(
            a=shape_a_affine_a, b=shape_b_affine_a, raise_error=True)
    with pytest.raises(ValueError,
                       match='[ab] and [ab] do not have the same affine'
                       ):
        niimg_conversions._check_same_fov(
            a=shape_b_affine_b, b=shape_a_affine_a, raise_error=True)

    with pytest.raises(ValueError,
                       match='[ab] and [ab] do not have the same shape'
                       ):
        niimg_conversions._check_same_fov(
            a=shape_b_affine_b, b=shape_a_affine_a, raise_error=True)


def test_check_niimg_3d():
    # check error for non-forced but necessary resampling
    with pytest.raises(TypeError, match='nibabel format'):
        _utils.check_niimg(0)

    # check error for non-forced but necessary resampling
    with pytest.raises(TypeError, match='empty object'):
        _utils.check_niimg([])

    # Test dimensionality error
    img = Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    with pytest.raises(
            TypeError,
            match="Input data has incompatible dimensionality: "
                  "Expected dimension is 3D and you provided a list "
                  "of 3D images \\(4D\\)."):
        _utils.check_niimg_3d([img, img])

    # Check that a filename does not raise an error
    data = np.zeros((40, 40, 40, 1))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True) as filename:
        _utils.check_niimg_3d(filename)

    # check data dtype equal with dtype='auto'
    img_check = _utils.check_niimg_3d(img, dtype='auto')
    assert get_data(img).dtype.kind == get_data(img_check).dtype.kind


def test_check_niimg_4d():
    with pytest.raises(TypeError, match='nibabel format'):
        _utils.check_niimg_4d(0)

    with pytest.raises(TypeError, match='empty object'):
        _utils.check_niimg_4d([])

    affine = np.eye(4)
    img_3d = Nifti1Image(np.ones((10, 10, 10)), affine)

    # Tests with return_iterator=False
    img_4d_1 = _utils.check_niimg_4d([img_3d, img_3d])
    assert get_data(img_4d_1).shape == (10, 10, 10, 2)
    assert_array_equal(img_4d_1.affine, affine)

    img_4d_2 = _utils.check_niimg_4d(img_4d_1)
    assert_array_equal(get_data(img_4d_2), get_data(img_4d_2))
    assert_array_equal(img_4d_2.affine, img_4d_2.affine)

    # Tests with return_iterator=True
    img_3d_iterator = _utils.check_niimg_4d([img_3d, img_3d],
                                            return_iterator=True)
    img_3d_iterator_length = sum(1 for _ in img_3d_iterator)
    assert img_3d_iterator_length == 2

    img_3d_iterator_1 = _utils.check_niimg_4d([img_3d, img_3d],
                                              return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimg_4d(img_3d_iterator_1,
                                              return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert get_data(img_1).shape == (10, 10, 10)
        assert_array_equal(get_data(img_1), get_data(img_2))
        assert_array_equal(img_1.affine, img_2.affine)

    img_3d_iterator_1 = _utils.check_niimg_4d([img_3d, img_3d],
                                              return_iterator=True)
    img_3d_iterator_2 = _utils.check_niimg_4d(img_4d_1,
                                              return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert get_data(img_1).shape == (10, 10, 10)
        assert_array_equal(get_data(img_1), get_data(img_2))
        assert_array_equal(img_1.affine, img_2.affine)

    # This should raise an error: a 3D img is given and we want a 4D
    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 4D and you provided a "
                             "3D image."):
        _utils.check_niimg_4d(img_3d)

    # Test a Niimg-like object that does not hold a shape attribute
    phony_img = PhonyNiimage()
    _utils.check_niimg_4d(phony_img)

    a = nibabel.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    b = np.zeros((10, 10, 10))
    c = _utils.check_niimg_4d([a, b], return_iterator=True)
    with pytest.raises(TypeError,
                       match='Error encountered while loading image #1'):
        list(c)

    b = nibabel.Nifti1Image(np.zeros((10, 20, 10)), np.eye(4))
    c = _utils.check_niimg_4d([a, b], return_iterator=True)
    with pytest.raises(
            ValueError,
            match='Field of view of image #1 is different from reference FOV'):
        list(c)


def test_check_niimg():
    affine = np.eye(4)
    img_3d = Nifti1Image(np.ones((10, 10, 10)), affine)
    img_4d = Nifti1Image(np.ones((10, 10, 10, 4)), affine)
    img_3_3d = [[[img_3d, img_3d]]]
    img_2_4d = [[img_4d, img_4d]]

    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 2D and you provided "
                             "a list of list of list of 3D images \\(6D\\)"
                       ):
        _utils.check_niimg(img_3_3d, ensure_ndim=2)

    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 4D and you provided "
                             "a list of list of 4D images \\(6D\\)"
                       ):
        _utils.check_niimg(img_2_4d, ensure_ndim=4)

    # check data dtype equal with dtype='auto'
    img_3d_check = _utils.check_niimg(img_3d, dtype='auto')
    assert (
        get_data(img_3d).dtype.kind == get_data(img_3d_check).dtype.kind)

    img_4d_check = _utils.check_niimg(img_4d, dtype='auto')
    assert (
        get_data(img_4d).dtype.kind == get_data(img_4d_check).dtype.kind)


def test_check_niimg_pathlike():
    img = Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))

    with testing.write_tmp_imgs(img, create_files=True) as filename:
        filename = Path(filename)
        _utils.check_niimg_3d(filename)


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

    assert ni.EXPAND_PATH_WILDCARDS == True
    # Check bad filename
    # Non existing file (with no magic) raise a ValueError exception
    with pytest.raises(ValueError, match=file_not_found_msg % nofile_path):
        _utils.check_niimg(nofile_path)
    # Non matching wildcard raises a ValueError exception
    with pytest.raises(
            ValueError,
            match=wildcards_msg % re.escape(nofile_path_wildcards)):
        _utils.check_niimg(nofile_path_wildcards)

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
        assert_array_equal(get_data(_utils.check_niimg(filename)),
                           get_data(img_3d))
    # No globbing behavior
    with testing.write_tmp_imgs(img_3d, create_files=True) as filename:
        assert_array_equal(
            get_data(_utils.check_niimg(filename, wildcards=False)),
            get_data(img_3d))

    #######
    # Testing with an existing filename
    with testing.write_tmp_imgs(img_4d, create_files=True) as filename:
        assert_array_equal(get_data(_utils.check_niimg(filename)),
                           get_data(img_4d))
    # No globbing behavior
    with testing.write_tmp_imgs(img_4d, create_files=True) as filename:
        assert_array_equal(get_data(_utils.check_niimg(filename,
                                                       wildcards=False)),
                           get_data(img_4d))

    #######
    # Testing with a glob matching exactly one filename
    # Using a glob matching one file containing a 3d image returns a 4d image
    # with 1 as last dimension.
    with testing.write_tmp_imgs(img_3d,
                                create_files=True,
                                use_wildcards=True) as globs:
        glob_input = tmp_dir + globs
        assert_array_equal(get_data(_utils.check_niimg(glob_input))[..., 0],
                           get_data(img_3d))
    # Disabled globbing behavior should raise an ValueError exception
    with testing.write_tmp_imgs(img_3d,
                                create_files=True,
                                use_wildcards=True) as globs:
        glob_input = tmp_dir + globs
        with pytest.raises(ValueError,
                           match=file_not_found_msg % re.escape(glob_input)
                           ):
            _utils.check_niimg(glob_input, wildcards=False)

    #######
    # Testing with a glob matching multiple filenames
    img_4d = _utils.check_niimg_4d((img_3d, img_3d))
    with testing.write_tmp_imgs(img_3d, img_3d,
                                create_files=True,
                                use_wildcards=True) as globs:
        assert_array_equal(get_data(_utils.check_niimg(glob_input)),
                           get_data(img_4d))

    #######
    # Test when global variable is set to False => no globbing allowed
    ni.EXPAND_PATH_WILDCARDS = False

    # Non existing filename (/tmp/nofile) could match an existing one through
    # globbing but global wildcards variable overrides this feature => raises
    # a ValueError
    with pytest.raises(ValueError, match=file_not_found_msg % nofile_path):
        _utils.check_niimg(nofile_path)

    # Verify wildcards function parameter has no effect
    with pytest.raises(ValueError, match=file_not_found_msg % nofile_path):
        _utils.check_niimg(nofile_path, wildcards=False)

    # Testing with an exact filename matching (3d case)
    with testing.write_tmp_imgs(img_3d, create_files=True) as filename:
        assert_array_equal(get_data(_utils.check_niimg(filename)),
                           get_data(img_3d))

    # Testing with an exact filename matching (4d case)
    with testing.write_tmp_imgs(img_4d, create_files=True) as filename:
        assert_array_equal(get_data(_utils.check_niimg(filename)),
                           get_data(img_4d))

    # Reverting to default behavior
    ni.EXPAND_PATH_WILDCARDS = True


def test_iter_check_niimgs(tmp_path):
    no_file_matching = "No files matching path: %s"
    affine = np.eye(4)
    img_4d = Nifti1Image(np.ones((10, 10, 10, 4)), affine)
    img_2_4d = [[img_4d, img_4d]]

    for empty in ((), [], (i for i in ()), [i for i in ()]):
        with pytest.raises(ValueError, match="Input niimgs list is empty."):
            list(_iter_check_niimg(empty))

    nofile_path = "/tmp/nofile"
    with pytest.raises(ValueError, match=no_file_matching % nofile_path):
        list(_iter_check_niimg(nofile_path))

    # Create a test file
    fd, filename = tempfile.mkstemp(prefix="nilearn_test",
                                    suffix=".nii",
                                    dir=str(tmp_path))
    os.close(fd)
    img_4d.to_filename(filename)
    niimgs = list(_iter_check_niimg([filename]))
    assert_array_equal(get_data(niimgs[0]),
                       get_data(_utils.check_niimg(img_4d)))
    del img_4d
    del niimgs
    os.remove(filename)

    # Regular case
    niimgs = list(_iter_check_niimg(img_2_4d))
    assert_array_equal(get_data(niimgs[0]),
                       get_data(_utils.check_niimg(img_2_4d)))


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


def test_repr_niimgs(tmp_path):
    # Tests with file path
    assert _utils._repr_niimgs("test") == "test"
    assert _utils._repr_niimgs("test", shorten=False) == "test"

    # Shortening long names by default
    long_name = 'this-is-a-very-long-name-for-a-nifti-file.nii'
    short_name = 'this-is-a-very-lon...'
    assert _utils._repr_niimgs(long_name) == short_name
    # Explicit shortening of long names
    assert _utils._repr_niimgs(long_name, shorten=True) == short_name
    # Force long display of long names
    assert _utils._repr_niimgs(long_name, shorten=False) == long_name

    # Tests with list of file paths
    assert _utils._repr_niimgs(["test", "retest"]) == "[test, retest]"
    assert _utils._repr_niimgs(["test", "retest"], shorten=False) == "[test, retest]"

    # Lists of long names up to length 3
    list_of_size_3 = ['this-is-a-very-long-name-for-a-nifti-file.nii',
                      'this-is-another-very-long-name-for-a-nifti-file.nii',
                      'this-is-again-another-very-long-name-for-a-nifti-file.nii']
    # Explicit shortening, all 3 names are displayed, but shortened
    shortened_rep_list_of_size_3 = ("[this-is-a-very-lon...,"
                                    " this-is-another-ve...,"
                                    " this-is-again-anot...]")

    assert (_utils._repr_niimgs(list_of_size_3, shorten=True) ==
            shortened_rep_list_of_size_3)

    # Force display, all 3 names are displayed
    long_rep_list_of_size_3 = ("[this-is-a-very-long-name-for-a-nifti-file.nii,"
                               " this-is-another-very-long-name-for-a-nifti-file.nii,"
                               " this-is-again-another-very-long-name-for-a-nifti-file.nii]")

    assert (_utils._repr_niimgs(list_of_size_3, shorten=False) ==
            long_rep_list_of_size_3)

    # Lists longer than 3
    # Small names - Explicit shortening
    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    shortened_rep_long_list_small_names = ("[test,\n"
                                           "         ...\n"
                                           " rereretest]")

    assert (_utils._repr_niimgs(long_list_small_names, shorten=True) ==
            shortened_rep_long_list_small_names)

    # Small names - Force full display
    long_rep_long_list_small_names = ("[test,\n"
                                      " retest,\n"
                                      " reretest,\n"
                                      " rereretest]")

    assert (_utils._repr_niimgs(long_list_small_names, shorten=False) ==
            long_rep_long_list_small_names)

    # Long names - Explicit shortening
    list_of_size_4 = list_of_size_3 + ['this-is-again-another-super-very-long-name-for-a-nifti-file.nii']
    shortened_rep_long_list_long_names = ("[this-is-a-very-lon...,\n"
                                          "         ...\n"
                                          " this-is-again-anot...]")

    assert (_utils._repr_niimgs(list_of_size_4, shorten=True) ==
            shortened_rep_long_list_long_names)

    # Long names - Force full display in pretty print style for readability

    long_rep_long_list_long_names = (long_rep_list_of_size_3[:-1].replace(",", ",\n") +
                                     ",\n this-is-again-another-super-very-long-name-for-a-nifti-file.nii]")

    assert (_utils._repr_niimgs(list_of_size_4, shorten=False) ==
            long_rep_long_list_long_names)

    # Tests with pathlib
    # Case with very long path and small filename
    long_path = Path('/this/is/a/fake/long/path/to/file.nii')
    short_path = Path('.../path/to/file.nii')
    assert _utils._repr_niimgs(long_path, shorten=True) == str(short_path)
    assert _utils._repr_niimgs(long_path, shorten=False) == str(long_path)

    # Case with very long path but very long filename
    long_path_long_name = Path('/this/is/a/fake/long/path/to/my_file_with_a_very_long_name.nii')
    short_name = 'my_file_with_a_ver...'
    assert _utils._repr_niimgs(long_path_long_name, shorten=True) == short_name
    assert _utils._repr_niimgs(long_path_long_name, shorten=False) == str(long_path_long_name)

    # Case with lists
    list_of_paths = [Path('/this/is/a/fake/long/path/to/file.nii'),
                     Path('/this/is/a/fake/long/path/to/another/file2.nii'),
                     Path('/again/another/fake/long/path/to/file3.nii'),
                     Path('/this/is/a/fake/long/path/to/a-very-long-file-name.nii')]

    shortened_list_of_paths = ("[...{0},\n"
                               "         ...\n"
                               " a-very-long-file-n...]".format(str(Path("/path/to/file.nii"))))

    assert _utils._repr_niimgs(list_of_paths, shorten=True) == shortened_list_of_paths

    long_list_of_paths = "[%s]" % ',\n '.join(_ for _ in [str(_) for _ in list_of_paths])

    assert _utils._repr_niimgs(list_of_paths, shorten=False) == long_list_of_paths

    # Create phony Niimg without filename
    affine = np.eye(4)
    shape = (10, 10, 10)
    img1 = Nifti1Image(np.ones(shape), affine)
    # Shorten has no effect in this case
    for shorten in [True, False]:
        assert (
            _utils._repr_niimgs(img1, shorten=shorten).replace("10L", "10") ==
            ("%s(\nshape=%s,\naffine=%s\n)" %
                (img1.__class__.__name__,
                repr(shape), repr(affine))))

    # Add filename long enough to qualify for shortening
    fd, tmpimg1 = tempfile.mkstemp(suffix='_very_long.nii',
                                   dir=str(tmp_path))
    os.close(fd)
    nibabel.save(img1, tmpimg1)
    assert (
        _utils._repr_niimgs(img1, shorten=False) ==
        ("%s('%s')" % (img1.__class__.__name__, img1.get_filename())))
    assert (
        _utils._repr_niimgs(img1, shorten=True) ==
        ("%s('%s...')" % (img1.__class__.__name__, Path(img1.get_filename()).name[:18])))


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
                            "a list of 4D images \\(5D\\)")
    with pytest.raises(DimensionError, match=_dimension_error_msg):
        _utils.concat_niimgs([img4d], ensure_ndim=4)

    # check basic concatenation with equal shape/affine
    concatenated = _utils.concat_niimgs((img1, img3, img1))

    with pytest.raises(DimensionError, match=_dimension_error_msg):
        _utils.concat_niimgs([img1, img4d])

    # smoke-test auto_resample
    concatenated = _utils.concat_niimgs((img1, img1b, img1c),
                                        auto_resample=True)
    assert concatenated.shape == img1.shape + (3, )

    # check error for non-forced but necessary resampling
    with pytest.raises(ValueError, match='Field of view of image'):
        _utils.concat_niimgs([img1, img2], auto_resample=False)

    # test list of 4D niimgs as input
    tempdir = tempfile.mkdtemp()
    tmpimg1 = os.path.join(tempdir, '1.nii')
    tmpimg2 = os.path.join(tempdir, '2.nii')
    try:
        nibabel.save(img1, tmpimg1)
        nibabel.save(img3, tmpimg2)
        concatenated = _utils.concat_niimgs(os.path.join(tempdir, '*'))
        assert_array_equal(
            get_data(concatenated)[..., 0], get_data(img1))
        assert_array_equal(
            get_data(concatenated)[..., 1], get_data(img3))
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)
        if os.path.exists(tempdir):
            os.removedirs(tempdir)

    img5d = Nifti1Image(np.ones((2, 2, 2, 2, 2)), affine)
    with pytest.raises(TypeError,
                       match='Concatenated images must be 3D or 4D. '
                             'You gave a list of 5D images'):
        _utils.concat_niimgs([img5d, img5d])


def test_concat_niimg_dtype():
    shape = [2, 3, 4]
    vols = [nibabel.Nifti1Image(
        np.zeros(shape + [n_scans]).astype(np.int16), np.eye(4))
            for n_scans in [1, 5]]
    nimg = _utils.concat_niimgs(vols)
    assert get_data(nimg).dtype == np.float32
    nimg = _utils.concat_niimgs(vols, dtype=None)
    assert get_data(nimg).dtype == np.int16


def nifti_generator(buffer):
    rng = np.random.RandomState(42)
    for i in range(10):
        buffer.append(Nifti1Image(rng.random_sample((10, 10, 10)), np.eye(4)))
        yield buffer[-1]


def test_iterator_generator():
    # Create a list of random images
    rng = np.random.RandomState(42)
    list_images = [
        Nifti1Image(
            rng.random_sample((10, 10, 10)), np.eye(4)
        )
        for i in range(10)
    ]
    cc = _utils.concat_niimgs(list_images)
    assert cc.shape[-1] == 10
    assert_array_almost_equal(get_data(cc)[..., 0], get_data(list_images[0]))

    # Same with iteration
    i = image.iter_img(list_images)
    cc = _utils.concat_niimgs(i)
    assert cc.shape[-1] == 10
    assert_array_almost_equal(get_data(cc)[..., 0], get_data(list_images[0]))

    # Now, a generator
    b = []
    g = nifti_generator(b)
    cc = _utils.concat_niimgs(g)
    assert cc.shape[-1] == 10
    assert len(b) == 10
