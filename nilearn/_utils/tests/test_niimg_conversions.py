"""Test the niimg_conversions.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

# Author: Gael Varoquaux, Alexandre Abraham

import os
import re
from pathlib import Path
from tempfile import mkstemp

import numpy as np
import pytest
from nibabel import Nifti1Image, spatialimages
from numpy.testing import assert_array_equal

import nilearn as ni
from nilearn._utils import (
    check_niimg,
    check_niimg_3d,
    check_niimg_4d,
    repr_niimgs,
)
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.niimg_conversions import check_same_fov, iter_check_niimg
from nilearn._utils.testing import (
    assert_memory_less_than,
    with_memory_profiler,
    write_imgs_to_path,
)
from nilearn.image import get_data


class PhonyNiimage(spatialimages.SpatialImage):
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


def test_check_same_fov(affine_eye):
    affine_b = affine_eye * 2

    shape_a = (2, 2, 2)
    shape_b = (3, 3, 3)

    shape_a_affine_a = Nifti1Image(np.empty(shape_a), affine_eye)
    shape_a_affine_a_2 = Nifti1Image(np.empty(shape_a), affine_eye)
    shape_a_affine_b = Nifti1Image(np.empty(shape_a), affine_b)
    shape_b_affine_a = Nifti1Image(np.empty(shape_b), affine_eye)
    shape_b_affine_b = Nifti1Image(np.empty(shape_b), affine_b)

    check_same_fov(a=shape_a_affine_a, b=shape_a_affine_a_2, raise_error=True)

    with pytest.raises(
        ValueError, match="[ac] and [ac] do not have the same affine"
    ):
        check_same_fov(
            a=shape_a_affine_a,
            b=shape_a_affine_a_2,
            c=shape_a_affine_b,
            raise_error=True,
        )
    with pytest.raises(
        ValueError, match="[ab] and [ab] do not have the same shape"
    ):
        check_same_fov(
            a=shape_a_affine_a, b=shape_b_affine_a, raise_error=True
        )
    with pytest.raises(
        ValueError, match="[ab] and [ab] do not have the same affine"
    ):
        check_same_fov(
            a=shape_b_affine_b, b=shape_a_affine_a, raise_error=True
        )

    with pytest.raises(
        ValueError, match="[ab] and [ab] do not have the same shape"
    ):
        check_same_fov(
            a=shape_b_affine_b, b=shape_a_affine_a, raise_error=True
        )


def test_check_niimg_3d(affine_eye, img_3d_zeros_eye, tmp_path):
    # check error for non-forced but necessary resampling
    with pytest.raises(TypeError, match="nibabel format"):
        check_niimg(0)

    # check error for non-forced but necessary resampling
    with pytest.raises(TypeError, match="empty object"):
        check_niimg([])

    # Test dimensionality error
    with pytest.raises(
        TypeError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided a list "
        "of 3D images \\(4D\\).",
    ):
        check_niimg_3d([img_3d_zeros_eye, img_3d_zeros_eye])

    # Check that a filename does not raise an error
    data = np.zeros((40, 40, 40, 1))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, affine_eye)

    filename = write_imgs_to_path(
        data_img, file_path=tmp_path, create_files=True
    )
    check_niimg_3d(filename)

    # check data dtype equal with dtype='auto'
    img_check = check_niimg_3d(img_3d_zeros_eye, dtype="auto")
    assert (
        get_data(img_3d_zeros_eye).dtype.kind == get_data(img_check).dtype.kind
    )


def test_check_niimg_4d_errors(affine_eye, img_3d_zeros_eye, shape_3d_default):
    with pytest.raises(TypeError, match="nibabel format"):
        check_niimg_4d(0)

    with pytest.raises(TypeError, match="empty object"):
        check_niimg_4d([])

    # This should raise an error: a 3D img is given and we want a 4D
    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided a 3D image.",
    ):
        check_niimg_4d(img_3d_zeros_eye)

    a = img_3d_zeros_eye
    b = np.zeros(shape_3d_default)
    c = check_niimg_4d([a, b], return_iterator=True)
    with pytest.raises(
        TypeError, match="Error encountered while loading image #1"
    ):
        list(c)

    b = Nifti1Image(np.zeros((10, 20, 10)), affine_eye)
    c = check_niimg_4d([a, b], return_iterator=True)
    with pytest.raises(
        ValueError,
        match="Field of view of image #1 is different from reference FOV",
    ):
        list(c)


def test_check_niimg_4d(affine_eye, img_3d_zeros_eye, shape_3d_default):
    # Tests with return_iterator=False
    img_4d_1 = check_niimg_4d([img_3d_zeros_eye, img_3d_zeros_eye])
    assert get_data(img_4d_1).shape == (*shape_3d_default, 2)
    assert_array_equal(img_4d_1.affine, affine_eye)

    img_4d_2 = check_niimg_4d(img_4d_1)
    assert_array_equal(get_data(img_4d_2), get_data(img_4d_2))
    assert_array_equal(img_4d_2.affine, img_4d_2.affine)

    # Tests with return_iterator=True
    img_3d_iterator = check_niimg_4d(
        [img_3d_zeros_eye, img_3d_zeros_eye], return_iterator=True
    )
    img_3d_iterator_length = sum(1 for _ in img_3d_iterator)
    assert img_3d_iterator_length == 2

    img_3d_iterator_1 = check_niimg_4d(
        [img_3d_zeros_eye, img_3d_zeros_eye], return_iterator=True
    )
    img_3d_iterator_2 = check_niimg_4d(img_3d_iterator_1, return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert get_data(img_1).shape == shape_3d_default
        assert_array_equal(get_data(img_1), get_data(img_2))
        assert_array_equal(img_1.affine, img_2.affine)

    img_3d_iterator_1 = check_niimg_4d(
        [img_3d_zeros_eye, img_3d_zeros_eye], return_iterator=True
    )
    img_3d_iterator_2 = check_niimg_4d(img_4d_1, return_iterator=True)
    for img_1, img_2 in zip(img_3d_iterator_1, img_3d_iterator_2):
        assert get_data(img_1).shape == shape_3d_default
        assert_array_equal(get_data(img_1), get_data(img_2))
        assert_array_equal(img_1.affine, img_2.affine)

    # Test a Niimg-like object that does not hold a shape attribute
    phony_img = PhonyNiimage()
    check_niimg_4d(phony_img)


def test_check_niimg(img_3d_zeros_eye, img_4d_zeros_eye):
    img_3_3d = [[[img_3d_zeros_eye, img_3d_zeros_eye]]]
    img_2_4d = [[img_4d_zeros_eye, img_4d_zeros_eye]]

    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 2D and you provided "
        "a list of list of list of 3D images \\(6D\\)",
    ):
        check_niimg(img_3_3d, ensure_ndim=2)

    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a list of list of 4D images \\(6D\\)",
    ):
        check_niimg(img_2_4d, ensure_ndim=4)

    # check data dtype equal with dtype='auto'
    img_3d_check = check_niimg(img_3d_zeros_eye, dtype="auto")
    assert (
        get_data(img_3d_zeros_eye).dtype.kind
        == get_data(img_3d_check).dtype.kind
    )

    img_4d_check = check_niimg(img_4d_zeros_eye, dtype="auto")
    assert (
        get_data(img_4d_zeros_eye).dtype.kind
        == get_data(img_4d_check).dtype.kind
    )


def test_check_niimg_pathlike(img_3d_zeros_eye, tmp_path):
    filename = write_imgs_to_path(
        img_3d_zeros_eye, file_path=tmp_path, create_files=True
    )
    filename = Path(filename)
    check_niimg_3d(filename)


def test_check_niimg_wildcards_errors():
    # Check bad filename
    # Non existing file (with no magic) raise a ValueError exception
    nofile_path = "/tmp/nofile"
    file_not_found_msg = "File not found: '%s'"
    with pytest.raises(ValueError, match=file_not_found_msg % nofile_path):
        check_niimg(nofile_path)

    # Non matching wildcard raises a ValueError exception
    nofile_path_wildcards = "/tmp/no*file"
    with pytest.raises(
        ValueError, match="You may have left wildcards usage activated"
    ):
        check_niimg(nofile_path_wildcards)


@pytest.mark.parametrize("shape", [(10, 10, 10), (10, 10, 10, 3)])
@pytest.mark.parametrize(
    "wildcards", [True, False]
)  # (With globbing behavior or not)
def test_check_niimg_wildcards(affine_eye, shape, wildcards, tmp_path):
    # First create some testing data
    img = Nifti1Image(np.zeros(shape), affine_eye)

    filename = write_imgs_to_path(img, file_path=tmp_path, create_files=True)
    assert_array_equal(
        get_data(check_niimg(filename, wildcards=wildcards)),
        get_data(img),
    )


@pytest.fixture
def img_in_home_folder(img_3d_mni):
    """Create a test file in the home folder.

    Teardown: use yield instead of return to make sure the file
    is deleted after the test,
    even if the test fails.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization
    """
    created_file = Path("~/test.nii")
    img_3d_mni.to_filename(created_file.expanduser())
    assert created_file.expanduser().exists()

    yield img_3d_mni

    created_file.expanduser().unlink()


@pytest.mark.parametrize(
    "filename", ["~/test.nii", r"~/test.nii", Path("~/test.nii")]
)
def test_check_niimg_user_expand(img_in_home_folder, filename):
    """Check that user path are expanded."""
    found_file = check_niimg(filename)

    assert_array_equal(
        get_data(found_file),
        get_data(img_in_home_folder),
    )


@pytest.mark.parametrize(
    "filename",
    [
        "~/*.nii",
        r"~/*.nii",
        ["~/test.nii"],
        [r"~/test.nii"],
        [Path("~/test.nii")],
    ],
)
def test_check_niimg_user_expand_4d(img_in_home_folder, filename):
    """Check that user path are expanded.

    Wildcards and lists should expected 4D data to be returned.
    """
    found_file = check_niimg(filename)

    assert_array_equal(
        get_data(found_file),
        get_data(check_niimg(img_in_home_folder, atleast_4d=True)),
    )


def test_check_niimg_wildcards_one_file_name(img_3d_zeros_eye, tmp_path):
    file_not_found_msg = "File not found: '%s'"

    # Testing with a glob matching exactly one filename
    # Using a glob matching one file containing a 3d image returns a 4d image
    # with 1 as last dimension.
    globs = write_imgs_to_path(
        img_3d_zeros_eye,
        file_path=tmp_path,
        create_files=True,
        use_wildcards=True,
    )
    assert_array_equal(
        get_data(check_niimg(globs))[..., 0],
        get_data(img_3d_zeros_eye),
    )
    # Disabled globbing behavior should raise an ValueError exception
    with pytest.raises(
        ValueError, match=file_not_found_msg % re.escape(globs)
    ):
        check_niimg(globs, wildcards=False)

    # Testing with a glob matching multiple filenames
    img_4d = check_niimg_4d((img_3d_zeros_eye, img_3d_zeros_eye))
    globs = write_imgs_to_path(
        img_3d_zeros_eye,
        img_3d_zeros_eye,
        file_path=tmp_path,
        create_files=True,
        use_wildcards=True,
    )
    assert_array_equal(get_data(check_niimg(globs)), get_data(img_4d))


def test_check_niimg_wildcards_no_expand_wildcards(
    img_3d_zeros_eye, img_4d_zeros_eye, tmp_path
):
    nofile_path = "/tmp/nofile"

    file_not_found_msg = "File not found: '%s'"

    #######
    # Test when global variable is set to False => no globbing allowed
    ni.EXPAND_PATH_WILDCARDS = False

    # Non existing filename (/tmp/nofile) could match an existing one through
    # globbing but global wildcards variable overrides this feature => raises
    # a ValueError
    with pytest.raises(ValueError, match=file_not_found_msg % nofile_path):
        check_niimg(nofile_path)

    # Verify wildcards function parameter has no effect
    with pytest.raises(ValueError, match=file_not_found_msg % nofile_path):
        check_niimg(nofile_path, wildcards=False)

    # Testing with an exact filename matching (3d case)
    filename = write_imgs_to_path(
        img_3d_zeros_eye, file_path=tmp_path, create_files=True
    )
    assert_array_equal(
        get_data(check_niimg(filename)), get_data(img_3d_zeros_eye)
    )

    # Testing with an exact filename matching (4d case)
    filename = write_imgs_to_path(
        img_4d_zeros_eye, file_path=tmp_path, create_files=True
    )
    assert_array_equal(
        get_data(check_niimg(filename)), get_data(img_4d_zeros_eye)
    )

    # Reverting to default behavior
    ni.EXPAND_PATH_WILDCARDS = True


def test_iter_check_niimgs_error():
    no_file_matching = "No files matching path: %s"

    for empty in ((), [], iter(())):
        with pytest.raises(ValueError, match="Input niimgs list is empty."):
            list(iter_check_niimg(empty))

    nofile_path = "/tmp/nofile"
    with pytest.raises(ValueError, match=no_file_matching % nofile_path):
        list(iter_check_niimg(nofile_path))


def test_iter_check_niimgs(tmp_path, img_4d_zeros_eye):
    img_2_4d = [[img_4d_zeros_eye, img_4d_zeros_eye]]

    # Create a test file
    filename = tmp_path / "nilearn_test.nii"
    img_4d_zeros_eye.to_filename(filename)
    niimgs = list(iter_check_niimg([filename]))
    assert_array_equal(
        get_data(niimgs[0]), get_data(check_niimg(img_4d_zeros_eye))
    )
    del niimgs

    # Regular case
    niimgs = list(iter_check_niimg(img_2_4d))
    assert_array_equal(get_data(niimgs[0]), get_data(check_niimg(img_2_4d)))


def _check_memory(list_img_3d):
    # We intentionally add an offset of memory usage to avoid non trustable
    # measures with memory_profiler.
    mem_offset = b"a" * 100 * 1024**2
    list(iter_check_niimg(list_img_3d))
    return mem_offset


@with_memory_profiler
def test_iter_check_niimgs_memory(affine_eye):
    # Verify that iterating over a list of images doesn't consume extra
    # memory.
    assert_memory_less_than(
        100,
        0.1,
        _check_memory,
        [Nifti1Image(np.ones((100, 100, 200)), affine_eye) for _ in range(10)],
    )


def test_repr_niimgs():
    # Tests with file path
    assert repr_niimgs("test") == "test"
    assert repr_niimgs("test", shorten=False) == "test"

    # Shortening long names by default
    long_name = "this-is-a-very-long-name-for-a-nifti-file.nii"
    short_name = "this-is-a-very-lon..."
    assert repr_niimgs(long_name) == short_name
    # Explicit shortening of long names
    assert repr_niimgs(long_name, shorten=True) == short_name

    # Lists of long names up to length 3
    list_of_size_3 = [
        "this-is-a-very-long-name-for-a-nifti-file.nii",
        "this-is-another-very-long-name-for-a-nifti-file.nii",
        "this-is-again-another-very-long-name-for-a-nifti-file.nii",
    ]
    # Explicit shortening, all 3 names are displayed, but shortened
    shortened_rep_list_of_size_3 = (
        "[this-is-a-very-lon...,"
        " this-is-another-ve...,"
        " this-is-again-anot...]"
    )

    assert (
        repr_niimgs(list_of_size_3, shorten=True)
        == shortened_rep_list_of_size_3
    )

    # Lists longer than 3
    # Small names - Explicit shortening
    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    shortened_rep_long_list_small_names = "[test,\n         ...\n rereretest]"

    assert (
        repr_niimgs(long_list_small_names, shorten=True)
        == shortened_rep_long_list_small_names
    )

    # Long names - Explicit shortening
    list_of_size_4 = [
        *list_of_size_3,
        "this-is-again-another-super-very-long-name-for-a-nifti-file.nii",
    ]
    shortened_rep_long_list_long_names = (
        "[this-is-a-very-lon...,\n         ...\n this-is-again-anot...]"
    )

    assert (
        repr_niimgs(list_of_size_4, shorten=True)
        == shortened_rep_long_list_long_names
    )


def test_repr_niimgs_force_long_names():
    long_name = "this-is-a-very-long-name-for-a-nifti-file.nii"
    # Force long display of long names
    assert repr_niimgs(long_name, shorten=False) == long_name

    # Tests with list of file paths
    assert repr_niimgs(["test", "retest"]) == "[test, retest]"
    assert repr_niimgs(["test", "retest"], shorten=False) == "[test, retest]"

    # Force display, all 3 names are displayed
    list_of_size_3 = [
        "this-is-a-very-long-name-for-a-nifti-file.nii",
        "this-is-another-very-long-name-for-a-nifti-file.nii",
        "this-is-again-another-very-long-name-for-a-nifti-file.nii",
    ]
    long_rep_list_of_size_3 = (
        "[this-is-a-very-long-name-for-a-nifti-file.nii,"
        " this-is-another-very-long-name-for-a-nifti-file.nii,"
        " this-is-again-another-very-long-name-for-a-nifti-file.nii]"
    )
    assert (
        repr_niimgs(list_of_size_3, shorten=False) == long_rep_list_of_size_3
    )

    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    long_rep_long_list_small_names = (
        "[test,\n retest,\n reretest,\n rereretest]"
    )

    assert (
        repr_niimgs(long_list_small_names, shorten=False)
        == long_rep_long_list_small_names
    )

    # Long names - Force full display in pretty print style for readability
    list_of_size_4 = [
        *list_of_size_3,
        "this-is-again-another-super-very-long-name-for-a-nifti-file.nii",
    ]
    long_rep_long_list_long_names = (
        long_rep_list_of_size_3[:-1].replace(",", ",\n")
        + ",\n "
        + "this-is-again-another-super-very-long-name-for-a-nifti-file.nii]"
    )

    assert (
        repr_niimgs(list_of_size_4, shorten=False)
        == long_rep_long_list_long_names
    )


def test_repr_niimgs_with_niimg_pathlib():
    # Tests with pathlib
    # Case with very long path and small filename
    long_path = Path("/this/is/a/fake/long/path/to/file.nii")
    short_path = Path(".../path/to/file.nii")
    assert repr_niimgs(long_path, shorten=True) == str(short_path)
    assert repr_niimgs(long_path, shorten=False) == str(long_path)

    # Case with very long path but very long filename
    long_path_long_name = Path(
        "/this/is/a/fake/long/path/to/my_file_with_a_very_long_name.nii"
    )
    short_name = "my_file_with_a_ver..."
    assert repr_niimgs(long_path_long_name, shorten=True) == short_name
    assert repr_niimgs(long_path_long_name, shorten=False) == str(
        long_path_long_name
    )

    # Case with lists
    list_of_paths = [
        Path("/this/is/a/fake/long/path/to/file.nii"),
        Path("/this/is/a/fake/long/path/to/another/file2.nii"),
        Path("/again/another/fake/long/path/to/file3.nii"),
        Path("/this/is/a/fake/long/path/to/a-very-long-file-name.nii"),
    ]

    shortened_list_of_paths = (
        f"[...{Path('/path/to/file.nii')!s},\n"
        f"         ...\n"
        f" a-very-long-file-n...]"
    )

    assert repr_niimgs(list_of_paths, shorten=True) == shortened_list_of_paths
    long_list_of_paths = ",\n ".join([str(_) for _ in list_of_paths])
    long_list_of_paths = f"[{long_list_of_paths}]"
    assert repr_niimgs(list_of_paths, shorten=False) == long_list_of_paths


@pytest.mark.parametrize("shorten", [True, False])
def test_repr_niimgs_with_niimg(
    shorten, tmp_path, affine_eye, img_3d_ones_eye, shape_3d_default
):
    # Shorten has no effect in this case
    assert repr_niimgs(img_3d_ones_eye, shorten=shorten).replace(
        "10L", "10"
    ) == (
        f"{img_3d_ones_eye.__class__.__name__}(\nshape={shape_3d_default!r},\naffine={affine_eye!r}\n)"
    )

    # Add filename long enough to qualify for shortening
    fd, tmpimg1 = mkstemp(suffix="_very_long.nii", dir=str(tmp_path))
    os.close(fd)
    img_3d_ones_eye.to_filename(tmpimg1)
    class_name = img_3d_ones_eye.__class__.__name__
    filename = Path(img_3d_ones_eye.get_filename())
    assert (
        repr_niimgs(img_3d_ones_eye, shorten=False)
        == f"{class_name}('{filename}')"
    )
    assert (
        repr_niimgs(img_3d_ones_eye, shorten=True)
        == f"{class_name}('{Path(filename).name[:18]}...')"
    )
