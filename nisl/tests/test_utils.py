"""
Test the utils module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD


import os
import os.path as osp
import tempfile

import nose
from nose.tools import assert_raises, assert_equal, assert_true

import numpy as np

import nibabel
from nibabel import Nifti1Image

from .. import utils


class PhonyNiimage:

    def __init__(self):
        self.data = np.ones((9, 9, 9, 9))
        self.affine = np.ones((4, 4))

    def get_data(self):
        return self.data

    def get_affine(self):
        return self.affine


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    yield assert_raises, ValueError, utils.largest_connected_component, a
    a[1:3, 1:3, 1:3] = 1
    yield np.testing.assert_equal, a, utils.largest_connected_component(a)
    b = a.copy()
    b[5, 5, 5] = 1
    yield np.testing.assert_equal, a, utils.largest_connected_component(b)


def test_check_niimg():
    assert_raises(TypeError, utils.check_niimg, 0)


def test_check_niimgs():
    yield assert_raises, TypeError, utils.check_niimgs, 0
    affine = np.eye(4)
    niimg = Nifti1Image(np.ones((10, 10, 10)), affine)

    utils.check_niimgs([niimg, niimg])
    # This should raise an error: a 3D niimg is given and we want a 4D
    yield assert_raises, TypeError, utils.check_niimgs, niimg
    # This shouldn't raise an error
    utils.check_niimgs(niimg, accept_3d=True)

    # Test a Niimage that does not hold a shape attribute
    phony_niimg = PhonyNiimage()
    utils.check_niimgs(phony_niimg)


def test_repr_niimgs():
    # Test with file path
    yield assert_equal, utils._repr_niimgs("test"), "test"
    yield assert_equal, utils._repr_niimgs(["test", "retest"]), \
        "[test, retest]"
    # Create phony Niimg with filename
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones((10, 10, 10)), affine)
    yield assert_equal, utils._repr_niimgs(niimg1), \
        ("%s('%s')" % (niimg1.__class__.__name__, niimg1.get_filename()))
    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    nibabel.save(niimg1, tmpimg1)
    yield assert_equal, utils._repr_niimgs(niimg1), \
        ("%s('%s')" % (niimg1.__class__.__name__, niimg1.get_filename()))


def _remove_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


def test_concat_niimgs():
    shape = (10, 11, 12)
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones(shape), affine)
    niimg2 = Nifti1Image(np.ones(shape), 2 * affine)
    niimg3 = Nifti1Image(np.zeros(shape), affine)

    concatenated = utils.concat_niimgs((niimg1, niimg3, niimg1))
    concatenate_true = np.ones(shape + (3,))
    concatenate_true[..., 1] = 0
    np.testing.assert_almost_equal(concatenated.get_data(), concatenate_true)

    assert_raises(ValueError, utils.concat_niimgs, [niimg1, niimg2])

    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    _, tmpimg2 = tempfile.mkstemp(suffix='.nii')
    try:
        nibabel.save(niimg1, tmpimg1)
        nibabel.save(niimg2, tmpimg2)
        nose.tools.assert_raises(ValueError, utils.concat_niimgs,
                                 [tmpimg1, tmpimg2])
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)


def are_arrays_identical(arr1, arr2):
    """Check if two 1-dimensional array point to the same buffer.

    The check is performed only on the first value of the arrays. For
    this test to be reliable, arr2 must not point to a subset of arr1.
    For example, if arr2 = arr1[1:] has been executed just before calling
    this function, the test will FAIL, even if the same buffer is used by
    both arrays. arr2 = arr1[:1] will succeed though.

    dtypes are not supposed to be identical.
    """
    # Modify the first value in arr1 twice, and see if corresponding
    # value in arr2 has changed. Changing the value twice is required, since
    # the original value could be the first value that we use.

    orig1 = arr1[0]
    orig2 = arr2[0]

    arr1[0] = 0
    if arr2[0] != orig2:
        arr1[0] = orig1
        return True

    arr1[0] = 1
    if arr2[0] != orig2:
        arr1[0] = orig1
        return True

    arr1[0] = orig1
    return False


def test_are_array_identical():
    arr1 = np.ones(4)
    orig1 = arr1.copy()

    arr2 = arr1
    orig2 = arr2.copy()

    assert(are_arrays_identical(arr1, arr2))
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)

    arr2 = arr1[:1]
    orig2 = arr2.copy()
    assert(are_arrays_identical(arr1, arr2))
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)

    arr2 = arr1[1:]
    orig2 = arr2.copy()
    assert(not are_arrays_identical(arr1, arr2))
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)

    arr2 = arr1.copy()
    orig2 = arr2.copy()
    assert(not are_arrays_identical(arr1, arr2))
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)


def test_as_ndarray():
    # All test cases
    # input dtype, input order, should copy, output dtype, output order, copied
    test_cases = [
        # no-op
        (np.float, "C", False, None, None, False),
        (np.float, "F", False, None, None, False),

        # simple copy
        (np.float, "C", True, None, None, True),
        (np.float, "F", True, None, None, True),

        # dtype provided, identical
        (np.float, "C", False, np.float, None, False),
        (np.float, "F", False, np.float, None, False),

        # dtype changed
        (np.float, "C", False, np.float32, None, True),
        (np.float, "F", False, np.float32, None, True),

        # dtype and order provided, but identical
        (np.float, "C", False, np.float, "C", False),
        (np.float, "F", False, np.float, "F", False),

        # order provided, unchanged
        (np.float, "C", False, None, "C", False),
        (np.float, "F", False, None, "F", False),
        (np.float, "C", True, None, "C", True),
        (np.float, "F", True, None, "F", True),

        # order provided, changed
        (np.float, "C", False, None, "F", True),
        (np.float, "F", False, None, "C", True),
        (np.float, "C", True, None, "F", True),
        (np.float, "F", True, None, "C", True),

        # Special case for int8 <-> bool conversion.
        (np.int8, "C", False, np.bool, None, False),
        (np.int8, "F", False, np.bool, None, False),
        (np.int8, "C", False, np.bool, "C", False),
        (np.int8, "F", False, np.bool, "F", False),
        (np.int8, "C", False, np.bool, "F", True),
        (np.int8, "F", False, np.bool, "C", True),

        (np.int8, "C", True, np.bool, None, True),
        (np.int8, "F", True, np.bool, None, True),
        (np.int8, "C", True, np.bool, "C", True),
        (np.int8, "F", True, np.bool, "F", True),

        (np.bool, "C", False, np.int8, None, False),
        (np.bool, "F", False, np.int8, None, False),
        (np.bool, "C", False, np.int8, "C", False),
        (np.bool, "F", False, np.int8, "F", False),
        (np.bool, "C", False, np.int8, "F", True),
        (np.bool, "F", False, np.int8, "C", True),

        (np.bool, "C", True, np.int8, None, True),
        (np.bool, "F", True, np.int8, None, True),
        (np.bool, "C", True, np.int8, "C", True),
        (np.bool, "F", True, np.int8, "F", True),
        ]

    shape = (10, 11)
    for case in test_cases:
        in_dtype, in_order, copy, out_dtype, out_order, copied = case
        arr1 = np.ones(shape, dtype=in_dtype, order=in_order)
        arr2 = utils.as_ndarray(arr1,
                                copy=copy, dtype=out_dtype, order=out_order)
        assert_true(not are_arrays_identical(arr1[0], arr2[0]) == copied,
                    msg=str(case))
        if out_dtype is None:
            assert_true(arr2.dtype == in_dtype, msg=str(case))
        else:
            assert_true(arr2.dtype == out_dtype, msg=str(case))

        result_order = out_order if out_order is not None else in_order
        if result_order == "F":
            assert_true(arr2.flags["F_CONTIGUOUS"], msg=str(case))
        else:
            assert_true(arr2.flags["C_CONTIGUOUS"], msg=str(case))

    ## memmap
    filename = osp.join(osp.dirname(__file__), "data", "mmap.dat")

    # same dtype, no copy requested
    arr1 = np.memmap(filename, dtype='float32', mode='w+', shape=(5,))
    arr2 = utils.as_ndarray(arr1)
    assert(not are_arrays_identical(arr1, arr2))

    # same dtype, copy requested
    arr1 = np.memmap(filename, dtype='float32', mode='w+', shape=(5,))
    arr2 = utils.as_ndarray(arr1, copy=True)
    assert(not are_arrays_identical(arr1, arr2))

    # different dtype
    arr1 = np.memmap(filename, dtype='float32', mode='w+', shape=(5,))
    arr2 = utils.as_ndarray(arr1, dtype=np.int)
    assert(arr2.dtype == np.int)
    assert(not are_arrays_identical(arr1, arr2))

    # same dtype, explicitly provided: must copy
    arr1 = np.memmap(filename, dtype='float32', mode='w+', shape=(5,))
    arr2 = utils.as_ndarray(arr1, dtype=np.float32)
    assert(arr2.dtype == np.float32)
    assert(not are_arrays_identical(arr1, arr2))

    # same dtype, order provided
    arr1 = np.memmap(filename, dtype='float32', mode='w+', shape=(10, 10))
    arr2 = utils.as_ndarray(arr1, order="F")
    assert(arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"])
    assert(arr2.dtype == arr1.dtype)
    assert(not are_arrays_identical(arr1[0], arr2[0]))

    # same dtype, order unchanged but provided
    arr1 = np.memmap(filename, dtype='float32', mode='w+',
                     shape=(10, 10), order="F")
    arr2 = utils.as_ndarray(arr1, order="F")
    assert(arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"])
    assert(arr2.dtype == arr1.dtype)
    assert(not are_arrays_identical(arr1[0], arr2[0]))

    # dtype and order specified
    arr1 = np.memmap(filename, dtype='float32', mode='w+',
                     shape=(10, 10), order="F")
    arr2 = utils.as_ndarray(arr1, order="F", dtype=np.int32)
    assert(arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"])
    assert(arr2.dtype == np.int32)
    assert(not are_arrays_identical(arr1[0], arr2[0]))

    ## list
    # same dtype, no copy requested
    arr1 = [0, 1, 2, 3]
    arr2 = utils.as_ndarray(arr1)
    assert(not are_arrays_identical(arr1, arr2))

    # same dtype, copy requested
    arr1 = [0, 1, 2, 3]
    arr2 = utils.as_ndarray(arr1, copy=True)
    assert(not are_arrays_identical(arr1, arr2))

    # different dtype
    arr1 = [0, 1, 2, 3]
    arr2 = utils.as_ndarray(arr1, dtype=np.float)
    assert(arr2.dtype == np.float)
    assert(not are_arrays_identical(arr1, arr2))

    # order specified
    arr1 = [[0, 1, 2, 3], [0, 1, 2, 3]]
    arr2 = utils.as_ndarray(arr1, dtype=np.float, order="F")
    assert(arr2.dtype == np.float)
    assert(arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"])
    assert(not are_arrays_identical(arr1[0], arr2[0]))

    ## Unhandled cases
    assert_raises(ValueError, utils.as_ndarray, "test string")
    assert_raises(ValueError, utils.as_ndarray, [], order="invalid")
