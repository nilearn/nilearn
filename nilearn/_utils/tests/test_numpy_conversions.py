"""Test the numpy_conversions module.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

from pathlib import Path

import numpy as np
import pytest

from nilearn._utils.numpy_conversions import as_ndarray, csv_to_array


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

    assert are_arrays_identical(arr1, arr2)
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)

    arr2 = arr1[:1]
    orig2 = arr2.copy()
    assert are_arrays_identical(arr1, arr2)
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)

    arr2 = arr1[1:]
    orig2 = arr2.copy()
    assert not are_arrays_identical(arr1, arr2)
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)

    arr2 = arr1.copy()
    orig2 = arr2.copy()
    assert not are_arrays_identical(arr1, arr2)
    np.testing.assert_array_almost_equal(orig1, arr1, decimal=10)
    np.testing.assert_array_almost_equal(orig2, arr2, decimal=10)


@pytest.mark.parametrize(
    "input_dtype, input_order, copy, output_dtype, output_order, was_copied",
    [
        # no-op
        (float, "C", False, None, None, False),
        (float, "F", False, None, None, False),
        # simple copy
        (float, "C", True, None, None, True),
        (float, "F", True, None, None, True),
        # dtype provided, identical
        (float, "C", False, float, None, False),
        (float, "F", False, float, None, False),
        # dtype changed
        (float, "C", False, np.float32, None, True),
        (float, "F", False, np.float32, None, True),
        # dtype and order provided, but identical
        (float, "C", False, float, "C", False),
        (float, "F", False, float, "F", False),
        # order provided, unchanged
        (float, "C", False, None, "C", False),
        (float, "F", False, None, "F", False),
        (float, "C", True, None, "C", True),
        (float, "F", True, None, "F", True),
        # order provided, changed
        (float, "C", False, None, "F", True),
        (float, "F", False, None, "C", True),
        (float, "C", True, None, "F", True),
        (float, "F", True, None, "C", True),
        # Special case for int8 <-> bool conversion.
        (np.int8, "C", False, bool, None, False),
        (np.int8, "F", False, bool, None, False),
        (np.int8, "C", False, bool, "C", False),
        (np.int8, "F", False, bool, "F", False),
        (np.int8, "C", False, bool, "F", True),
        (np.int8, "F", False, bool, "C", True),
        (np.int8, "C", True, bool, None, True),
        (np.int8, "F", True, bool, None, True),
        (np.int8, "C", True, bool, "C", True),
        (np.int8, "F", True, bool, "F", True),
        (bool, "C", False, np.int8, None, False),
        (bool, "F", False, np.int8, None, False),
        (bool, "C", False, np.int8, "C", False),
        (bool, "F", False, np.int8, "F", False),
        (bool, "C", False, np.int8, "F", True),
        (bool, "F", False, np.int8, "C", True),
        (bool, "C", True, np.int8, None, True),
        (bool, "F", True, np.int8, None, True),
        (bool, "C", True, np.int8, "C", True),
        (bool, "F", True, np.int8, "F", True),
    ],
)
def test_as_ndarray(
    input_dtype, input_order, copy, output_dtype, output_order, was_copied
):
    shape = (10, 11)
    arr1 = np.ones(shape, dtype=input_dtype, order=input_order)
    arr2 = as_ndarray(arr1, copy=copy, dtype=output_dtype, order=output_order)

    # check if nd_array copied the original array or not
    assert are_arrays_identical(arr1[0], arr2[0]) != was_copied

    if output_dtype is None:
        assert arr2.dtype == input_dtype
    else:
        assert arr2.dtype == output_dtype

    result_order = output_order if output_order is not None else input_order

    if result_order == "F":
        assert arr2.flags["F_CONTIGUOUS"]
    else:
        assert arr2.flags["C_CONTIGUOUS"]


def test_as_ndarray_memmap():
    # memmap
    filename = Path(__file__).parent / "data" / "mmap.dat"

    # same dtype, no copy requested
    arr1 = np.memmap(filename, dtype="float32", mode="w+", shape=(5,))
    arr2 = as_ndarray(arr1)
    assert not are_arrays_identical(arr1, arr2)

    # same dtype, copy requested
    arr1 = np.memmap(filename, dtype="float32", mode="readwrite", shape=(5,))
    arr2 = as_ndarray(arr1, copy=True)
    assert not are_arrays_identical(arr1, arr2)

    # different dtype
    arr1 = np.memmap(filename, dtype="float32", mode="readwrite", shape=(5,))
    arr2 = as_ndarray(arr1, dtype=int)
    assert arr2.dtype == int
    assert not are_arrays_identical(arr1, arr2)

    # same dtype, explicitly provided: must copy
    arr1 = np.memmap(filename, dtype="float32", mode="readwrite", shape=(5,))
    arr2 = as_ndarray(arr1, dtype=np.float32)
    assert arr2.dtype == np.float32
    assert not are_arrays_identical(arr1, arr2)

    # same dtype, order provided
    arr1 = np.memmap(
        filename, dtype="float32", mode="readwrite", shape=(10, 10)
    )
    arr2 = as_ndarray(arr1, order="F")
    assert arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"]
    assert arr2.dtype == arr1.dtype
    assert not are_arrays_identical(arr1[0], arr2[0])

    # same dtype, order unchanged but provided
    arr1 = np.memmap(
        filename, dtype="float32", mode="readwrite", shape=(10, 10), order="F"
    )
    arr2 = as_ndarray(arr1, order="F")
    assert arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"]
    assert arr2.dtype == arr1.dtype
    assert not are_arrays_identical(arr1[0], arr2[0])

    # dtype and order specified
    arr1 = np.memmap(
        filename, dtype="float32", mode="readwrite", shape=(10, 10), order="F"
    )
    arr2 = as_ndarray(arr1, order="F", dtype=np.int32)
    assert arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"]
    assert arr2.dtype == np.int32
    assert not are_arrays_identical(arr1[0], arr2[0])


def test_as_ndarray_more():
    # list
    # same dtype, no copy requested
    arr1 = [0, 1, 2, 3]
    arr2 = as_ndarray(arr1)
    assert not are_arrays_identical(arr1, arr2)

    # same dtype, copy requested
    arr1 = [0, 1, 2, 3]
    arr2 = as_ndarray(arr1, copy=True)
    assert not are_arrays_identical(arr1, arr2)

    # different dtype
    arr1 = [0, 1, 2, 3]
    arr2 = as_ndarray(arr1, dtype=float)
    assert arr2.dtype == float
    assert not are_arrays_identical(arr1, arr2)

    # order specified
    arr1 = [[0, 1, 2, 3], [0, 1, 2, 3]]
    arr2 = as_ndarray(arr1, dtype=float, order="F")
    assert arr2.dtype == float
    assert arr2.flags["F_CONTIGUOUS"] and not arr2.flags["C_CONTIGUOUS"]
    assert not are_arrays_identical(arr1[0], arr2[0])

    # Unhandled cases
    with pytest.raises(ValueError):
        as_ndarray("test_string")
    with pytest.raises(ValueError):
        as_ndarray([], order="invalid")


def test_csv_to_array(tmp_path):
    # Create a phony CSV file
    filename = tmp_path / "tmp.csv"
    with filename.open(mode="w") as fp:
        fp.write("1.,2.,3.,4.,5.\n")
    assert np.allclose(
        csv_to_array(filename), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    with pytest.raises(TypeError):
        csv_to_array(filename, delimiters="?!")
