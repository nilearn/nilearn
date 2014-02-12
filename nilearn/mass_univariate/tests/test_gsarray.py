"""
Tests for the GrowableSparseArray class used in permuted_ols function.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import warnings
import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from nilearn.mass_univariate.permuted_least_squares import GrowableSparseArray


def test_gsarray_append_data():
    """This function tests GrowableSparseArray creation and filling.

    It is especially important to check that the threshold is respected
    and that the structure is robust to threshold choice.

    """
    # Simplest example
    gsarray = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    assert_array_equal(gsarray.get_data()['iter_id'], np.zeros(5))
    assert_array_equal(gsarray.get_data()['x_id'], np.zeros(5))
    assert_array_equal(gsarray.get_data()['y_id'], np.arange(5))
    assert_array_equal(gsarray.get_data()['score'], np.ones(5))

    # Void array
    gsarray = GrowableSparseArray(n_iter=1, threshold=10)
    gsarray.append(0, np.ones((5, 1)))
    assert_array_equal(gsarray.get_data()['iter_id'], [])
    assert_array_equal(gsarray.get_data()['x_id'], [])
    assert_array_equal(gsarray.get_data()['y_id'], [])
    assert_array_equal(gsarray.get_data()['score'], [])

    # Toy example
    gsarray = GrowableSparseArray(n_iter=10, threshold=8)
    for i in range(10):
        gsarray.append(i, (np.arange(10) - i).reshape((-1, 1)))
    assert_array_equal(gsarray.get_data()['iter_id'], np.array([0., 0., 1.]))
    assert_array_equal(gsarray.get_data()['x_id'], np.zeros(3))
    assert_array_equal(gsarray.get_data()['y_id'], [8, 9, 9])
    assert_array_equal(gsarray.get_data()['score'], [8., 9., 8.])


def test_gsarray_merge():
    """This function tests GrowableSparseArray merging.

    Because of the specific usage of GrowableSparseArrays, only a reduced
    number of manipulations has been implemented.

    """
    # Basic merge
    gsarray = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray2.merge(gsarray)
    assert_array_equal(
        gsarray.get_data()['iter_id'], gsarray2.get_data()['iter_id'])
    assert_array_equal(
        gsarray.get_data()['x_id'], gsarray2.get_data()['x_id'])
    assert_array_equal(
        gsarray.get_data()['y_id'], gsarray2.get_data()['y_id'])
    assert_array_equal(
        gsarray.get_data()['score'], gsarray2.get_data()['score'])

    # Merge list
    gsarray = GrowableSparseArray(n_iter=2, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=2, threshold=0)
    gsarray2.append(1, 2 * np.ones((5, 1)), y_offset=5)
    gsarray3 = GrowableSparseArray(n_iter=2, threshold=0)
    gsarray3.merge([gsarray, gsarray2])
    assert_array_equal(gsarray3.get_data()['iter_id'],
                       np.array([0.] * 5 + [1.] * 5))
    assert_array_equal(gsarray3.get_data()['x_id'], np.zeros(10))
    assert_array_equal(gsarray3.get_data()['y_id'], np.arange(10))
    assert_array_equal(gsarray3.get_data()['score'],
                       np.array([1.] * 5 + [2.] * 5))

    # Test failure case (merging arrays with different n_iter)
    gsarray_wrong = GrowableSparseArray(n_iter=1)
    gsarray_wrong.append(0, np.ones((5, 1)))
    gsarray = GrowableSparseArray(n_iter=2)
    assert_raises(Exception, gsarray.merge, gsarray_wrong)

    # Test failure case (merge a numpy array)
    gsarray = GrowableSparseArray(n_iter=1)
    assert_raises(Exception, gsarray.merge, np.ones(5))

    # Check the threshold is respected when merging
    # merging a gsarray into another one that has a higher threhold
    # (nothing should be left in the parent array)
    gsarray = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=1, threshold=2)  # higher threshold
    gsarray2.merge(gsarray)
    assert_array_equal(gsarray2.get_data()['score'], [])

    # merging a gsarray into another one that has a higher threhold
    # (should raises a warning on potential information loss)
    gsarray = GrowableSparseArray(n_iter=1, threshold=1)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=1, threshold=0)  # lower threshold
    with warnings.catch_warnings(True) as warning:
        gsarray2.merge(gsarray)
        assert(len(warning) >= 1)
        assert(isinstance(warning[0], warnings.WarningMessage))
    assert_array_equal(
        gsarray.get_data()['iter_id'], gsarray2.get_data()['iter_id'])
    assert_array_equal(
        gsarray.get_data()['x_id'], gsarray2.get_data()['x_id'])
    assert_array_equal(
        gsarray.get_data()['y_id'], gsarray2.get_data()['y_id'])
    assert_array_equal(
        gsarray.get_data()['score'], gsarray2.get_data()['score'])
