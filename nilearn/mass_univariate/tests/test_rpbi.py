"""


"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
from numpy.testing import assert_array_equal, assert_raises, assert_warns
from nilearn.mass_univariate.rpbi import GrowableSparseArray


### Tests for the GrowableSparseArray class ###################################
def test_gsarray_append_data():
    """This function tests GrowableSparseArray creation and filling.

    It is especially important to check that the threshold is respected
    and that the structure is robust to threshold choice.

    """
    # Simplest example
    gs_array = GrowableSparseArray(n_iter=1, threshold=0)
    gs_array.append(0, np.ones((5, 1)))
    assert_array_equal(gs_array.get_data()['iter_id'], np.zeros(5))
    assert_array_equal(gs_array.get_data()['x_id'], np.zeros(5))
    assert_array_equal(gs_array.get_data()['y_id'], np.arange(5))
    assert_array_equal(gs_array.get_data()['score'], np.ones(5))

    # Void array
    gs_array = GrowableSparseArray(n_iter=1, threshold=10)
    gs_array.append(0, np.ones((5, 1)))
    assert_array_equal(gs_array.get_data()['iter_id'], [])
    assert_array_equal(gs_array.get_data()['x_id'], [])
    assert_array_equal(gs_array.get_data()['y_id'], [])
    assert_array_equal(gs_array.get_data()['score'], [])

    # Toy example
    gs_array = GrowableSparseArray(n_iter=10, threshold=8)
    for i in range(10):
        gs_array.append(i, (np.arange(10) - i).reshape((-1, 1)))
    assert_array_equal(gs_array.get_data()['iter_id'], np.array([0., 0., 1.]))
    assert_array_equal(gs_array.get_data()['x_id'], np.zeros(3))
    assert_array_equal(gs_array.get_data()['y_id'], [8, 9, 9])
    assert_array_equal(gs_array.get_data()['score'], [8., 9., 8.])


def test_gsarray_merge():
    """This function tests GrowableSparseArray merging.

    Because of the specific usage of GrowableSparseArrays, only a reduced
    number of manipulations has been implemented.

    """
    # Basic merge
    gs_array = GrowableSparseArray(n_iter=1, threshold=0)
    gs_array.append(0, np.ones((5, 1)))
    gs_array2 = GrowableSparseArray(n_iter=1, threshold=0)
    gs_array2.merge(gs_array)
    assert_array_equal(gs_array.get_data()['iter_id'],
                       gs_array2.get_data()['iter_id'])
    assert_array_equal(gs_array.get_data()['x_id'],
                       gs_array2.get_data()['x_id'])
    assert_array_equal(gs_array.get_data()['y_id'],
                       gs_array2.get_data()['y_id'])
    assert_array_equal(gs_array.get_data()['score'],
                       gs_array2.get_data()['score'])

    # Merge list
    gs_array = GrowableSparseArray(n_iter=2, threshold=0)
    gs_array.append(0, np.ones((5, 1)))
    gs_array2 = GrowableSparseArray(n_iter=2, threshold=0)
    gs_array2.append(1, 2 * np.ones((5, 1)), y_offset=5)
    gs_array3 = GrowableSparseArray(n_iter=2, threshold=0)
    gs_array3.merge([gs_array, gs_array2])
    assert_array_equal(gs_array3.get_data()['iter_id'],
                       np.array([0.] * 5 + [1.] * 5))
    assert_array_equal(gs_array3.get_data()['x_id'], np.zeros(10))
    assert_array_equal(gs_array3.get_data()['y_id'], np.arange(10))
    assert_array_equal(gs_array3.get_data()['score'],
                       np.array([1.] * 5 + [2.] * 5))

    # Test failure case (merging arrays with different n_iter)
    gs_array_wrong = GrowableSparseArray(n_iter=2)
    gs_array_wrong.append(0, np.ones((5, 1)))
    gs_array_wrong.append(1, np.ones((5, 1)))
    gs_array = GrowableSparseArray(n_iter=1)
    assert_raises(ValueError, gs_array.merge, gs_array_wrong)

    # Test failure case (merge a numpy array)
    gs_array = GrowableSparseArray(n_iter=1)
    assert_raises(TypeError, gs_array.merge, np.ones(5))

    # Check the threshold is respected when merging
    # merging a gsarray into another one that has a higher threhold
    # (nothing should be left in the parent array)
    gs_array = GrowableSparseArray(n_iter=1, threshold=0)
    gs_array.append(0, np.ones((5, 1)))
    gs_array2 = GrowableSparseArray(n_iter=1, threshold=2)  # higher threshold
    gs_array2.merge(gs_array)
    assert_array_equal(gs_array2.get_data()['score'], [])

    # merging a gsarray into another one that has a higher threhold
    # (should raises a warning on potential information loss)
    gs_array = GrowableSparseArray(n_iter=1, threshold=1)
    gs_array.append(0, np.ones((5, 1)))
    gs_array2 = GrowableSparseArray(n_iter=1, threshold=0)  # lower threshold
    assert_warns(UserWarning, gs_array2.merge, gs_array)
    assert_array_equal(gs_array.get_data()['iter_id'],
                       gs_array2.get_data()['iter_id'])
    assert_array_equal(gs_array.get_data()['x_id'],
                       gs_array2.get_data()['x_id'])
    assert_array_equal(gs_array.get_data()['y_id'],
                       gs_array2.get_data()['y_id'])
    assert_array_equal(gs_array.get_data()['score'],
                       gs_array2.get_data()['score'])
