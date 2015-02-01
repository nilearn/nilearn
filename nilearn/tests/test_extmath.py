"""
Test the _utils.extmath module
"""

import nose

import numpy as np

from .._utils.extmath import fast_abs_percentile, is_spd


def test_fast_abs_percentile():
    data = np.arange(1, 100)
    for p in range(10, 100, 10):
        yield nose.tools.assert_equal, fast_abs_percentile(data, p-1), p


def test_is_spd_with_non_symmetrical_matrix():
    matrix = np.arange(4).reshape(4, 1)
    assert not is_spd(matrix, verbose=0)

    matrix = np.array([[1, 1e-3 + 9e-19],
                       [1e-3, 1]])
    assert is_spd(matrix, verbose=0)

    matrix = np.array([[1, 1e-3 + 1e-18],
                       [1e-3, 1]])
    assert not is_spd(matrix, verbose=0)

    matrix = np.array([[1, 1e-3 + 9e-8],
                       [1e-3, 1]])
    assert is_spd(matrix, decimal=4, verbose=0)

    matrix = np.array([[1, 1e-3 + 1e-7],
                       [1e-3, 1]])
    assert not is_spd(matrix, decimal=4, verbose=0)


def test_is_spd_with_symmetrical_matrix():
    # matrix with negative eigenvalue
    matrix = np.array([[0, 1],
                       [1, 0]])
    assert not is_spd(matrix, verbose=0)

    # matrix with 0 eigenvalue
    matrix = np.arange(4).reshape(2, 2)
    assert not is_spd(matrix, verbose=0)

    # spd matrix
    matrix = np.array([[2, 1],
                       [1, 1]])
    assert is_spd(matrix, verbose=0)
