"""Test the _utils.extmath module."""

import numpy as np

from nilearn._utils.extmath import fast_abs_percentile, is_spd


def test_fast_abs_percentile(rng):
    data = np.arange(100)
    rng.shuffle(data)
    for p in data:
        assert fast_abs_percentile(data, p) == p


def test_is_spd_with_non_symmetrical_matrix():
    matrix = np.arange(4).reshape(4, 1)
    assert not is_spd(matrix, verbose=0)

    matrix = np.array([[1, 1e-3 + 9e-19], [1e-3, 1]])
    assert is_spd(matrix, verbose=0)

    matrix = np.array([[1, 1e-3 + 1e-18], [1e-3, 1]])
    assert not is_spd(matrix, verbose=0)

    matrix = np.array([[1, 1e-3 + 9e-8], [1e-3, 1]])
    assert is_spd(matrix, decimal=4, verbose=0)

    matrix = np.array([[1, 1e-3 + 1e-7], [1e-3, 1]])
    assert not is_spd(matrix, decimal=4, verbose=0)


def test_is_spd_with_symmetrical_matrix():
    # matrix with negative eigenvalue
    matrix = np.array([[0, 1], [1, 0]])
    assert not is_spd(matrix, verbose=0)

    # matrix with 0 eigenvalue
    matrix = np.arange(4).reshape(2, 2)
    assert not is_spd(matrix, verbose=0)

    # spd matrix
    matrix = np.array([[2, 1], [1, 1]])
    assert is_spd(matrix, verbose=0)


def test_fast_abs_percentile_no_index_error():
    # check the offending low-level function
    fast_abs_percentile(np.arange(4))
