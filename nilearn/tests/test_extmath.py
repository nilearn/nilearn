"""
Test the _utils.extmath module
"""

import nose

import numpy as np

from nose.tools import assert_true

from nilearn._utils.testing import assert_raises_regex
from nilearn._utils.extmath import fast_abs_percentile, is_spd, check_threshold


def test_fast_abs_percentile():
    data = np.arange(1, 100)
    for p in range(10, 100, 10):
        yield nose.tools.assert_equal, fast_abs_percentile(data, p - 1), p


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


def test_check_threshold():
    matrix = np.array([[1., 2.],
                       [2., 1.]])

    name = 'threshold'
    # few not correctly formatted strings for 'threshold'
    wrong_thresholds = ['0.1', '10', '10.2.3%', 'asdf%']
    for wrong_threshold in wrong_thresholds:
        assert_raises_regex(ValueError,
                            '{0}.+should be a number followed by '
                            'the percent sign'.format(name),
                            check_threshold,
                            wrong_threshold, matrix,
                            'fast_abs_percentile', name)

    threshold = object()
    assert_raises_regex(TypeError,
                        '{0}.+should be either a number '
                        'or a string'.format(name),
                        check_threshold, threshold, matrix,
                        'fast_abs_percentile', name)

    # To check if it also gives the expected score to given threshold
    assert_true(1. < check_threshold("50%", matrix,
                                     percentile_calculate=fast_abs_percentile,
                                     name=name) <= 2.)
