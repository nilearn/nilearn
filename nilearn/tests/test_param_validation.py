"""
Test the _utils.param_validation module
"""

import warnings
import numpy as np

from nose.tools import assert_true, assert_equal

from nilearn._utils.testing import assert_raises_regex, assert_warns

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import check_threshold


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

    # Test threshold as int, threshold=2 should return as it is
    # since it is not string
    assert_equal(check_threshold(2, matrix, percentile_func=fast_abs_percentile), 2)

    # check whether raises a warning if given threshold is higher than expected
    assert_warns(UserWarning, check_threshold, 3., matrix,
                 percentile_func=fast_abs_percentile)

    # test with numpy scalar as argument
    threshold = 2.
    threshold_numpy_scalar = np.float64(threshold)
    assert_equal(
        check_threshold(threshold, matrix, percentile_func=fast_abs_percentile),
        check_threshold(threshold_numpy_scalar, matrix, percentile_func=fast_abs_percentile))

    # Test for threshold provided as a percentile of the data (str ending with a
    # %)
    assert_true(1. < check_threshold("50%", matrix,
                                     percentile_func=fast_abs_percentile,
                                     name=name) <= 2.)
