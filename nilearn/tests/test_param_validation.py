"""
Test the _utils.param_validation module
"""

import warnings
import numpy as np

from nose.tools import assert_true, assert_equal

from nilearn._utils.testing import assert_raises_regex, assert_warns

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (check_threshold,
                                             check_parameters_megatrawls_datasets)


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


def test_check_parameters_megatrawls_datasets():
    # testing whether the function raises the same error message as in
    # main function if wrong input parameters are given
    # parameters are dimensionality, timeseries, matrices
    message = "Invalid {0} name is given: {1}"

    invalid_inputs_dimensionality = [1, 5, 30]
    valid_inputs_dimensionality = [25, 50, 100, 200, 300]
    assert_raises_regex(ValueError,
                        message.format('dimensionality', invalid_inputs_dimensionality),
                        check_parameters_megatrawls_datasets,
                        invalid_inputs_dimensionality, valid_inputs_dimensionality,
                        'dimensionality')

    invalid_inputs_timeseries = ['asdf', 'time', 'st2']
    valid_inputs_timeseries = ['multiple_spatial_regression', 'eigen_regression']
    assert_raises_regex(ValueError,
                        message.format('timeseries', invalid_inputs_timeseries),
                        check_parameters_megatrawls_datasets,
                        invalid_inputs_timeseries, valid_inputs_timeseries,
                        'timeseries')

    invalid_output_names = ['net1', 'net2']
    valid_output_names = ['correlation', 'partial_correlation']
    assert_raises_regex(ValueError,
                        message.format('matrices', invalid_output_names),
                        check_parameters_megatrawls_datasets,
                        invalid_output_names, valid_output_names, 'matrices')

    # giving a valid input as a single element but not as a list to test
    # if it raises same error message
    message = ("Input given for {0} should be in list. "
               "You have given as single variable: {1}")
    valid_matrix_name = 'correlation'
    assert_raises_regex(TypeError,
                        message.format('matrices', valid_matrix_name),
                        check_parameters_megatrawls_datasets,
                        valid_matrix_name, valid_output_names, 'matrices')
