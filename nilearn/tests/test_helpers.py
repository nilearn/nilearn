import warnings

from nilearn._utils import helpers


def _mock_args_for_testing_replace_parameter():
    """
    :return: Creates mock deprecated & replacement parameters for use with
    testing functions related to replace_parameters().
    """
    mock_kwargs_with_deprecated_params_used = {
        'unchanged_param_0': 'unchanged_param_0_val',
        'deprecated_param_0': 'deprecated_param_0_val',
        'deprecated_param_1': 'deprecated_param_1_val',
        'unchanged_param_1': 'unchanged_param_1_val',
        }
    replacement_params = {
        'deprecated_param_0': 'replacement_param_0',
        'deprecated_param_1': 'replacement_param_1',
        }
    return mock_kwargs_with_deprecated_params_used, replacement_params


def test_rename_parameters():
    """ Integration tests that deprecates mock parameters in a mock function
    and checks that the deprecated parameters transfer their values correctly
    to replacement parameters and all deprecation warning are raised as
    expected.
    """
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_output = ('dp0', 'dp1', 'up0', 'up1')
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in 0.6.1rc '
         'release of other_lib. Please use the parameter "replacement_param_0"'
         ' instead.'
         ),
        ('The parameter "deprecated_param_1" will be removed in 0.6.1rc '
         'release of other_lib. Please use the parameter "replacement_param_1"'
         ' instead.'
         ),
        ]

    @helpers.rename_parameters(replacement_params, '0.6.1rc', 'other_lib', )
    def mock_function(replacement_param_0, replacement_param_1,
                      unchanged_param_0, unchanged_param_1):
        return (replacement_param_0, replacement_param_1, unchanged_param_0,
                unchanged_param_1
                )

    with warnings.catch_warnings(record=True) as raised_warnings:
        actual_output = mock_function(deprecated_param_0='dp0',
                                      deprecated_param_1='dp1',
                                      unchanged_param_0='up0',
                                      unchanged_param_1='up1',
                                      )

    assert actual_output == expected_output

    expected_warnings.sort()
    raised_warnings.sort(key=lambda mem: str(mem.message))
    for raised_warning_, expected_warning_ in zip(raised_warnings,
                                                  expected_warnings):
        assert raised_warning_.category is FutureWarning
        assert str(raised_warning_.message) == expected_warning_


def test_transfer_deprecated_param_vals():
    """ Unit test to check that values assigned to deprecated parameters are
    correctly reassigned to the replacement parameters.
    """
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_output = {
        'unchanged_param_0': 'unchanged_param_0_val',
        'replacement_param_0': 'deprecated_param_0_val',
        'replacement_param_1': 'deprecated_param_1_val',
        'unchanged_param_1': 'unchanged_param_1_val',
        }
    actual_ouput = helpers._transfer_deprecated_param_vals(
            replacement_params,
            mock_input,
            )
    assert actual_ouput == expected_output


def test_future_warn_deprecated_params():
    """ Unit test to check that the correct warning is displayed.
    """
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in sometime '
         'release of somelib. Please use the parameter "replacement_param_0" '
         'instead.'
         ),
        ('The parameter "deprecated_param_1" will be removed in sometime '
         'release of somelib. Please use the parameter "replacement_param_1" '
         'instead.'
         ),
        ]
    with warnings.catch_warnings(record=True) as raised_warnings:
        helpers._warn_deprecated_params(
                replacement_params,
                end_version='sometime',
                lib_name='somelib',
                kwargs=mock_input,
                )
    expected_warnings.sort()
    raised_warnings.sort(key=lambda mem: str(mem.message))
    for raised_warning_, expected_warning_ in zip(raised_warnings,
                                                  expected_warnings
                                                  ):
        assert raised_warning_.category is FutureWarning
        assert str(raised_warning_.message) == expected_warning_
