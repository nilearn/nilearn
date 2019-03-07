import warnings

from nilearn._utils import helpers


def _mock_args_for_testing_replace_parameter():
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


def test_transfer_deprecated_param_vals():
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
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in a future Nilearn version. Please use the parameter "replacement_param_0" instead.'),
        ('The parameter "deprecated_param_1" will be removed in a future Nilearn version. Please use the parameter "replacement_param_1" instead.'),
        ]
    with warnings.catch_warnings(record=True) as raised_warnings:
        helpers._warn_deprecated_params(
                replacement_params,
                end_version='future',
                lib_name='Nilearn',
                kwargs=mock_input,
                )
    for raised_warning_, expected_warning_ in zip(raised_warnings, expected_warnings):
        assert str(raised_warning_.message) == expected_warning_


def test_next_warn_deprecated_params():
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in the next Nilearn version. Please use the parameter "replacement_param_0" instead.'),
        ('The parameter "deprecated_param_1" will be removed in the next Nilearn version. Please use the parameter "replacement_param_1" instead.'),
        ]
    with warnings.catch_warnings(record=True) as raised_warnings:
        helpers._warn_deprecated_params(
                replacement_params,
                end_version='next',
                lib_name='Nilearn',
                kwargs=mock_input,
                )
    for raised_warning_, expected_warning_ in zip(raised_warnings, expected_warnings):
        assert str(raised_warning_.message) == expected_warning_


def test_version_warn_deprecated_params():
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in Nilearn version 0.6.1rc. Please use the parameter "replacement_param_0" instead.'),
        ('The parameter "deprecated_param_1" will be removed in Nilearn version 0.6.1rc. Please use the parameter "replacement_param_1" instead.'),
        ]
    with warnings.catch_warnings(record=True) as raised_warnings:
        helpers._warn_deprecated_params(
                replacement_params,
                end_version='0.6.1rc',
                lib_name='Nilearn',
                kwargs=mock_input,
                )
    for raised_warning_, expected_warning_ in zip(raised_warnings,
                                                  expected_warnings):
        assert str(raised_warning_.message) == expected_warning_


def test_other_lib_warn_deprecated_params():
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in the next other_lib version. Please use the parameter "replacement_param_0" instead.'),
        ('The parameter "deprecated_param_1" will be removed in the next other_lib version. Please use the parameter "replacement_param_1" instead.'),
        ]
    with warnings.catch_warnings(record=True) as raised_warnings:
        helpers._warn_deprecated_params(
                replacement_params,
                end_version='next',
                lib_name='other_lib',
                kwargs=mock_input,
                )
    for raised_warning_, expected_warning_ in zip(raised_warnings,
                                                  expected_warnings):
        assert str(raised_warning_.message) == expected_warning_


def test_replace_parameters():
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_output = ('dp0', 'dp1', 'up0', 'up1')
    expected_warnings = [
        ('The parameter "deprecated_param_0" will be removed in other_lib version 0.6.1rc. Please use the parameter "replacement_param_0" instead.'),
        ('The parameter "deprecated_param_1" will be removed in other_lib version 0.6.1rc. Please use the parameter "replacement_param_1" instead.'),
        ]

    @helpers.replace_parameters(replacement_params, '0.6.1rc', 'other_lib',)
    def mock_function(replacement_param_0, replacement_param_1, unchanged_param_0, unchanged_param_1):
        return replacement_param_0, replacement_param_1, unchanged_param_0, unchanged_param_1
    
    with warnings.catch_warnings(record=True) as raised_warnings:
        actual_output = mock_function(deprecated_param_0='dp0',
                                      deprecated_param_1='dp1',
                                      unchanged_param_0='up0',
                                      unchanged_param_1='up1',
                                      )
    
    assert actual_output == expected_output
    for raised_warning_, expected_warning_ in zip(raised_warnings,
                                                  expected_warnings):
        assert str(raised_warning_.message) == expected_warning_
