from pathlib import Path

import pytest

from nilearn._utils.helpers import (
    _warn_deprecated_params,
    is_kaleido_installed,
    is_plotly_installed,
    rename_parameters,
    stringify_path,
    transfer_deprecated_param_vals,
)


def _mock_args_for_testing_replace_parameter():
    """Create mock deprecated & replacement parameters for use \
       with testing functions related to replace_parameters().
    """
    mock_kwargs_with_deprecated_params_used = {
        "unchanged_param_0": "unchanged_param_0_val",
        "deprecated_param_0": "deprecated_param_0_val",
        "deprecated_param_1": "deprecated_param_1_val",
        "unchanged_param_1": "unchanged_param_1_val",
    }
    replacement_params = {
        "deprecated_param_0": "replacement_param_0",
        "deprecated_param_1": "replacement_param_1",
    }
    return mock_kwargs_with_deprecated_params_used, replacement_params


@pytest.mark.thread_unsafe
def test_rename_parameters():
    """Test deprecated mock parameters in a mock function.

    Checks that the deprecated parameters transfer their values correctly
    to replacement parameters and all deprecation warning are raised as
    expected.
    """
    _, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_output = ("dp0", "dp1", "up0", "up1")

    @rename_parameters(
        replacement_params,
        "0.6.1rc",
        "other_lib",
    )
    def mock_function(
        replacement_param_0,
        replacement_param_1,
        unchanged_param_0,
        unchanged_param_1,
    ):
        return (
            replacement_param_0,
            replacement_param_1,
            unchanged_param_0,
            unchanged_param_1,
        )

    with pytest.warns(
        FutureWarning,
        match=r'"deprecated_param_[01]" will be removed in 0\.6\.1rc ',
    ):
        actual_output = mock_function(
            deprecated_param_0="dp0",
            deprecated_param_1="dp1",
            unchanged_param_0="up0",
            unchanged_param_1="up1",
        )

    assert actual_output == expected_output


def test_transfer_deprecated_param_vals():
    """Check that values assigned to deprecated parameters are \
       correctly reassigned to the replacement parameters.
    """
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_output = {
        "unchanged_param_0": "unchanged_param_0_val",
        "replacement_param_0": "deprecated_param_0_val",
        "replacement_param_1": "deprecated_param_1_val",
        "unchanged_param_1": "unchanged_param_1_val",
    }
    actual_output = transfer_deprecated_param_vals(
        replacement_params,
        mock_input,
    )
    assert actual_output == expected_output


@pytest.mark.thread_unsafe
def test_future_warn_deprecated_params():
    """Check that the correct warning is displayed."""
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    with pytest.warns(
        FutureWarning, match="be removed in sometime release of somelib"
    ):
        _warn_deprecated_params(
            replacement_params,
            end_version="sometime",
            lib_name="somelib",
            kwargs=mock_input,
        )


def test_is_plotly_installed():
    is_plotly_installed()


def test_is_kaleido_installed():
    is_kaleido_installed()


def test_stringify_path():
    assert isinstance(stringify_path(Path("foo") / "bar"), str)
    assert stringify_path([]) == []
