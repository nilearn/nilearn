import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from nilearn._utils.helpers import (
    _warn_deprecated_params,
    compare_version,
    is_kaleido_installed,
    is_matplotlib_installed,
    is_plotly_installed,
    rename_parameters,
    set_mpl_backend,
    stringify_path,
    transfer_deprecated_param_vals,
)
from nilearn._utils.testing import on_windows_with_old_mpl_and_new_numpy


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


@pytest.mark.skipif(
    on_windows_with_old_mpl_and_new_numpy(),
    reason="Old matplotlib not compatible with numpy 2.0 on windows.",
)
@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_should_raise_custom_warning_if_mpl_not_installed():
    """Tests if, when provided, custom message is displayed together with
    default warning.
    """
    warning = "This package requires nilearn.plotting package."
    with (
        pytest.warns(UserWarning, match=warning + "\nSome dependencies"),
        pytest.raises(
            ModuleNotFoundError, match="No module named 'matplotlib'"
        ),
    ):
        set_mpl_backend(warning)


@pytest.mark.skipif(
    on_windows_with_old_mpl_and_new_numpy(),
    reason="Old matplotlib not compatible with numpy 2.0 on windows.",
)
@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_should_raise_warning_if_mpl_not_installed():
    """Tests if default warning is displayed when no custom message is
    specified.
    """
    with (
        pytest.warns(
            UserWarning, match="Some dependencies of nilearn.plotting"
        ),
        pytest.raises(
            ModuleNotFoundError, match="No module named 'matplotlib'"
        ),
    ):
        set_mpl_backend()


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="Test requires matplotlib to be installed.",
)
@patch("matplotlib.use")
@patch("matplotlib.get_backend", side_effect=["backend_1", "backend_2"])
def test_should_raise_warning_if_backend_changes(*_):
    # The backend values returned by matplotlib.get_backend are different.
    # Warning should be raised to inform user of the backend switch.
    with pytest.warns(UserWarning, match="Backend changed to backend_2..."):
        set_mpl_backend()


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="Test requires matplotlib to be installed.",
)
@patch("matplotlib.use")
@patch("matplotlib.get_backend", side_effect=["backend_1", "backend_1"])
def test_should_not_raise_warning_if_backend_is_not_changed(*_):
    # The backend values returned by matplotlib.get_backend are identical.
    # Warning should not be raised.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_mpl_backend()


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="Test requires matplotlib to be installed.",
)
@patch(
    "matplotlib.use", side_effect=[Exception("Failed to switch backend"), True]
)
def test_should_switch_to_agg_backend_if_current_backend_fails(use_mock):
    # First call to `matplotlib.use` raises an exception, hence the default Agg
    # backend should be triggered
    set_mpl_backend()

    assert use_mock.call_count == 2
    # Check that the most recent call to `matplotlib.use` has arg `Agg`
    use_mock.assert_called_with("Agg")


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="Test requires matplotlib to be installed.",
)
@patch("matplotlib.__version__", "0.0.0")
def test_should_raise_import_error_for_version_check():
    with pytest.raises(ImportError, match="A matplotlib version of at least"):
        set_mpl_backend()


def test_rename_parameters():
    """Test deprecated mock parameters in a mock function.

    Checks that the deprecated parameters transfer their values correctly
    to replacement parameters and all deprecation warning are raised as
    expected.
    """
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_output = ("dp0", "dp1", "up0", "up1")
    expected_warnings = [
        (
            'The parameter "deprecated_param_0" will be removed in 0.6.1rc '
            "release of other_lib. "
            'Please use the parameter "replacement_param_0" instead.'
        ),
        (
            'The parameter "deprecated_param_1" will be removed in 0.6.1rc '
            "release of other_lib. "
            'Please use the parameter "replacement_param_1" instead.'
        ),
    ]

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

    with warnings.catch_warnings(record=True) as raised_warnings:
        actual_output = mock_function(
            deprecated_param_0="dp0",
            deprecated_param_1="dp1",
            unchanged_param_0="up0",
            unchanged_param_1="up1",
        )

    assert actual_output == expected_output

    expected_warnings.sort()
    raised_warnings.sort(key=lambda mem: str(mem.message))
    for raised_warning_, expected_warning_ in zip(
        raised_warnings, expected_warnings
    ):
        assert raised_warning_.category is DeprecationWarning
        assert str(raised_warning_.message) == expected_warning_


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


def test_future_warn_deprecated_params():
    """Check that the correct warning is displayed."""
    mock_input, replacement_params = _mock_args_for_testing_replace_parameter()
    expected_warnings = [
        (
            'The parameter "deprecated_param_0" will be removed in sometime '
            "release of somelib. "
            'Please use the parameter "replacement_param_0" instead.'
        ),
        (
            'The parameter "deprecated_param_1" will be removed in sometime '
            "release of somelib. "
            'Please use the parameter "replacement_param_1" instead.'
        ),
    ]
    with warnings.catch_warnings(record=True) as raised_warnings:
        _warn_deprecated_params(
            replacement_params,
            end_version="sometime",
            lib_name="somelib",
            kwargs=mock_input,
        )
    expected_warnings.sort()
    raised_warnings.sort(key=lambda mem: str(mem.message))
    for raised_warning_, expected_warning_ in zip(
        raised_warnings, expected_warnings
    ):
        assert raised_warning_.category is DeprecationWarning
        assert str(raised_warning_.message) == expected_warning_


@pytest.mark.parametrize(
    "version_a,operator,version_b",
    [
        ("0.1.0", ">", "0.0.1"),
        ("0.1.0", ">=", "0.0.1"),
        ("0.1", "==", "0.1.0"),
        ("0.0.0", "<", "0.1.0"),
        ("1.0", "!=", "0.1.0"),
    ],
)
def test_compare_version(version_a, operator, version_b):
    assert compare_version(version_a, operator, version_b)


def test_compare_version_error():
    with pytest.raises(
        ValueError,
        match="'compare_version' received an unexpected operator <>.",
    ):
        compare_version("0.1.0", "<>", "1.1.0")


def test_is_plotly_installed():
    is_plotly_installed()


def test_is_kaleido_installed():
    is_kaleido_installed()


def test_stringify_path():
    assert isinstance(stringify_path(Path("foo") / "bar"), str)
    assert stringify_path([]) == []
