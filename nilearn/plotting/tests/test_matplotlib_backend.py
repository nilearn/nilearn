import warnings
from unittest.mock import patch

import pytest

from nilearn.plotting import _set_mpl_backend


@patch("matplotlib.use")
@patch("matplotlib.get_backend", side_effect=["backend_1", "backend_2"])
def test_should_raise_warning_if_backend_changes(*_):
    # The backend values returned by matplotlib.get_backend are different.
    # Warning should be raised to inform user of the backend switch.
    with pytest.warns(UserWarning, match="Backend changed to backend_2..."):
        _set_mpl_backend()


@patch("matplotlib.use")
@patch("matplotlib.get_backend", side_effect=["backend_1", "backend_1"])
def test_should_not_raise_warning_if_backend_is_not_changed(*_):
    # The backend values returned by matplotlib.get_backend are identical.
    # Warning should not be raised.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _set_mpl_backend()


@patch(
    "matplotlib.use", side_effect=[Exception("Failed to switch backend"), True]
)
def test_should_switch_to_agg_backend_if_current_backend_fails(use_mock):
    # First call to `matplotlib.use` raises an exception, hence the default Agg
    # backend should be triggered
    _set_mpl_backend()

    assert use_mock.call_count == 2
    # Check that the most recent call to `matplotlib.use` has arg `Agg`
    use_mock.assert_called_with("Agg")


@patch("matplotlib.__version__", "0.0.0")
def test_should_raise_import_error_for_version_check():
    with pytest.raises(ImportError, match="A matplotlib version of at least"):
        _set_mpl_backend()
