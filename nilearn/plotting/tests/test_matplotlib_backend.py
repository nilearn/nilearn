from unittest.mock import patch

SKIP_REASON = "matplotlib missing; necessary for this test"
try:
    import matplotlib  # noqa: F401
except ImportError:
    MATPLOTLIB_INSTALLED = False
else:
    MATPLOTLIB_INSTALLED = True
import pytest
import warnings

from nilearn.plotting import _set_mpl_backend


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason=SKIP_REASON)
@patch("matplotlib.use")
@patch("matplotlib.get_backend", side_effect=["backend_1", "backend_2"])
def test_should_raise_warning_if_backend_changes(*_):
    # The backend values returned by matplotlib.get_backend are different.
    # Warning should be raised to inform user of the backend switch.
    with pytest.warns(UserWarning, match="Backend changed to backend_2..."):
        _set_mpl_backend()


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason=SKIP_REASON)
@patch("matplotlib.use")
@patch("matplotlib.get_backend",
       side_effect=["backend_1", "backend_1"])
def test_should_not_raise_warning_if_backend_is_not_changed(*_):
    # The backend values returned by matplotlib.get_backend are identical.
    # Warning should not be raised.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _set_mpl_backend()


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason=SKIP_REASON)
@patch("matplotlib.use",
       side_effect=[Exception("Failed to switch backend"), True])
def test_should_switch_to_agg_backend_if_current_backend_fails(use_mock):
    # First call to `matplotlib.use` raises an exception, hence the default Agg
    # backend should be triggered
    _set_mpl_backend()

    assert use_mock.call_count == 2
    # Check that the most recent call to `matplotlib.use` has arg `Agg`
    use_mock.assert_called_with("Agg")
