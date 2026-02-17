"""Test related to importing nilearn optional dependencies are missing."""

import pytest

from nilearn._utils.helpers import is_matplotlib_installed


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test should run only if matplotlib is not installed.",
)
def test_import_plotting_should_raise_warning_if_matplotlib_not_installed():
    """Tests if importing nilearn.plotting displays correct warning and raises
    error when matplotlib is not installed.
    """
    with (
        pytest.raises(
            ModuleNotFoundError, match="No module named 'matplotlib'"
        ),
        pytest.warns(
            UserWarning, match="Some dependencies of nilearn.plotting"
        ),
    ):
        from nilearn.plotting import cm  # noqa: F401
