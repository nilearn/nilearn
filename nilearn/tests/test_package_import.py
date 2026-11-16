"""Test related to importing nilearn optional dependencies are missing."""

import sys

import pytest

from nilearn._utils.helpers import is_matplotlib_installed


@pytest.mark.thread_unsafe
@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test should run only if matplotlib is not installed.",
)
def test_import_plotting_should_raise_warning_if_matplotlib_not_installed(
    monkeypatch,
):
    """Test if importing nilearn.plotting displays correct warning and raises
    error when matplotlib is not installed.
    """
    # the warning is only thrown on first import:
    # make sure nilearn.plotting is not already imported by another test
    monkeypatch.delitem(sys.modules, "nilearn.plotting", raising=False)
    with (
        pytest.raises(
            ModuleNotFoundError, match="No module named 'matplotlib'"
        ),
        pytest.warns(
            UserWarning, match="Some dependencies of nilearn.plotting"
        ),
    ):
        from nilearn.plotting import cm  # noqa: F401
