"""Test related to importing nilearn optional dependencies are missing."""

import sys

import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.testing import on_windows_with_old_mpl_and_new_numpy


@pytest.mark.skipif(
    on_windows_with_old_mpl_and_new_numpy(),
    reason="Old matplotlib not compatible with numpy 2.0 on windows.",
)
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


@pytest.mark.skipif(
    on_windows_with_old_mpl_and_new_numpy(),
    reason="Old matplotlib not compatible with numpy 2.0 on windows.",
)
@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test should run only if matplotlib is not installed.",
)
def test_import_reporting_should_raise_warning_if_matplotlib_not_installed():
    """Tests if importing nilearn.reporting.make_glm_report displays correct
    warning and raises error when matplotlib is not installed.
    """
    del sys.modules["nilearn.reporting"]
    with (
        pytest.warns(UserWarning, match="nilearn.reporting.glm_reporter and"),
        pytest.raises(
            ImportError,
            match="cannot import name 'make_glm_report' from*",
        ),
    ):
        from nilearn.reporting import make_glm_report  # noqa: F401


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test should run only if matplotlib is not installed.",
)
def test_import_get_clusters_table_when_matplotlib_not_installed():
    """Tests if nilearn.reporting.get_clusters_table can be imported without
    problems when matplotlib is not installed.
    """
    del sys.modules["nilearn.reporting"]
    with pytest.warns(UserWarning, match="nilearn.reporting.glm_reporter and"):
        from nilearn.reporting import get_clusters_table  # noqa: F401
