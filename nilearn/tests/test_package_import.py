import pytest

from nilearn._utils.helpers import is_matplotlib_installed


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test should run only if matplotlib is not installed.",
)
def test_should_raise_warning_if_matplotlib_not_installed():
    with (
        pytest.raises(
            ModuleNotFoundError, match="No module named 'matplotlib'"
        ),
        pytest.warns(
            UserWarning, match="Some dependencies of nilearn.plotting"
        ),
    ):
        from nilearn.plotting import cm  # noqa

    with (
        pytest.raises(
            ModuleNotFoundError, match="No module named 'matplotlib'"
        ),
        pytest.warns(
            UserWarning, match="Some dependencies of nilearn.plotting"
        ),
    ):
        from nilearn.reporting import make_glm_report  # noqa
