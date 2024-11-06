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
        pytest.warns(UserWarning, match="Some plotting dependencies"),
    ):
        from nilearn.plotting import cm  # noqa
