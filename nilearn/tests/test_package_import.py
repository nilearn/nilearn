import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.plotting import _set_mpl_backend


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test should run only if matplotlib is not installed.",
)
def test_should_raise_warning_if_matplotlib_not_installed():
    with pytest.warns(UserWarning, match="Some plotting dependencies"):
        _set_mpl_backend()
