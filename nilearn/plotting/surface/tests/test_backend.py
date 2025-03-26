import pytest

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn.plotting.surface._backend import (
    _check_hemisphere_is_valid,
    _check_view_is_valid,
    get_surface_backend,
)


@pytest.mark.parametrize(
    "view,is_valid",
    [
        ("lateral", True),
        ("medial", True),
        ("latreal", False),
        ((100, 100), True),
        ([100.0, 100.0], True),
        ((100, 100, 1), False),
        (("lateral", "medial"), False),
        ([100, "bar"], False),
    ],
)
def test_check_view_is_valid(view, is_valid):
    assert _check_view_is_valid(view) is is_valid


@pytest.mark.parametrize(
    "hemi,is_valid",
    [
        ("left", True),
        ("right", True),
        ("both", True),
        ("lft", False),
    ],
)
def test_check_hemisphere_is_valid(hemi, is_valid):
    assert _check_hemisphere_is_valid(hemi) is is_valid


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test is run only if matplotlib is not installed.",
)
def test_get_surface_backend_matplotlib_not_installed():
    """Tests to see if get_surface_backend raises error when matplotlib is not
    installed.
    """
    with pytest.raises(ImportError, match="Using engine"):
        get_surface_backend("matplotlib")


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="This test is run only if plotly is not installed.",
)
def test_get_surface_backend_plotly_not_installed():
    """Tests to see if get_surface_backend raises error when plotly is not
    installed.
    """
    with pytest.raises(ImportError, match="Using engine"):
        get_surface_backend("plotly")


def test_get_surface_backend_unknown_error():
    """Tests to see if get_surface_backend raises error when plotly is not
    installed.
    """
    with pytest.raises(ValueError, match="Unknown plotting"):
        get_surface_backend("unknown")
