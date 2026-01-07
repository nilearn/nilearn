"""Run tests on nilearn.utils."""

import contextlib

import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.utils.discovery import all_displays, all_estimators, all_functions

with contextlib.suppress(ImportError):
    from rich import print


@pytest.mark.parametrize(
    "type_filter, n_expected",
    [
        (None, 33),
        ("classifier", 3),
        ("regressor", 3),
        ("cluster", 2),
        ("masker", 15),
        ("multi_masker", 6),
        ("transformer", 22),
    ],
)
def test_all_estimators(
    type_filter,
    n_expected,
):
    """Check number of estimators in public API."""
    estimators = all_estimators(type_filter=type_filter)
    print(estimators)
    assert len(estimators) == n_expected


def test_all_functions():
    """Check number of functions in public API."""
    fn = all_functions()
    print(fn)
    if is_matplotlib_installed():
        assert len(fn) == 173
    else:
        assert len(fn) == 136


@pytest.mark.parametrize(
    "type_filter, n_expected",
    [
        (None, 27),
        ("slicer", 24),
        ("axe", 3),
    ],
)
def test_all_displays(
    type_filter,
    n_expected,
):
    """Check number of functions in public API."""
    disp = all_displays(type_filter)
    print(disp)
    if is_matplotlib_installed():
        assert len(disp) == n_expected
    else:
        assert len(disp) == 0
