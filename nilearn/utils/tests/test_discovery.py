import pytest

from nilearn.utils.discovery import all_estimators, all_functions


@pytest.mark.parametrize(
    "type_filter, n_expected",
    [
        (None, 31),
        ("classifier", 3),
        ("regressor", 3),
        ("cluster", 2),
        ("masker", 13),
        ("multi_masker", 4),
        ("transformer", 20),
    ],
)
def test_all_estimators(
    matplotlib_pyplot,  # noqa : ARG001
    type_filter,
    n_expected,
):
    """Check number of estimators in public API."""
    estimators = all_estimators(type_filter=type_filter)
    print(estimators)
    assert len(estimators) == n_expected


def test_all_functions(
    matplotlib_pyplot,  # noqa : ARG001
):
    """Check number of functions in public API."""
    fn = all_functions()
    print(fn)
    assert len(fn) == 168
