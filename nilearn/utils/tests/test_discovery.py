import pytest

from nilearn.utils.discovery import all_estimators


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
def test_all_estimators(type_filter, n_expected):
    """Check number of estimators in public API."""
    estimators = all_estimators(type_filter=type_filter)
    assert len(estimators) == n_expected
