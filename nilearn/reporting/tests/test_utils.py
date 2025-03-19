"""Test reporting utilities that do no require Matplotlib."""

import pytest

from nilearn.reporting._utils import (
    check_report_dims,
)


def test_check_report_dims():
    """Check that invalid report dimensions are overridden with warning."""
    test_input = (1200, "a")
    expected_output = (1600, 800)
    expected_warning_text = (
        "Report size has invalid values. Using default 1600x800"
    )

    with pytest.warns(UserWarning, match=expected_warning_text):
        actual_output = check_report_dims(test_input)
    assert actual_output == expected_output
