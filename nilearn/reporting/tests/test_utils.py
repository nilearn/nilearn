"""Test reporting utilities that do no require Matplotlib."""

import numpy as np
import pytest

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting._utils import (
    check_report_dims,
    coerce_to_dict,
    make_headings,
)


@pytest.mark.parametrize(
    "input, output",
    (
        # None
        [None, None],
        # string
        ["StopSuccess - Go", {"StopSuccess - Go": "StopSuccess - Go"}],
        # list_of_strings,
        [
            ["contrast_name_0", "contrast_name_1"],
            {
                "contrast_name_0": "contrast_name_0",
                "contrast_name_1": "contrast_name_1",
            },
        ],
        # dict
        [
            {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]},
            {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]},
        ],
        # list of lists
        [
            [[0, 0, 1], [0, 1, 0]],
            {"[0, 0, 1]": [0, 0, 1], "[0, 1, 0]": [0, 1, 0]},
        ],
    ),
)
def test_coerce_to_dict(input, output):
    """Check that proper dictionary of contrasts are generated."""
    actual_output = coerce_to_dict(input)

    assert actual_output == output


@pytest.mark.parametrize(
    "input, output",
    (
        # list of ints
        [[1, 0, 1], {"[1, 0, 1]": [1, 0, 1]}],
        # array
        [np.array([1, 0, 1]), {"[1 0 1]": np.array([1, 0, 1])}],
        # list of arrays
        [
            [np.array([0, 0, 1]), np.array([0, 1, 0])],
            {
                "[0 0 1]": np.array([0, 0, 1]),
                "[0 1 0]": np.array([0, 1, 0]),
            },
        ],
    ),
)
def test_coerce_to_dict_with_arrays(input, output):
    """Check that proper dictionary of contrasts are generated from arrays."""
    actual_output = coerce_to_dict(input)

    assert actual_output.keys() == output.keys()
    for key in actual_output:
        assert np.array_equal(actual_output[key], output[key])


def test_make_headings_with_contrasts_title_none():
    """Check SLM report with no title headings."""
    model = SecondLevelModel()
    test_input = (
        {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]},
        None,
        model,
    )
    expected_output = (
        "Report: Second Level Model for contrast_0, contrast_1",
        "Statistical Report for contrast_0, contrast_1",
        "Second Level Model",
    )
    actual_output = make_headings(*test_input)

    assert actual_output == expected_output


def test_make_headings_with_contrasts_title_custom():
    """Check SLM report with custom title headings."""
    model = SecondLevelModel()
    test_input = (
        {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]},
        "Custom Title for report",
        model,
    )
    expected_output = (
        "Custom Title for report",
        "Custom Title for report",
        "Second Level Model",
    )
    actual_output = make_headings(*test_input)

    assert actual_output == expected_output


def test_make_headings_with_contrasts_none_title_custom():
    """Check FLM report with custom title headings."""
    model = FirstLevelModel()
    test_input = (None, "Custom Title for report", model)
    expected_output = (
        "Custom Title for report",
        "Custom Title for report",
        "First Level Model",
    )
    actual_output = make_headings(*test_input)

    assert actual_output == expected_output


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
