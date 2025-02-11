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


def test_coerce_to_dict_with_string():
    test_input = "StopSuccess - Go"
    expected_output = {"StopSuccess - Go": "StopSuccess - Go"}
    actual_output = coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_strings():
    test_input = ["contrast_name_0", "contrast_name_1"]
    expected_output = {
        "contrast_name_0": "contrast_name_0",
        "contrast_name_1": "contrast_name_1",
    }
    actual_output = coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_dict():
    test_input = {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]}
    expected_output = {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]}
    actual_output = coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_lists():
    test_input = [[0, 0, 1], [0, 1, 0]]
    expected_output = {"[0, 0, 1]": [0, 0, 1], "[0, 1, 0]": [0, 1, 0]}
    actual_output = coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_arrays():
    test_input = [np.array([0, 0, 1]), np.array([0, 1, 0])]
    expected_output = {
        "[0 0 1]": np.array([0, 0, 1]),
        "[0 1 0]": np.array([0, 1, 0]),
    }
    actual_output = coerce_to_dict(test_input)
    assert actual_output.keys() == expected_output.keys()
    for key in actual_output:
        assert np.array_equal(actual_output[key], expected_output[key])


def test_coerce_to_dict_with_list_of_ints():
    test_input = [1, 0, 1]
    expected_output = {"[1, 0, 1]": [1, 0, 1]}
    actual_output = coerce_to_dict(test_input)
    assert np.array_equal(
        actual_output["[1, 0, 1]"], expected_output["[1, 0, 1]"]
    )


def test_coerce_to_dict_with_array_of_ints():
    test_input = np.array([1, 0, 1])
    expected_output = {"[1 0 1]": np.array([1, 0, 1])}
    actual_output = coerce_to_dict(test_input)
    assert expected_output.keys() == actual_output.keys()
    assert np.array_equal(actual_output["[1 0 1]"], expected_output["[1 0 1]"])


def test_make_headings_with_contrasts_title_none():
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
    test_input = (1200, "a")
    expected_output = (1600, 800)
    expected_warning_text = (
        "Report size has invalid values. Using default 1600x800"
    )
    with pytest.warns(UserWarning, match=expected_warning_text):
        actual_output = check_report_dims(test_input)
    assert actual_output == expected_output
