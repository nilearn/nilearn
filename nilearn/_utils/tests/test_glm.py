import numpy as np
import pytest

from nilearn._utils.glm import (
    coerce_to_dict,
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
