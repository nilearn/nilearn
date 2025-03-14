import numpy as np
import pandas as pd
import pytest

from nilearn._utils.glm import check_and_load_tables, coerce_to_dict


def test_img_table_checks():
    # check tables type and that can be loaded
    with pytest.raises(
        ValueError, match="Tables to load can only be TSV or CSV."
    ):
        check_and_load_tables([".csv", ".csv"], "")
    with pytest.raises(
        TypeError,
        match="can only be a pandas DataFrame, a Path object or a string",
    ):
        check_and_load_tables([[], pd.DataFrame()], "")
    with pytest.raises(
        ValueError, match="Tables to load can only be TSV or CSV."
    ):
        check_and_load_tables([".csv", pd.DataFrame()], "")


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
