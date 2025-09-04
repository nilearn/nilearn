from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import (
    assert_almost_equal,
)

from nilearn._utils.glm import (
    check_and_load_tables,
    coerce_to_dict,
    create_cosine_drift,
)


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


# load the spm file to test cosine basis
my_path = Path(__file__).parents[2] / "glm" / "tests"
full_path_design_matrix_file = my_path / "spm_dmtx.npz"
DESIGN_MATRIX = np.load(full_path_design_matrix_file)


def test_cosine_drift():
    # add something so that when the tests are launched
    # from a different directory
    spm_drifts = DESIGN_MATRIX["cosbf_dt_1_nt_20_hcut_0p1"]
    frame_times = np.arange(20)
    high_pass_frequency = 0.1
    nilearn_drifts = create_cosine_drift(high_pass_frequency, frame_times)
    assert_almost_equal(spm_drifts[:, 1:], nilearn_drifts[:, :-2])
    # nilearn_drifts is placing the constant at the end [:, : - 1]
