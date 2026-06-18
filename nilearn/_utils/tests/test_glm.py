import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from nilearn._utils.glm import (
    check_and_load_tables,
    coerce_to_dict,
    validate_design_matrix,
)
from nilearn.glm.first_level import make_first_level_design_matrix


def test_design_matrix_no_experimental_paradigm(frame_times):
    """Test design matrix creation \
        when no experimental paradigm is provided.
    """
    _, X, names = validate_design_matrix(
        make_first_level_design_matrix(
            frame_times, drift_model="polynomial", drift_order=3
        )
    )
    assert len(names) == 4
    x = np.linspace(-0.5, 0.5, len(frame_times))
    assert_almost_equal(X[:, 0], x)


def test_design_matrix_regressors_provided_manually(rng, frame_times):
    """Test design matrix creation when regressors are provided manually."""
    ax = rng.standard_normal(size=(len(frame_times), 4))
    _, X, names = validate_design_matrix(
        make_first_level_design_matrix(
            frame_times, drift_model="polynomial", drift_order=3, add_regs=ax
        )
    )
    assert_almost_equal(X[:, 0], ax[:, 0])
    assert len(names) == 8
    assert X.shape[1] == 8

    # with pandas Dataframe
    axdf = pd.DataFrame(ax)
    _, X1, names = validate_design_matrix(
        make_first_level_design_matrix(
            frame_times, drift_model="polynomial", drift_order=3, add_regs=axdf
        )
    )
    assert_almost_equal(X1[:, 0], ax[:, 0])
    assert_array_equal(names[:4], np.arange(4))


def test_img_table_checks():
    """Check tables type and that can be loaded."""
    with pytest.raises(
        ValueError, match=r"Tables to load can only be TSV or CSV."
    ):
        check_and_load_tables([".csv", ".csv"], "")
    with pytest.raises(
        TypeError,
        match="can only be a pandas DataFrame, a Path object or a string",
    ):
        check_and_load_tables([[], pd.DataFrame()], "")
    with pytest.raises(
        ValueError, match=r"Tables to load can only be TSV or CSV."
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
