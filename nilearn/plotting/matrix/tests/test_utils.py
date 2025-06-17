import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.plotting.matrix._utils import (
    VALID_REORDER_VALUES,
    VALID_TRI_VALUES,
    pad_contrast_matrix,
    sanitize_labels,
    sanitize_reorder,
    sanitize_tri,
)


def test_pad_contrast_matrix():
    """Test for contrasts padding before plotting.

    See https://github.com/nilearn/nilearn/issues/4211
    """
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    contrast = np.array([[1, -1]])
    padded_contrast = pad_contrast_matrix(contrast, dmtx)
    assert_array_equal(padded_contrast, np.array([[1, -1, 0, 0]]))

    contrast = np.eye(3)
    padded_contrast = pad_contrast_matrix(contrast, dmtx)
    assert_array_equal(
        padded_contrast,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        ),
    )


def test_sanitize_labels():
    labs = ["foo", "bar"]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of labels (2) unequal to length of matrix (6)."
        ),
    ):
        sanitize_labels((6, 6), labs)
    for lab in [labs, np.array(labs)]:
        assert sanitize_labels((2, 2), lab) == labs


@pytest.mark.parametrize("reorder", VALID_REORDER_VALUES)
def test_sanitize_reorder(reorder):
    if reorder is not True:
        assert sanitize_reorder(reorder) == reorder
    else:
        assert sanitize_reorder(reorder) == "average"


@pytest.mark.parametrize("reorder", [None, "foo", 2])
def test_sanitize_reorder_error(reorder):
    with pytest.raises(
        ValueError, match=("Parameter reorder needs to be one of")
    ):
        sanitize_reorder(reorder)


@pytest.mark.parametrize("tri", VALID_TRI_VALUES)
def test_sanitize_tri(tri):
    sanitize_tri(tri)


@pytest.mark.parametrize("tri", [None, "foo", 2])
def test_sanitize_tri_error(tri):
    with pytest.raises(
        ValueError,
        match=(
            f"Parameter tri needs to be one of: {', '.join(VALID_TRI_VALUES)}"
        ),
    ):
        sanitize_tri(tri)
