import re

import numpy as np
import pytest

from nilearn.plotting.matrix._backend import (
    _sanitize_labels,
    _sanitize_reorder,
    _sanitize_tri,
)
from nilearn.plotting.matrix.tests.test_matrix_plotting import VALID_TRI_VALUES

VALID_REORDER_VALUES = (True, False, "single", "complete", "average")


##############################################################################
# Some smoke testing for graphics-related code


def test_sanitize_labels():
    labs = ["foo", "bar"]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of labels (2) unequal to length of matrix (6)."
        ),
    ):
        _sanitize_labels((6, 6), labs)
    for lab in [labs, np.array(labs)]:
        assert _sanitize_labels((2, 2), lab) == labs


@pytest.mark.parametrize("tri", VALID_TRI_VALUES)
def test_sanitize_tri(tri):
    _sanitize_tri(tri)


@pytest.mark.parametrize("tri", [None, "foo", 2])
def test_sanitize_tri_error(tri):
    with pytest.raises(
        ValueError,
        match=(
            f"Parameter tri needs to be one of: {', '.join(VALID_TRI_VALUES)}"
        ),
    ):
        _sanitize_tri(tri)


@pytest.mark.parametrize("reorder", VALID_REORDER_VALUES)
def test_sanitize_reorder(reorder):
    if reorder is not True:
        assert _sanitize_reorder(reorder) == reorder
    else:
        assert _sanitize_reorder(reorder) == "average"


@pytest.mark.parametrize("reorder", [None, "foo", 2])
def test_sanitize_reorder_error(reorder):
    with pytest.raises(
        ValueError, match=("Parameter reorder needs to be one of")
    ):
        _sanitize_reorder(reorder)
