import re

import numpy as np
import pytest

from nilearn.plotting.matrix._utils import (
    VALID_REORDER_VALUES,
    VALID_TRI_VALUES,
    sanitize_labels,
    sanitize_reorder,
    sanitize_tri,
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
    with pytest.raises(ValueError, match="'tri' must be one of"):
        sanitize_tri(tri)
