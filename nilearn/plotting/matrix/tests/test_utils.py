import pytest

from nilearn.plotting.matrix._utils import (
    VALID_REORDER_VALUES,
    VALID_TRI_VALUES,
    sanitize_reorder,
    sanitize_tri,
)


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
