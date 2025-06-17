import pytest

from nilearn.plotting.matrix._utils import (
    VALID_TRI_VALUES,
    sanitize_tri,
)


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
