"""Tests for private function \
   nilearn.plotting.img_plotting.get_colorbar_and_data_ranges.
"""

import numpy as np
import pytest

from nilearn.plotting.img_plotting import get_colorbar_and_data_ranges


@pytest.fixture
def data_pos_neg():
    """Fixture for data with both positive and negative values."""
    # min: -0.5, max: 3.0
    return np.array(
        [[-0.5, 1.0, np.nan], [0.0, np.nan, -0.2], [1.5, 2.5, 3.0]]
    )


@pytest.fixture
def data_pos(data_pos_neg):
    """Fixture for data with only positive values."""
    # min: 0.5, max: 3.0
    return np.abs(data_pos_neg)


@pytest.fixture
def data_neg(data_pos_neg):
    """Fixture for data with only negative values."""
    # min: -3, max: -0.5
    return -np.abs(data_pos_neg)


@pytest.fixture
def data_masked(data_pos_neg):
    """Fixture for data with masked values."""
    # min: -0.5, max: 1.5
    return np.ma.masked_greater(data_pos_neg, 2.0)


def test_get_colorbar_and_data_ranges_error():
    """Tests for expected errors in get_colorbar_and_data_ranges."""
    # incompatible vmin and vmax if symmetric_cbar is True
    with pytest.raises(ValueError, match="vmin must be equal to -vmax"):
        get_colorbar_and_data_ranges(
            data_pos_neg,
            vmin=0,
            vmax=1.0,
            symmetric_cbar=True,
        )


@pytest.mark.parametrize(
    "case,symmetric_cbar",
    [
        ("data_pos_neg", True),
        ("data_pos_neg", "auto"),
        ("data_pos", True),
        ("data_neg", True),
    ],
)
@pytest.mark.parametrize(
    "vmin,vmax,expected_results",
    [
        (None, None, (-3, 3, -3, 3)),
        (np.nan, None, (-3, 3, -3, 3)),
        (None, np.nan, (-3, 3, -3, 3)),
        (None, "-5", (-3, 3, -3, 3)),
        (-1, None, (-1, 1, -1, 1)),
        (None, 2, (-2, 2, -2, 2)),
        (None, 4, (-4, 4, -4, 4)),
    ],
)
def test_get_colorbar_and_data_ranges_symmetric_cbar_true(
    request,
    case,
    vmin,
    vmax,
    expected_results,
    symmetric_cbar,
):
    """Test for get_colorbar_and_data_ranges with symmetric colorbar."""
    data = request.getfixturevalue(case)
    assert expected_results == get_colorbar_and_data_ranges(
        data,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )


@pytest.mark.parametrize(
    "vmin,vmax,expected_results",
    [
        (None, None, (None, None, -0.5, 3)),
        (-1, None, (-1, None, -1, 3)),
        (None, 2, (None, 2, -0.5, 2)),
        (-0.25, 1, (-0.25, 1, -0.25, 1)),
    ],
)
def test_get_colorbar_and_data_ranges_symmetric_cbar_false_pos_neg(
    data_pos_neg,
    vmin,
    vmax,
    expected_results,
):
    """
    Test for get_colorbar_and_data_ranges with non-symmetric colorbar.

    Using data with both positive and negatives values.
    """
    assert expected_results == get_colorbar_and_data_ranges(
        data_pos_neg,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=False,
    )


@pytest.mark.parametrize(
    "vmin,vmax,expected_results",
    [
        (None, None, (0, None, 0, 3)),
        (None, 2, (0, 2, 0, 2)),
        (1, 2.5, (1, 2.5, 1, 2.5)),
    ],
)
@pytest.mark.parametrize("symmetric_cbar", ["auto", False])
def test_get_colorbar_and_data_ranges_symmetric_cbar_false_pos(
    data_pos,
    vmin,
    vmax,
    symmetric_cbar,
    expected_results,
):
    """
    Test for get_colorbar_and_data_ranges with non-symmetric colorbar.

    Using data with only positive values.
    """
    assert expected_results == get_colorbar_and_data_ranges(
        data_pos,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )


@pytest.mark.parametrize(
    "vmin,vmax,expected_results",
    [
        (None, None, (None, 0, -3, 0)),
        (None, 2, (None, 2, -3, 2)),
        (1, 2.5, (1, 2.5, 1, 2.5)),
    ],
)
@pytest.mark.parametrize("symmetric_cbar", ["auto", False])
def test_get_colorbar_and_data_ranges_symmetric_cbar_false_neg(
    data_neg,
    vmin,
    vmax,
    symmetric_cbar,
    expected_results,
):
    """
    Test for get_colorbar_and_data_ranges with non-symmetric colorbar.

    Using data with only negatives values.
    """
    assert expected_results == get_colorbar_and_data_ranges(
        data_neg,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )


@pytest.mark.parametrize(
    "vmin,vmax,symmetric_cbar,expected_results",
    [
        (None, None, True, (-1.5, 1.5, -1.5, 1.5)),
        (None, None, "auto", (-1.5, 1.5, -1.5, 1.5)),
        (None, None, False, (None, None, -0.5, 1.5)),
    ],
)
def test_get_colorbar_and_data_ranges_masked(
    data_masked, vmin, vmax, symmetric_cbar, expected_results
):
    """Test for get_colorbar_and_data_ranges with masked data."""
    assert expected_results == get_colorbar_and_data_ranges(
        data_masked,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )


def test_get_colorbar_and_data_ranges_force_min_stat_map_value(data_pos_neg):
    """Test for get_colorbar_and_data_ranges with force_min_stat_map_value."""
    expected_results = (0, None, 0, 3)
    assert expected_results == get_colorbar_and_data_ranges(
        data_pos_neg,
        vmin=None,
        vmax=None,
        symmetric_cbar="auto",
        force_min_stat_map_value=0,
    )
