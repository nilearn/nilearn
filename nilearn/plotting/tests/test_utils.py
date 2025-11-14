"""Tests for private function \
   nilearn.plotting.image.img_plotting.get_colorbar_and_data_ranges.
"""

import numpy as np
import pytest

from nilearn.plotting._utils import (
    get_cbar_ticks,
    get_colorbar_and_data_ranges,
)


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
    vmin = 0
    vmax = 1.0
    # incompatible vmin and vmax if symmetric_cbar is True
    with pytest.warns(UserWarning, match=f"Specified {vmin=} and {vmax=}"):
        get_colorbar_and_data_ranges(
            data_pos_neg,
            vmin=vmin,
            vmax=vmax,
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


@pytest.mark.parametrize(
    "vmin,vmax,threshold,num_ticks,expected",
    [
        (0, 0, None, 5, [0]),
        (0, 0, 0, 5, [0]),
        (3, 3, None, 5, [3]),
        (3, 3, 0, 5, [3]),
        (-3, -3, None, 5, [-3]),
        (0, 0, 0.5, 5, [0]),
        (0, 0, None, 5, [0]),
        (-5, 5, None, 5, [-5, -2.5, 0, 2.5, 5]),
        (-5, 5, 0, 5, [-5, -2.5, 0, 2.5, 5]),
        (-3, 5, None, 5, [-3, -1, 0, 1, 3, 5]),
        (-3, 5, 0, 5, [-3, -1, 0, 1, 3, 5]),
        (-5, 3, None, 5, [-5, -3, -1, 0, 1, 3]),
        (0, 3, None, 5, [0, 0.75, 1.5, 2.2, 3]),
        (-3, 0, None, 5, [-3, -2.2, -1.5, -0.75, 0]),
        (3, 3, 0.5, 5, [0.5, 3]),
        (-3, -3, 0.5, 5, [-3, -0.5]),
        (0, 3, 0.5, 5, [0, 0.5, 1.5, 2.2, 3]),
        (-7.9, 0, 3, 5, [-7.9, -5.9, -4, -3, 0]),
        (0, 10, 0.5, 5, [0, 0.5, 2.5, 5, 7.5, 10]),
        (-3, 0, 0.5, 5, [-3, -2.2, -1.5, -0.5, 0]),
        (-10, 0, 0.5, 5, [-10, -7.5, -5, -2.5, -0.5, 0]),
        (-3, 5, 0.5, 5, [-3, -0.5, 0.5, 3, 5]),
        (-3, 5, 1e-5, 5, [-3, -1, 0, 1, 3, 5]),
        (-3, 5, 0.5, 7, [-3, -1.7, -0.5, 0, 0.5, 1, 2.3, 3.7, 5]),
        (-3, 5, 1.5, 7, [-3, -1.5, 0, 1.5, 2.3, 3.7, 5]),
        (-5, 10, 0.5, 5, [-5, -0.5, 0.5, 2.5, 6.2, 10]),
        (-5, 10, 2.5, 5, [-5, -2.5, 0, 2.5, 6.25, 10]),
        (-5, 3, 0.5, 5, [-5, -3, -0.5, 0.5, 3]),
        (-5, 3, 0.5, 7, [-5, -3.7, -2.3, -1, -0.5, 0, 0.5, 1.7, 3]),
        (-3, 4, 5, 5, [-5, -3, 0, 4, 5]),
        (-3, 5, 3.5, 5, [-3.5, -3, 0, 3.5, 5]),
        (-8, 8, 6, 5, [-8, -6, 0, 6, 8]),
        (-5, 5, 2.75, 5, [-5, -2.8, 0, 2.8, 5]),
        (-5, 5, 0.5, 5, [-5, -2.5, -0.5, 0.5, 2.5, 5]),
        (-5, 5, 2.5, 6, [-5, -2.5, 0, 2.5, 5]),
        (-10, 10, 0.1, 5, [-10, -5, -0.1, 0.1, 5, 10]),
        (-10, 10, 1.3, 5, [-10, -5, -1.3, 1.3, 5, 10]),
        (-10, 10, 2.5, 5, [-10, -5, -2.5, 0, 2.5, 5, 10]),
        (-10, 10, 3.5, 5, [-10, -3.5, 0, 3.5, 10]),
        (-10, 10, 7.5, 5, [-10, -7.5, 0, 7.5, 10]),
    ],
)
def test_get_cbar_ticks_threshold(vmin, vmax, threshold, num_ticks, expected):
    """Test nilearn.plotting._utils.get_cbar_ticks."""
    ticks = get_cbar_ticks(vmin, vmax, threshold, num_ticks)
    assert np.allclose(ticks, expected, rtol=1e-02)


@pytest.mark.parametrize(
    "vmin,vmax,cbar_tick_format,expected",
    [
        (0, 0, "%i", [0]),
        (0, 3, "%i", [0, 1, 2, 3]),
        (0, 4, "%i", [0, 1, 2, 3, 4]),
        (1, 5, "%i", [1, 2, 3, 4, 5]),
        (0, 5, "%i", [0, 1, 2, 3, 5]),
        (0, 10, "%i", [0, 2, 5, 7, 10]),
        (0, 0, "%.1f", [0]),
        (0, 1, "%.1f", [0, 0.2, 0.5, 0.8, 1]),
        (1, 2, "%.1f", [1, 1.2, 1.5, 1.8, 2]),
        (1.1, 1.2, "%.1f", [1.1, 1.2]),
        (0, np.nextafter(0, 1), "%.1f", [0.0e000, 5.0e-324]),
    ],
)
def test_get_cbar_ticks_int_tick_format(
    vmin, vmax, cbar_tick_format, expected
):
    """Test nilearn.plotting._utils.get_cbar_ticks for integer tick format."""
    ticks = get_cbar_ticks(vmin, vmax, tick_format=cbar_tick_format)
    assert np.allclose(ticks, expected, rtol=1e-02)


def test_get_cbar_ticks_int_threshold_float():
    """Test nilearn.plotting._utils.get_cbar_ticks for when integer tick
    format with threshold of type float specified.
    """
    with pytest.warns(
        UserWarning, match="You provided a non integer threshold"
    ):
        get_cbar_ticks(
            vmin=3, vmax=5, threshold=2.4, n_ticks=5, tick_format="%i"
        )
