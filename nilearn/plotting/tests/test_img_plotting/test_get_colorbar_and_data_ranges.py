"""
Tests for private function
nilearn.plotting.img_plotting._get_colorbar_and_data_ranges.
"""

import numpy as np
import pytest

from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges

data_pos_neg = np.array(
    [[-0.5, 1.0, np.nan], [0.0, np.nan, -0.2], [1.5, 2.5, 3.0]]
)


data_pos = np.array([[0, 1.0, np.nan], [0.0, np.nan, 0], [1.5, 2.5, 3.0]])


data_neg = np.array([[-0.5, 0, np.nan], [0.0, np.nan, -0.2], [0, 0, 0]])


data_masked = np.ma.masked_greater(data_pos_neg, 2.0)


def test_get_colorbar_and_data_ranges_with_vmin():
    """Tests for _get_colorbar_and_data_ranges.

    Tests that a ValueError is raised when vmin and
    symmetric_cbar are both provided.
    """
    with pytest.raises(ValueError, match='does not accept a "vmin" argument'):
        _get_colorbar_and_data_ranges(
            data_pos_neg, vmax=None, symmetric_cbar=True, kwargs={"vmin": 1.0}
        )


def _expected_results_pos_neg(symmetric_cbar, vmax, data):
    """Help for expected_results.

    Return the expected `cbar_vmin`, `cbar_vmax`, `vmin`,
    and `vmax` for general case.
    """
    data_max = np.nanmax(data)
    if symmetric_cbar:
        if vmax is None:
            return (None, None, -data_max, data_max)
        else:
            return (None, None, -2, 2)
    else:
        if vmax is None:
            return (np.nanmin(data), data_max, -data_max, data_max)
        else:
            return (np.nanmin(data), data_max, -2, 2)


def _expected_results_pos(symmetric_cbar, vmax, data):
    """Help for expected_results.

    Return the expected `cbar_vmin`, `cbar_vmax`, `vmin`,
    and `vmax` for positive data.
    """
    data_max = np.nanmax(data)
    if symmetric_cbar is True:
        if vmax is None:
            return (None, None, -data_max, data_max)
        else:
            return (None, None, -2, 2)
    else:
        if vmax is None:
            return (0, None, -data_max, data_max)
        else:
            return (0, None, -2, 2)


def _expected_results_neg(symmetric_cbar, vmax, data):
    """Help for expected_results.

    Return the expected `cbar_vmin`, `cbar_vmax`, `vmin`,
    and `vmax` for negative data.
    """
    data_min = np.nanmin(data)
    if symmetric_cbar is True:
        if vmax is None:
            return (None, None, data_min, -data_min)
        else:
            return (None, None, -2, 2)
    else:
        if vmax is None:
            return (None, 0, data_min, -data_min)
        else:
            return (None, 0, -2, 2)


@pytest.fixture
def expected_results(case, data, symmetric_cbar, vmax):
    """Fixture to retrieve expected results."""
    expected = {
        "pos_neg": _expected_results_pos_neg,
        "pos": _expected_results_pos,
        "neg": _expected_results_neg,
        "masked": _expected_results_pos_neg,
    }
    return expected[case](symmetric_cbar, vmax, data)


@pytest.mark.parametrize(
    "case,data",
    [
        ("pos_neg", data_pos_neg),
        ("pos", data_pos),
        ("neg", data_neg),
        ("masked", data_masked),
    ],
)
@pytest.mark.parametrize("symmetric_cbar", [True, False, "auto"])
@pytest.mark.parametrize("vmax", [None, 2])
def test_get_colorbar_and_data_ranges(
    case, data, symmetric_cbar, vmax, expected_results
):
    """Tests for _get_colorbar_and_data_ranges.

    Tests values of `vmin`, `vmax`, `cbar_vmin`, and `cbar_vmax` with:

        - data having both positive and negatives values.
        - data having only negative values.
        - data having only positive values.
        - masked data.
    """
    kwargs = {"aspect": "auto", "alpha": 0.9}
    assert (
        _get_colorbar_and_data_ranges(
            data, vmax=vmax, symmetric_cbar=symmetric_cbar, kwargs=kwargs
        )
        == expected_results
    )
