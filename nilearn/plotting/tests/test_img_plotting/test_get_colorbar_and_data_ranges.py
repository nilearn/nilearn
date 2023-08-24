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


def _expected_results_pos_neg(
    symmetric_cbar, vmin, vmax, symmetric_data_range, data
):
    """Help for expected_results.

    Return the expected `cbar_vmin`, `cbar_vmax`, `vmin`,
    and `vmax` for general case (masked or unmasked).
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if symmetric_cbar:
        if vmax is None and vmin is None:
            return (None, None, -data_max, data_max)
        else:
            return (None, None, -2, 2)
    elif symmetric_data_range:
        if vmax is None:
            return (data_min, data_max, -data_max, data_max)
        else:
            return (data_min, data_max, -2, 2)
    else:
        if vmin is None and vmax is None:
            return (data_min, data_max, data_min, data_max)
        elif vmin is None:
            return (data_min, data_max, data_min, vmax)
        elif vmax is None:
            return (data_min, data_max, vmin, data_max)
        else:
            return (data_min, data_max, vmin, vmax)


def _expected_results_pos(
    symmetric_cbar, vmin, vmax, symmetric_data_range, data
):
    """Help for expected_results.

    Return the expected `cbar_vmin`, `cbar_vmax`, `vmin`,
    and `vmax` for positive data.
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if symmetric_cbar is True:
        if vmax is None and vmin is None:
            return (None, None, -data_max, data_max)
        else:
            return (None, None, -2, 2)
    elif symmetric_data_range:
        if vmax is None:
            return (0, None, -data_max, data_max)
        else:
            return (0, None, -2, 2)
    else:
        if vmin is None and vmax is None:
            return (0, None, data_min, data_max)
        elif vmin is None:
            return (0, None, data_min, vmax)
        elif vmax is None:
            return (0, None, vmin, data_max)
        else:
            return (0, None, vmin, vmax)


def _expected_results_neg(
    symmetric_cbar, vmin, vmax, symmetric_data_range, data
):
    """Help for expected_results.

    Return the expected `cbar_vmin`, `cbar_vmax`, `vmin`,
    and `vmax` for negative data.
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if symmetric_cbar is True:
        if vmax is None and vmin is None:
            return (None, None, data_min, -data_min)
        else:
            return (None, None, -2, 2)
    elif symmetric_data_range:
        if vmax is None:
            return (None, 0, data_min, -data_min)
        else:
            return (None, 0, -2, 2)
    else:
        if vmin is None and vmax is None:
            return (None, 0, data_min, data_max)
        elif vmin is None:
            return (None, 0, data_min, vmax)
        elif vmax is None:
            return (None, 0, vmin, data_max)
        else:
            return (None, 0, vmin, vmax)


@pytest.fixture
def expected_results(
    case, data, symmetric_cbar, vmin, vmax, symmetric_data_range
):
    """Fixture to retrieve expected results."""
    expected = {
        "pos_neg": _expected_results_pos_neg,
        "pos": _expected_results_pos,
        "neg": _expected_results_neg,
        "masked": _expected_results_pos_neg,
    }
    return expected[case](
        symmetric_cbar, vmin, vmax, symmetric_data_range, data
    )


def test_get_colorbar_and_data_ranges_error():
    """Tests for _get_colorbar_and_data_ranges."""
    # vmin not accepted by default
    with pytest.raises(ValueError, match='does not accept a "vmin" argument'):
        _get_colorbar_and_data_ranges(
            data_pos_neg,
            vmax=None,
            symmetric_cbar=True,
            vmin=1.0,
        )

    # incompatible vmin and vmax if symmetric_cbar is True
    with pytest.raises(ValueError, match="vmin must be equal to -vmax"):
        _get_colorbar_and_data_ranges(
            data_pos_neg,
            vmin=0,
            vmax=1.0,
            symmetric_cbar=True,
            symmetric_data_range=False,
        )


@pytest.mark.parametrize(
    "case,data",
    [
        ("pos_neg", data_pos_neg),
        ("pos", data_pos),
        ("neg", data_neg),
        ("masked", data_masked),
    ],
)
@pytest.mark.parametrize(
    "symmetric_cbar,vmin,symmetric_data_range",
    [
        (True, None, True),
        ("auto", None, True),
        (False, None, False),
        (False, -1, False),
        (True, -2, False),
    ],
)
@pytest.mark.parametrize("vmax", [None, 2])
def test_get_colorbar_and_data_ranges(
    case,
    data,
    symmetric_cbar,
    vmin,
    vmax,
    symmetric_data_range,
    expected_results,
):
    """Tests for _get_colorbar_and_data_ranges.

    Tests values of `vmin`, `vmax`, `cbar_vmin`, and `cbar_vmax` with:

        - data having both positive and negatives values.
        - data having only negative values.
        - data having only positive values.
        - masked data.
    """
    assert (
        _get_colorbar_and_data_ranges(
            data,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
            symmetric_data_range=symmetric_data_range,
        )
        == expected_results
    )
