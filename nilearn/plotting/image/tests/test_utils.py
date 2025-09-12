import numpy as np
import pytest

from nilearn.plotting.image.utils import get_cropped_cbar_ticks


@pytest.mark.parametrize(
    "vmin,vmax,threshold,num_ticks,expected",
    [
        (0, 0, None, 5, [0]),
        (3, 3, None, 5, [3]),
        (-3, -3, None, 5, [-3]),
        (0, 3, None, 5, [0, 0.75, 1.5, 2.25, 3]),
        (-3, 0, None, 5, [-3, -2.25, -1.5, -0.75, 0]),
        (-5, 5, None, 5, [-5, -2.5, 0, 2.5, 5]),
        (-3, 5, None, 5, [-3, -1, 1, 3, 5]),
        (-5, 3, None, 5, [-5, -3, -1, 1, 3]),
        # bug, should return [0], or at least 0.5 should appear after 0
        (0, 0, 0.5, 5, [0.5, 0, 0, 0, 0]),
        # bug, should return [3], or discuss behavior, should threshold appear
        # in the list when vmin==vmax
        (3, 3, 0.5, 5, [0.5, 3, 3, 3, 3]),
        # bug
        (-3, -3, 0.5, 5, [0.5, -3, -3, -3, -3]),
        (0, 3, 0.5, 5, [0, 0.5, 1.5, 2.25, 3]),
        # # bug 0.5 should not be between -1.5 and 0
        (-3, 0, 0.5, 5, [-3, -2.25, -1.5, 0.5, 0]),
        # # bug (threshold not in the list)
        (-3, 5, 0.5, 5, [-3, -1, 1, 3, 5]),
        # # bug (threshold not in the list)
        (-3, 5, 0.5, 7, [-3, -1.66, -0.33, 1, 2.33, 3.66, 5]),
        # # bug (threshold not in the list)
        (-3, 5, 0.5, 6, [-3, -1.4, 0.2, 1.8, 3.4, 5]),
        (-3, 5, 0.7, 5, [-3, -0.7, 0.7, 3, 5]),
        (-5, 3, 0.5, 5, [-5, -3, -0.5, 0.5, 3]),
        (-5, 3, 0.5, 7, [-5, -3.66, -2.33, -0.5, 0.5, 1.66, 3]),
        (-5, 3, 0.5, 6, [-5, -3.4, -1.8, -0.5, 0.5, 3]),
        # # bug (threshold not in the list)
        (-5, 5, 0.5, 5, [-5, -2.5, 0, 2.5, 5]),
        # # bug (threshold in the list)
        (-5, 5, 0.5, 7, [-5, -3.33, -1.66, 0, 1.66, 3.33, 5]),
        # # bug (threshold in the list)
        (-5, 5, 0.5, 6, [-5, -3, -1, 1, 3, 5]),
        (-5, 5, 2.5, 6, [-5, -2.5, -1, 1, 2.5, 5]),
        # # (negative threshold is not shown and positive threshold negative)
        (-3, 4, 5, 5, [-3, -1.25, 0.5, 2.25, -5]),
    ],
)
def test_get_cropped_cbar_ticks(vmin, vmax, threshold, num_ticks, expected):
    """Test nilearn.plotting.image.utils.get_cropped_cbar_ticks for valid
    values.
    """
    ticks = get_cropped_cbar_ticks(vmin, vmax, threshold, num_ticks)
    assert np.allclose(ticks, expected, rtol=1e-01)
