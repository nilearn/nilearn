# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Testing mriutils module"""

import pytest
from numpy.testing import assert_almost_equal

from ..mriutils import MRIError, calculate_dwell_time


def test_calculate_dwell_time():
    # Test dwell time calculation
    # This tests only that the calculation does what it appears to; needs some
    # external check
    assert_almost_equal(calculate_dwell_time(3.3, 2, 3), 3.3 / (42.576 * 3.4 * 3 * 3))
    # Echo train length of 1 is valid, but returns 0 dwell time
    assert_almost_equal(calculate_dwell_time(3.3, 1, 3), 0)
    with pytest.raises(MRIError):
        calculate_dwell_time(3.3, 0, 3.0)
    with pytest.raises(MRIError):
        calculate_dwell_time(3.3, 2, -0.1)
