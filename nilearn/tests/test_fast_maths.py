"""
Test the _utils.extmath module
"""

import nose

import numpy as np

from .._utils.extmath import fast_abs_percentile


def test_fast_abs_percentile():
    data = np.arange(1, 100)
    for p in range(10, 100, 10):
        yield nose.tools.assert_equal, fast_abs_percentile(data, p-1), p



