""" Test the ndimage module

This test file is in nilearn/tests because nosetests ignores modules whose
name starts with an underscore
"""
from nose.tools import assert_raises

import numpy as np

from nilearn._utils.ndimage import largest_connected_component


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    assert_raises(ValueError, largest_connected_component, a)
    a[1:3, 1:3, 1:3] = 1
    np.testing.assert_equal(a, largest_connected_component(a))
    b = a.copy()
    b[5, 5, 5] = 1
    np.testing.assert_equal(a, largest_connected_component(b))
