""" Test the ndimage module

This test file is in nilearn/tests because nosetests ignores modules whose
name starts with an underscore
"""
from scipy import ndimage
from nose.tools import assert_raises

import numpy as np

from nilearn._utils.ndimage import largest_connected_component, _peak_local_max


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


def test_empty_peak_local_max():
    image = np.zeros((10, 20))
    result = _peak_local_max(image, min_distance=1, threshold_rel=0)
    assert np.all(~ result)


def test_flat_peak_local_max():
    image = np.zeros((5, 5))
    image[1:3, 1:3] = 10
    peaks = _peak_local_max(image, min_distance=1)
    np.testing.assert_equal(len(peaks[peaks == 1]), 4)


def test_relative_and_absolute_thresholds_in_peak_local_max():
    image = np.zeros((5, 5))
    image[1, 1] = 10
    image[3, 3] = 20
    peaks_rel = _peak_local_max(image, min_distance=1, threshold_rel=0.5)
    np.testing.assert_equal(len(peaks_rel[peaks_rel == 1]), 1)
    peaks_abs = _peak_local_max(image, min_distance=1, threshold_abs=10)
    np.testing.assert_equal(len(peaks_abs[peaks_abs == 1]), 1)


def test_constant_image_in_peak_local_max():
    image = 128 * np.ones((20, 20))
    peaks = _peak_local_max(image, min_distance=1)
    np.testing.assert_equal(len(peaks[peaks == 1]), 0)


def test_trivial_cases_in_peak_local_max():
    trivial = np.zeros((25, 25))
    peaks = _peak_local_max(trivial, min_distance=1)
    assert (peaks.astype(np.bool) == trivial).all()
