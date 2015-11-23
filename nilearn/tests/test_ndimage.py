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
    result = _peak_local_max(image, min_distance=1,
                             threshold_rel=0, indices=False)
    assert np.all(~ result)


def test_flat_peak_local_max():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:3, 1:3] = 10
    peaks = _peak_local_max(image, min_distance=1)
    assert len(peaks) == 4


def test_num_peaks_in_peak_local_max():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[1, 1] = 10
    image[1, 3] = 11
    image[1, 5] = 12
    image[3, 5] = 8
    image[5, 3] = 7
    assert len(_peak_local_max(image, min_distance=1)) == 5
    peaks_limited = _peak_local_max(image, min_distance=1, num_peaks=2)
    assert len(peaks_limited) == 2
    assert (1, 3) in peaks_limited
    assert (1, 5) in peaks_limited
    peaks_limited = _peak_local_max(image, min_distance=1, num_peaks=4)
    assert len(peaks_limited) == 4
    assert (1, 3) in peaks_limited
    assert (1, 5) in peaks_limited
    assert (1, 1) in peaks_limited
    assert (3, 5) in peaks_limited


def test_relative_and_absolute_thresholds_in_peak_local_max():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 10
    image[3, 3] = 20
    peaks_rel = _peak_local_max(image, min_distance=1, threshold_rel=0.5)
    assert len(peaks_rel) == 1
    np.testing.assert_allclose(peaks_rel, [(3, 3)])
    peaks_abs = _peak_local_max(image, min_distance=1, threshold_abs=10)
    assert len(peaks_abs) == 1
    np.testing.assert_allclose(peaks_abs, [(3, 3)])


def test_constant_image_in_peak_local_max():
    image = 128 * np.ones((20, 20), dtype=np.uint8)
    peaks = _peak_local_max(image, min_distance=1)
    assert len(peaks) == 0
