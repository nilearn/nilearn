"""
Tests from skimage feature testing for peak.py
"""
import numpy as np

from scipy import ndimage

from nose import tools

from nilearn.externals.skimage import peak


def make_2d_syntheticdata(lx, ly=None):
    if ly is None:
        ly = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly)) + 0.1 * np.random.randn(lx, ly)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l,
         ly // 2 - small_l:ly // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1,
         ly // 2 - small_l + 1:ly // 2 + small_l - 1] = (
             0.1 * np.random.randn(2 * small_l - 2, 2 * small_l - 2))
    data[lx // 2 - small_l, ly // 2 - small_l // 8:ly // 2 + small_l // 8] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5] = 1
    seeds[lx // 2 + small_l // 4, ly // 2 - small_l // 4] = 2
    return data, seeds


def test_reorder_labels():
    image = np.random.uniform(size=(40, 60))
    i, j = np.mgrid[0:40, 0:60]
    labels = 1 + (i >= 20) + (j >= 30) * 2
    labels[labels == 4] = 5
    i, j = np.mgrid[-3:4, -3:4]
    footprint = (i * i + j * j <= 9)
    expected = np.zeros(image.shape, float)
    for imin, imax in ((0, 20), (20, 40)):
        for jmin, jmax in ((0, 30), (30, 60)):
            expected[imin:imax, jmin:jmax] = ndimage.maximum_filter(
                image[imin:imax, jmin:jmax], footprint=footprint)
    expected = (expected == image)
    result = peak.peak_local_max(image, labels=labels, min_distance=1,
                                 threshold_rel=0, footprint=footprint,
                                 indices=False, exclude_border=False)
    assert (result == expected).all()


def test_indices_with_labels():
    image = np.random.uniform(size=(40, 60))
    i, j = np.mgrid[0:40, 0:60]
    labels = 1 + (i >= 20) + (j >= 30) * 2
    i, j = np.mgrid[-3:4, -3:4]
    footprint = (i * i + j * j <= 9)
    expected = np.zeros(image.shape, float)
    for imin, imax in ((0, 20), (20, 40)):
        for jmin, jmax in ((0, 30), (30, 60)):
            expected[imin:imax, jmin:jmax] = ndimage.maximum_filter(
                image[imin:imax, jmin:jmax], footprint=footprint)
    expected = (expected == image)
    result = peak.peak_local_max(image, labels=labels, min_distance=1,
                                 threshold_rel=0, footprint=footprint,
                                 indices=True, exclude_border=False)
    assert (result == np.transpose(expected.nonzero())).all()


def test_empty():
    image = np.zeros((10, 20))
    labels = np.zeros((10, 20), int)
    result = peak.peak_local_max(image, labels=labels,
                                 footprint=np.ones((3, 3), bool),
                                 min_distance=1, threshold_rel=0,
                                 indices=False, exclude_border=False)
    assert np.all(~ result)


def test_flat_peak():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:3, 1:3] = 10
    peaks = peak.peak_local_max(image, min_distance=1)
    assert len(peaks) == 4


def test_num_peaks():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[1, 1] = 10
    image[1, 3] = 11
    image[1, 5] = 12
    image[3, 5] = 8
    image[5, 3] = 7
    assert len(peak.peak_local_max(image, min_distance=1)) == 5
    peaks_limited = peak.peak_local_max(image, min_distance=1, num_peaks=2)
    assert len(peaks_limited) == 2
    assert (1, 3) in peaks_limited
    assert (1, 5) in peaks_limited
    peaks_limited = peak.peak_local_max(image, min_distance=1, num_peaks=4)
    assert len(peaks_limited) == 4
    assert (1, 3) in peaks_limited
    assert (1, 5) in peaks_limited
    assert (1, 1) in peaks_limited
    assert (3, 5) in peaks_limited
