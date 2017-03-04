""" Test the ndimage module

This test file is in nilearn/tests because nosetests ignores modules whose
name starts with an underscore
"""
from nose.tools import assert_raises, assert_true

import numpy as np

from nibabel import Nifti1Image
from nilearn._utils.ndimage import (largest_connected_component,
                                    _peak_local_max,
                                    largest_connected_component_img)
from nilearn._utils import testing

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

    img = testing.generate_labeled_regions(shape = (10,11,12),
                                           n_regions = 2 )

    assert_raises(ValueError, largest_connected_component, img)
    assert_raises(ValueError, largest_connected_component, "Test String")


def test_largest_cc_img():
    """ Check the extraction of the largest connected component, for niftis

    Similiar to smooth_img tests for largest connected_component_img, here also
    only the added features for largest_connected_component are tested.
    """

    shapes = ((10, 11, 12), (13, 14, 15))
    regions = [1,3]

    img1 = testing.generate_labeled_regions(shape = shapes[0],
                                            n_regions =regions[0] )
    img2 = testing.generate_labeled_regions(shape = shapes[1],
                                            n_regions =regions[1] )

    for create_files in (False, True):
        with testing.write_tmp_imgs(img1, img2,
                                    create_files=create_files) as imgs:
            # List of images as input
            out = largest_connected_component_img(imgs)
            assert_true(isinstance(out, list))
            assert_true(len(out) == 2)
            for o, s in zip(out, shapes):
                assert_true(o.shape == (s))

            # Single image as input
            out = largest_connected_component_img(imgs[0])
            assert_true(isinstance(out, Nifti1Image))
            assert_true(out.shape == (shapes[0] ))


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
