"""
Test for roi module.
"""
# License: simplified BSD

import numpy as np
import scipy.signal as spsignal
from .. import roi
from .. import masking
from nose.tools import assert_raises


def generate_timeseries(instant_number, feature_number,
                        randgen=np.random.RandomState(0)):
    """Generate some random timeseries. """
    return randgen.randn(instant_number, feature_number)


def generate_hb_regions(feature_number, region_number,
                        randgen=np.random.RandomState(0),
                        window=None):
    """Generate some non-overlapping regions.

    Parameters
    ==========
    window (str)
        Name of a window in scipy.signal. e.g. "hamming".

    Returns
    =======
    regions (numpy.ndarray)
        shape (feature_number, region_number)
    """

    assert(feature_number > region_number)

    # Compute region boundaries indices.
    # Start at 1 to avoid getting an empty region
    boundaries = np.zeros(region_number + 1)
    boundaries[-1] = feature_number
    boundaries[1:-1] = randgen.permutation(range(1, feature_number)
                                           )[:region_number - 1]
    boundaries.sort()

    regions = np.zeros((feature_number, region_number))
    for n in xrange(len(boundaries) - 1):
        if window is not None:
            win = spsignal.get_window(window,
                                      boundaries[n + 1] - boundaries[n])
            win /= win.mean()  # unity mean
            regions[boundaries[n]:boundaries[n + 1], n] = win
        else:
            regions[boundaries[n]:boundaries[n + 1], n] = 1

    # Check that no regions overlap
    np.testing.assert_array_less((regions > 0).sum(axis=-1) - 0.1,
                                 np.ones(regions.shape[0]))
    return regions


def generate_labeled_regions(shape, region_number):
    """Generate a 3D volume with labeled regions"""
    voxel_number = shape[0] * shape[1] * shape[2]
    regions = generate_hb_regions(voxel_number, region_number)
    # replace weights with labels
    for n, col in enumerate(regions.T):
        col[col > 0] = n + 1
    return masking.unmask(regions.sum(axis=1), np.ones(shape, dtype=np.bool))


def test_apply_roi():
    instant_number = 101
    voxel_number = 54
    region_number = 11

    # First generate signal based on _non-overlapping_ regions, then do
    # the reverse. Check that the starting signals are recovered.
    ts_roi = generate_timeseries(instant_number, region_number)
    regions_nonflat = generate_hb_regions(voxel_number, region_number,
                                          window="hamming")
    regions = np.where(regions_nonflat > 0, 1, 0)
    timeseries = roi.unapply_roi(ts_roi, regions)
    recovered = roi.apply_roi(timeseries, regions)

    np.testing.assert_almost_equal(ts_roi, recovered, decimal=14)

    # Extract one timeseries from each region, they must be identical
    # to ROI timeseries.
    indices = regions.argmax(axis=0)
    recovered2 = roi.apply_roi(timeseries, regions_nonflat,
                               normalize_regions=True)
    region_signals = timeseries.T[indices].T
    np.testing.assert_almost_equal(recovered2, region_signals)


def test_regions_convert():
    """Test of conversion functions between different regions structures.
    Function tested:
    regions_labels_to_array
    regions_array_to_labels
    """

    shape = (4, 5, 6)
    region_number = 11

    ## labels <-> 4D array
    regions_labels = generate_labeled_regions(shape, region_number)

    # FIXME: test dtype argument
    # FIXME: test labels argument
    regions_4D, labels = roi.regions_labels_to_array(regions_labels)
    assert(regions_4D.shape == shape + (region_number,))
    regions_labels_recovered = roi.regions_array_to_labels(regions_4D)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    ## 4D array <-> list of 3D arrays
    regions_list = roi.regions_array_to_list(regions_4D, copy=False)
    assert (len(regions_list) == regions_4D.shape[-1])
    for n in xrange(regions_4D.shape[-1]):
        np.testing.assert_almost_equal(regions_list[n], regions_4D[..., n])

    regions_4D_recovered = roi.regions_list_to_array(regions_list)
    np.testing.assert_almost_equal(regions_4D_recovered, regions_4D)
    # check that arrays in list are views (modifies arrays)
    for n in xrange(regions_4D.shape[-1]):
        regions_list[n][0, 0, 0] = False
        regions_4D[0, 0, 0, n] = True
        assert(regions_list[n][0, 0, 0] == regions_4D[0, 0, 0, n])

    # Assert that data have been copied
    regions_list = roi.regions_array_to_list(regions_4D, copy=True)
    for n in xrange(regions_4D.shape[-1]):
        regions_list[n][0, 0, 0] = False
        regions_4D[0, 0, 0, n] = True
        assert(regions_list[n][0, 0, 0] != regions_4D[0, 0, 0, n])

    # Use "labels" argument
    regions_labels += 3
    regions_4D, labels = roi.regions_labels_to_array(regions_labels)
    regions_labels_recovered = roi.regions_array_to_labels(regions_4D,
                                                           labels=labels)
    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)
    regions_labels_recovered = roi.regions_array_to_labels(regions_4D,
                                                           labels=np.asarray(labels))
    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    assert_raises(ValueError, roi.regions_array_to_labels,
                  regions_4D, labels=[])

    ## list of 3D arrays <-> labels
    # First case
    regions_labels = generate_labeled_regions(shape, region_number)
    regions_list, labels = roi.regions_labels_to_list(regions_labels,
                                              background_label=1)
    assert(len(labels) == len(regions_list))
    assert(len(regions_list) == region_number - 1)
    assert(regions_list[0].shape == regions_labels.shape)
    assert(regions_list[0].dtype == np.bool)
    regions_labels_recovered = roi.regions_list_to_labels(regions_list,
                                                          labels=labels,
                                                          background_label=1)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    # same with different dtype
    regions_list, labels = roi.regions_labels_to_list(regions_labels,
                                                      background_label=1,
                                                      dtype=np.float)
    assert(regions_list[0].dtype == np.float)
    regions_labels_recovered = roi.regions_list_to_labels(regions_list,
                                                          labels=labels,
                                                          background_label=1)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)


    # second case: no background
    regions_labels = generate_labeled_regions(shape, region_number)
    regions_list, labels = roi.regions_labels_to_list(regions_labels,
                                                      background_label=None)
    assert(len(labels) == len(regions_list))
    assert(len(regions_list) == region_number)
    assert(regions_list[0].shape == regions_labels.shape)

    regions_labels_recovered = roi.regions_list_to_labels(regions_list)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    ## check conversion consistency (labels -> 4D -> list -> labels)
    # loop one way
    regions_labels = generate_labeled_regions(shape, region_number)
    regions_array, _ = roi.regions_labels_to_array(regions_labels)
    regions_list = roi.regions_array_to_list(regions_array)
    regions_labels_recovered = roi.regions_list_to_labels(regions_list)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    # loop the other way
    regions_list, _ = roi.regions_labels_to_list(regions_labels,
                                                 background_label=None)
    regions_array = roi.regions_list_to_array(regions_list)
    regions_labels_recovered = roi.regions_array_to_labels(regions_array)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)


def test_regions_are_overlapping():
    """Test of regions_are_overlapping()"""

    shape = (4, 5, 6)
    voxel_number = shape[0] * shape[1] * shape[2]
    region_number = 11


    # masked array of labels
    regions = generate_hb_regions(voxel_number, region_number,
                                  window="hamming")
    assert(not roi.regions_are_overlapping(regions))

    regions[0, :2] = 1  # make regions overlap
    assert(roi.regions_are_overlapping(regions))

    # 3D volume with labels. No possible overlap.
    regions_labels = generate_labeled_regions(shape, region_number)
    assert(not roi.regions_are_overlapping(regions_labels))

    # 4D volume, with weights
    regions_4D, labels = roi.regions_labels_to_array(regions_labels)
    assert(not roi.regions_are_overlapping(regions_4D))

    regions_4D[0, 0, 0, :2] = 1  # Make regions overlap
    assert(roi.regions_are_overlapping(regions_4D))

    # List of arrays
    regions_list = roi.regions_array_to_list(regions_4D)
    assert(roi.regions_are_overlapping(regions_list))

    regions_4D, labels = roi.regions_labels_to_array(regions_labels)
    regions_list = roi.regions_array_to_list(regions_4D)
    assert(not roi.regions_are_overlapping(regions_list))

    # Bad input
    assert_raises(TypeError, roi.regions_are_overlapping, None)
    assert_raises(TypeError, roi.regions_are_overlapping,
                  np.zeros((2, 2, 2, 2, 2)))



    # TODO:
    # - check with / without labels
    # - check overlapping / not overlapping
    # - check with / without holes
    # - check length consistency assertion

