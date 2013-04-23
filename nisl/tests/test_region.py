"""
Test for "region" module.
"""
# Author: Ph. Gervais
# License: simplified BSD

import numpy as np
import scipy.signal as spsignal
from nose.tools import assert_raises, assert_true

import nibabel

from .. import region
from .. import masking
from .. import utils


def generate_timeseries(n_instants, n_features,
                        randgen=None):
    """Generate some random timeseries. """
    if randgen is None:
        randgen = np.random.RandomState(0)
    return randgen.randn(n_instants, n_features)


def generate_regions_ts(n_features, n_regions,
                        overlap=0,
                        randgen=None,
                        window="boxcar"):
    """Generate some regions.

    Parameters
    ==========
    overlap (int)
        Number of overlapping voxels between two regions (more or less)
    window (str)
        Name of a window in scipy.signal. e.g. "hamming".

    Returns
    =======
    regions (numpy.ndarray)
        timeseries representing regions.
        shape (n_features, n_regions)
    """

    if randgen is None:
        randgen = np.random.RandomState(0)
    if window is None:
        window = "boxcar"

    assert(n_features > n_regions)

    # Compute region boundaries indices.
    # Start at 1 to avoid getting an empty region
    boundaries = np.zeros(n_regions + 1)
    boundaries[-1] = n_features
    boundaries[1:-1] = randgen.permutation(range(1, n_features)
                                           )[:n_regions - 1]
    boundaries.sort()

    regions = np.zeros((n_features, n_regions))
    overlap_end = int((overlap + 1) / 2)
    overlap_start = int(overlap / 2)
    for n in xrange(len(boundaries) - 1):
        start = max(0, boundaries[n] - overlap_start)
        end = min(n_features, boundaries[n + 1] + overlap_end)
        win = spsignal.get_window(window, end - start)
        win /= win.mean()  # unity mean
        regions[start:end, n] = win

    return regions


def test_generate_regions_ts():
    """Minimal testing of generate_regions_ts()"""

    # Check that no regions overlap
    regions = generate_regions_ts(50, 10, overlap=0)
    np.testing.assert_array_less((regions > 0).sum(axis=-1) - 0.1,
                                 np.ones(regions.shape[0]))

    regions = generate_regions_ts(50, 10, overlap=0, window="hamming")
    np.testing.assert_array_less((regions > 0).sum(axis=-1) - 0.1,
                                 np.ones(regions.shape[0]))

    # Check that some regions overlap
    regions = generate_regions_ts(50, 10, overlap=1)
    assert(np.any((regions > 0).sum(axis=-1) > 1.9))

    regions = generate_regions_ts(50, 10, overlap=1, window="hamming")
    assert(np.any((regions > 0).sum(axis=-1) > 1.9))


def generate_labeled_regions(shape, n_regions, randgen=None, labels=None):
    """Generate a 3D volume with labeled regions.

    Parameters
    ==========
    shape (tuple)
        shape of returned array
    n_regions (integer)
        number of regions to generate. By default (if "labels" is None),
        add a background with value zero.
    labels (iterable)
        labels to use for each zone. If provided, n_regions is unused.
    randgen (numpy.random.RandomState object)
        random generator to use for generation.
    """
    n_voxels = shape[0] * shape[1] * shape[2]
    if labels is None:
        labels = xrange(0, n_regions + 1)
        n_regions += 1
    else:
        n_regions = len(labels)

    regions = generate_regions_ts(n_voxels, n_regions, randgen=randgen)
    # replace weights with labels
    for n, col in zip(labels, regions.T):
        col[col > 0] = n
    return masking.unmask(regions.sum(axis=1), np.ones(shape, dtype=np.bool))


def test_apply_regions():
    n_instants = 101
    n_voxels = 54
    n_regions = 11

    # First generate signal based on _non-overlapping_ regions, then do
    # the reverse. Check that the starting signals are recovered.
    ts_roi = generate_timeseries(n_instants, n_regions)
    regions_nonflat = generate_regions_ts(n_voxels, n_regions,
                                          window="hamming")
    regions = np.where(regions_nonflat > 0, 1, 0)
    timeseries = region.unapply_regions(ts_roi, regions)
    recovered = region.apply_regions(timeseries, regions)

    np.testing.assert_almost_equal(ts_roi, recovered, decimal=14)

    # Extract one timeseries from each region, they must be identical
    # to ROI timeseries.
    indices = regions.argmax(axis=0)
    recovered2 = region.apply_regions(timeseries, regions_nonflat,
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
    n_regions = 11

    ## labels <-> 4D array
    regions_labels = generate_labeled_regions(shape, n_regions)

    # FIXME: test dtype argument
    # FIXME: test labels argument
    regions_4D, labels = region.regions_labels_to_array(regions_labels)
    assert(regions_4D.shape == shape + (n_regions,))
    regions_labels_recovered = region.regions_array_to_labels(regions_4D)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    ## 4D array <-> list of 3D arrays
    regions_list = region.regions_array_to_list(regions_4D, copy=False)
    assert (len(regions_list) == regions_4D.shape[-1])
    for n in xrange(regions_4D.shape[-1]):
        np.testing.assert_almost_equal(regions_list[n], regions_4D[..., n])

    regions_4D_recovered = region.regions_list_to_array(regions_list)
    np.testing.assert_almost_equal(regions_4D_recovered, regions_4D)
    # check that arrays in list are views (modifies arrays)
    for n in xrange(regions_4D.shape[-1]):
        regions_list[n][0, 0, 0] = False
        regions_4D[0, 0, 0, n] = True
        assert(regions_list[n][0, 0, 0] == regions_4D[0, 0, 0, n])

    # Assert that data have been copied
    regions_list = region.regions_array_to_list(regions_4D, copy=True)
    for n in xrange(regions_4D.shape[-1]):
        regions_list[n][0, 0, 0] = False
        regions_4D[0, 0, 0, n] = True
        assert(regions_list[n][0, 0, 0] != regions_4D[0, 0, 0, n])

    # Use "labels" argument
    regions_labels += 3
    regions_4D, labels = region.regions_labels_to_array(regions_labels)
    regions_labels_recovered = region.regions_array_to_labels(regions_4D,
                                                           labels=labels)
    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)
    regions_labels_recovered = region.regions_array_to_labels(regions_4D,
                                                 labels=np.asarray(labels))
    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    assert_raises(ValueError, region.regions_array_to_labels,
                  regions_4D, labels=[])

    ## list of 3D arrays <-> labels
    # First case
    regions_labels = generate_labeled_regions(shape, n_regions)
    assert_true(len(np.unique(regions_labels)) == n_regions + 1)

    regions_list, labels = region.regions_labels_to_list(regions_labels,
                                              background_label=1)
    assert_true(len(labels) == len(regions_list))
    assert_true(len(regions_list) == n_regions)
    assert_true(regions_list[0].shape == regions_labels.shape)
    assert_true(regions_list[0].dtype == np.bool)
    regions_labels_recovered = region.regions_list_to_labels(regions_list,
                                                          labels=labels,
                                                          background_label=1)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    # same with different dtype
    regions_list, labels = region.regions_labels_to_list(regions_labels,
                                                      background_label=1,
                                                      dtype=np.float)
    assert_true(regions_list[0].dtype == np.float)
    regions_labels_recovered = region.regions_list_to_labels(regions_list,
                                                          labels=labels,
                                                          background_label=1)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    # second case: no background
    regions_labels = generate_labeled_regions(shape, n_regions,
                                              labels=range(1, n_regions + 1))
    regions_list, labels = region.regions_labels_to_list(regions_labels,
                                                      background_label=None)
    assert_true(len(labels) == len(regions_list))
    assert_true(len(regions_list) == n_regions)
    assert_true(regions_list[0].shape == regions_labels.shape)

    regions_labels_recovered = region.regions_list_to_labels(regions_list)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    ## check conversion consistency (labels -> 4D -> list -> labels)
    # loop one way, with background
    regions_labels = generate_labeled_regions(shape, n_regions)
    regions_array, _ = region.regions_labels_to_array(regions_labels)
    regions_list = region.regions_array_to_list(regions_array)
    regions_labels_recovered = region.regions_list_to_labels(regions_list)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    # loop the other way
    regions_list, _ = region.regions_labels_to_list(regions_labels)
    regions_array = region.regions_list_to_array(regions_list)
    regions_labels_recovered = region.regions_array_to_labels(regions_array)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)


def test_regions_are_overlapping():
    """Test of regions_are_overlapping()"""

    shape = (4, 5, 6)
    n_voxels = shape[0] * shape[1] * shape[2]
    n_regions = 11

    # masked array of labels
    regions = generate_regions_ts(n_voxels, n_regions,
                                  window="hamming")
    assert_true(not region.regions_are_overlapping(regions))

    regions[0, :2] = 1  # make regions overlap
    assert_true(region.regions_are_overlapping(regions))

    # 3D volume with labels. No possible overlap.
    regions_labels = generate_labeled_regions(shape, n_regions)
    assert_true(not region.regions_are_overlapping(regions_labels))

    # 4D volume, with weights
    regions_4D, labels = region.regions_labels_to_array(regions_labels)
    assert_true(not region.regions_are_overlapping(regions_4D))

    regions_4D[0, 0, 0, :2] = 1  # Make regions overlap
    assert_true(region.regions_are_overlapping(regions_4D))

    # List of arrays
    regions_list = region.regions_array_to_list(regions_4D)
    assert_true(region.regions_are_overlapping(regions_list))

    regions_4D, labels = region.regions_labels_to_array(regions_labels)
    regions_list = region.regions_array_to_list(regions_4D)
    assert_true(not region.regions_are_overlapping(regions_list))

    # Bad input
    assert_raises(TypeError, region.regions_are_overlapping, None)
    assert_raises(TypeError, region.regions_are_overlapping,
                  np.zeros((2, 2, 2, 2, 2)))


    # TODO:
    # - check with / without labels
    # - check overlapping / not overlapping
    # - check with / without holes
    # - check length consistency assertion


def test_mask_regions():
    """Test masking of regions.
    The procedure is slightly different from that for masking fMRI signals.
    """
    shape = (4, 5, 6)
    n_voxels = shape[0] * shape[1] * shape[2]
    n_regions = 11

    # Generate data
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones(shape, dtype=np.int8), affine)
    regions_ts = generate_regions_ts(n_voxels, n_regions,
                                  overlap=2, window="hamming")

    # 4D volume with weights
    region_array = region.unapply_mask_to_regions(regions_ts, mask_img)
    assert_true(region_array.shape == shape + (n_regions,))
    regions_ts_recovered = region.apply_mask_to_regions(region_array, mask_img)
    np.testing.assert_almost_equal(regions_ts, regions_ts_recovered)

    # list of 3D volumes
    region_list = region.unapply_mask_to_regions(regions_ts, mask_img)
    assert_true(region_list.shape == shape + (n_regions,))
    regions_ts_recovered = region.apply_mask_to_regions(region_list, mask_img)
    np.testing.assert_almost_equal(regions_ts, regions_ts_recovered)

    # array with labels
    region_labels = region.unapply_mask_to_regions(regions_ts, mask_img)
    assert_true(region_labels.shape == shape + (n_regions,))
    regions_ts_recovered = region.apply_mask_to_regions(region_labels,
                                                        mask_img)
    np.testing.assert_almost_equal(regions_ts, regions_ts_recovered)


def test_regions_to_mask():
    """Test of regions_to_mask(): union of regions."""
    shape = (4, 5, 6)
    n_regions = 11

    # Generate data
    affine = np.eye(4)
    mask_data = np.ones(shape, dtype=np.bool)
    mask_data[..., 0] = False
    mask_data[0, ...] = False
    n_voxels = mask_data.sum()
    mask_img = utils.NislImage(mask_data, affine)

    regions_ts = generate_regions_ts(n_voxels, n_regions,
                                     overlap=2, window="hamming")
    region_img = region.unapply_mask_to_regions(regions_ts, mask_img)
    regions_ts[0, 0] = 0  # change something
    region_broken_img = region.unapply_mask_to_regions(regions_ts, mask_img)

    region_labels_img = utils.NislImage(
        region.regions_array_to_labels(region_img.get_data()), affine)
    region_labels_broken_img = utils.NislImage(
        region.regions_array_to_labels(region_broken_img.get_data()), affine)

    region_list_img = [utils.NislImage(data, affine)
                       for data
                       in region.regions_array_to_list(region_img.get_data())]
    region_list_broken_img = [
        utils.NislImage(data, affine)
        for data
        in region.regions_array_to_list(region_broken_img.get_data())]

    # _r stands for "recovered"
    # TODO: list of filenames
    # list of Nifti1Image
    mask_r_img = region.regions_to_mask(region_list_img)
    np.testing.assert_array_equal(mask_data, mask_r_img.get_data())

    mask_r_img = region.regions_to_mask(region_list_broken_img)
    assert_raises(AssertionError,
                  np.testing.assert_array_equal,
                  mask_data, mask_r_img.get_data())

    # one Nifti1Image, 4D array
    mask_r_img = region.regions_to_mask(region_img)
    assert_true(utils.is_a_niimg(mask_r_img))
    np.testing.assert_array_equal(mask_data, mask_r_img.get_data())

    mask_r_img = region.regions_to_mask(region_broken_img)
    assert_true(utils.is_a_niimg(mask_r_img))
    assert_raises(AssertionError,
                  np.testing.assert_array_equal,
                  mask_data, mask_r_img.get_data())

    # TODO: one filename, 4D array
    # TODO: Check effect of "threshold" argument

    # TODO: one filename, 3D file (labels)
    # one Nifti1Image, 3D array (labels)
    mask_r_img = region.regions_to_mask(region_labels_img)
    assert_true(utils.is_a_niimg(mask_r_img))
    np.testing.assert_array_equal(mask_data, mask_r_img.get_data())

    mask_r_img = region.regions_to_mask(region_labels_broken_img)
    assert_true(utils.is_a_niimg(mask_r_img))
    assert_raises(AssertionError,
                  np.testing.assert_array_equal,
                  mask_data, mask_r_img.get_data())

    # TODO: Check effect of "background" argument

    ## Error checking:
    # TODO: list of Nifti1Image with inconsistent shape
    # TODO: list of Nifti1Image with a non-3D image as first element.
