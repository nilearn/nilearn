"""
Test for "region" module.
"""
# Author: Ph. Gervais
# License: simplified BSD

import numpy as np
from nose.tools import assert_raises, assert_true, assert_false

import nibabel

from .. import region
from .. import masking
from .. import utils
from ..testing import generate_timeseries, generate_regions_ts
from ..testing import generate_labeled_regions


def test_generate_regions_ts():
    """Minimal testing of generate_regions_ts()"""

    # Check that no regions overlap
    n_voxels = 50
    n_regions = 10
    regions = generate_regions_ts(n_voxels, n_regions, overlap=0)
    assert_true(regions.shape == (n_regions, n_voxels))
    # check: no overlap
    np.testing.assert_array_less((regions > 0).sum(axis=0) - 0.1,
                                 np.ones(regions.shape[1]))
    # check: a region everywhere
    np.testing.assert_array_less(np.zeros(regions.shape[1]),
                                 (regions > 0).sum(axis=0))

    regions = generate_regions_ts(n_voxels, n_regions, overlap=0,
                                  window="hamming")
    assert_true(regions.shape == (n_regions, n_voxels))
    # check: no overlap
    np.testing.assert_array_less((regions > 0).sum(axis=0) - 0.1,
                                 np.ones(regions.shape[1]))
    # check: a region everywhere
    np.testing.assert_array_less(np.zeros(regions.shape[1]),
                                 (regions > 0).sum(axis=0))

    # Check that some regions overlap
    regions = generate_regions_ts(n_voxels, n_regions, overlap=1)
    assert_true(regions.shape == (n_regions, n_voxels))
    assert(np.any((regions > 0).sum(axis=-1) > 1.9))

    regions = generate_regions_ts(n_voxels, n_regions, overlap=1,
                                  window="hamming")
    assert(np.any((regions > 0).sum(axis=-1) > 1.9))


def test_generate_labeled_regions():
    """Minimal testing of generate_labeled_regions"""
    shape = (3, 4, 5)
    n_regions = 10
    regions = generate_labeled_regions(shape, n_regions)
    assert_true(regions.shape == shape)
    assert (len(np.unique(regions.get_data())) == n_regions + 1)


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
    indices = regions.argmax(axis=1)
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
    regions_labels = generate_labeled_regions(shape, n_regions).get_data()

    # FIXME: test dtype argument
    # FIXME: test labels argument
    regions_4D, labels = region._regions_labels_to_array(regions_labels)
    assert(regions_4D.shape == shape + (n_regions,))
    regions_labels_recovered = region._regions_array_to_labels(regions_4D)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    ## 4D array <-> list of 3D arrays
    regions_list = region._regions_array_to_list(regions_4D, copy=False)
    assert (len(regions_list) == regions_4D.shape[-1])
    for n in xrange(regions_4D.shape[-1]):
        np.testing.assert_almost_equal(regions_list[n], regions_4D[..., n])

    regions_4D_recovered = region._regions_list_to_array(regions_list)
    np.testing.assert_almost_equal(regions_4D_recovered, regions_4D)
    # check that arrays in list are views (modifies arrays)
    for n in xrange(regions_4D.shape[-1]):
        regions_list[n][0, 0, 0] = False
        regions_4D[0, 0, 0, n] = True
        assert(regions_list[n][0, 0, 0] == regions_4D[0, 0, 0, n])

    # Assert that data have been copied
    regions_list = region._regions_array_to_list(regions_4D, copy=True)
    for n in xrange(regions_4D.shape[-1]):
        regions_list[n][0, 0, 0] = False
        regions_4D[0, 0, 0, n] = True
        assert(regions_list[n][0, 0, 0] != regions_4D[0, 0, 0, n])

    # Use "labels" argument
    regions_labels += 3
    regions_4D, labels = region._regions_labels_to_array(regions_labels)
    regions_labels_recovered = region._regions_array_to_labels(regions_4D,
                                                           labels=labels)
    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)
    regions_labels_recovered = region._regions_array_to_labels(regions_4D,
                                                 labels=np.asarray(labels))
    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    assert_raises(ValueError, region._regions_array_to_labels,
                  regions_4D, labels=[])

    ## list of 3D arrays <-> labels
    # First case
    regions_labels = generate_labeled_regions(shape, n_regions).get_data()
    assert_true(len(np.unique(regions_labels)) == n_regions + 1)

    regions_list, labels = region._regions_labels_to_list(regions_labels,
                                              background_label=1)
    assert_true(len(labels) == len(regions_list))
    assert_true(len(regions_list) == n_regions)
    assert_true(regions_list[0].shape == regions_labels.shape)
    assert_true(regions_list[0].dtype == np.bool)
    regions_labels_recovered = region._regions_list_to_labels(regions_list,
                                                          labels=labels,
                                                          background_label=1)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    # same with different dtype
    regions_list, labels = region._regions_labels_to_list(regions_labels,
                                                      background_label=1,
                                                      dtype=np.float)
    assert_true(regions_list[0].dtype == np.float)
    regions_labels_recovered = region._regions_list_to_labels(regions_list,
                                                          labels=labels,
                                                          background_label=1)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    # second case: no background
    regions_labels = generate_labeled_regions(shape, n_regions,
                                              labels=range(1, n_regions + 1)
                                              ).get_data()
    regions_list, labels = region._regions_labels_to_list(regions_labels,
                                                      background_label=None)
    assert_true(len(labels) == len(regions_list))
    assert_true(len(regions_list) == n_regions)
    assert_true(regions_list[0].shape == regions_labels.shape)

    regions_labels_recovered = region._regions_list_to_labels(regions_list)
    np.testing.assert_almost_equal(regions_labels, regions_labels_recovered)

    ## check conversion consistency (labels -> 4D -> list -> labels)
    # loop one way, with background
    regions_labels = generate_labeled_regions(shape, n_regions).get_data()
    regions_array, _ = region._regions_labels_to_array(regions_labels)
    regions_list = region._regions_array_to_list(regions_array)
    regions_labels_recovered = region._regions_list_to_labels(regions_list)

    np.testing.assert_almost_equal(regions_labels_recovered, regions_labels)

    # loop the other way
    regions_list, _ = region._regions_labels_to_list(regions_labels)
    regions_array = region._regions_list_to_array(regions_list)
    regions_labels_recovered = region._regions_array_to_labels(regions_array)

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

    regions[:2, 0] = 1  # make regions overlap
    assert_true(region.regions_are_overlapping(regions))

    # 3D volume with labels. No possible overlap.
    regions_labels = generate_labeled_regions(shape, n_regions).get_data()
    assert_false(region.regions_are_overlapping(regions_labels))

    # 4D volume, with weights
    regions_4D, labels = region._regions_labels_to_array(regions_labels)
    assert_false(region.regions_are_overlapping(regions_4D))

    regions_4D[0, 0, 0, :2] = 1  # Make regions overlap
    assert_true(region.regions_are_overlapping(regions_4D))

    # List of arrays
    regions_list = region._regions_array_to_list(regions_4D)
    assert_true(region.regions_are_overlapping(regions_list))

    regions_4D, labels = region._regions_labels_to_array(regions_labels)
    regions_list = region._regions_array_to_list(regions_4D)
    assert_false(region.regions_are_overlapping(regions_list))

    # Bad input
    assert_raises(TypeError, region.regions_are_overlapping, None)
    assert_raises(TypeError, region.regions_are_overlapping,
                  np.zeros((2, 2, 2, 2, 2)))


    # TODO:
    # - check with / without labels
    # - check overlapping / not overlapping
    # - check with / without holes
    # - check length consistency assertion


def test_signals_extraction():
    """Test conversion between signals and images."""

    shape = (8, 9, 10)
    n_instants = 11
    n_regions = 8  # must be 8

    eps = np.finfo(np.float).eps
    # data
    ## rand_gen = np.random.RandomState(1)
    ## data = rand_gen.randn(*(shape + (instants, )))
    affine = np.eye(4)
    ## data_img = nibabel.Nifti1Image(data, affine)
    signals = generate_timeseries(n_instants, n_regions)

    # mask
    mask_data = np.zeros(shape)
    mask_data[1:-1, 1:-1, 1:-1] = 1
    mask_img = nibabel.Nifti1Image(mask_data, affine)

    # labels
    labels_data = np.zeros(shape, dtype=np.int)
    h0 = shape[0] / 2
    h1 = shape[1] / 2
    h2 = shape[2] / 2
    labels_data[:h0, :h1, :h2] = 1
    labels_data[:h0, :h1, h2:] = 2
    labels_data[:h0, h1:, :h2] = 3
    labels_data[:h0, h1:, h2:] = 4
    labels_data[h0:, :h1, :h2] = 5
    labels_data[h0:, :h1, h2:] = 6
    labels_data[h0:, h1:, :h2] = 7
    labels_data[h0:, h1:, h2:] = 8

    labels_img = nibabel.Nifti1Image(labels_data, affine)

    ## Without mask
    # from labels
    data_img = region.img_from_labels(signals, labels_img)
    data = data_img.get_data()
    assert_true(data_img.shape == (shape + (n_instants,)))
    assert_true(np.all(data.std(axis=-1) > 0))

    # There must be non-zero data (safety net)
    assert_true(abs(data).max() > 1e-9)

    # Check that signals in each region are identical in each voxel
    for n in xrange(1, n_regions + 1):
        sigs = data[labels_data == n, :]
        np.testing.assert_almost_equal(sigs[0, :], signals[:, n - 1])
        assert_true(abs(sigs - sigs[0, :]).max() < eps)

    # and back
    signals_r, labels_r = region.signals_from_labels(data_img, labels_img)
    np.testing.assert_almost_equal(signals_r, signals)
    assert_true(labels_r == range(1, 9))

    ## Same thing, with mask.
    data_img = region.img_from_labels(signals, labels_img, mask_img=mask_img)
    assert_true(data_img.shape == (shape + (n_instants,)))

    data = data_img.get_data()
    assert_true(abs(data).max() > 1e-9)
    # Zero outside of the mask
    assert_true(np.all(data[np.logical_not(mask_img.get_data())
                            ].std(axis=-1) < eps)
                )
    # mask labels before checking
    masked_labels_data = labels_data.copy()
    masked_labels_data[np.logical_not(mask_img.get_data())] = 0
    for n in xrange(1, n_regions + 1):
        sigs = data[masked_labels_data == n, :]
        np.testing.assert_almost_equal(sigs[0, :], signals[:, n - 1])
        assert_true(abs(sigs - sigs[0, :]).max() < eps)

    # and back
    signals_r, labels_r = region.signals_from_labels(data_img, labels_img,
                                           mask_img=mask_img)
    np.testing.assert_almost_equal(signals_r, signals)
    assert_true(labels_r == range(1, 9))

    # Test input validation
    data_img = nibabel.Nifti1Image(np.zeros((2, 3, 4, 5)), np.eye(4))

    good_labels_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    bad_labels1_img = nibabel.Nifti1Image(np.zeros((2, 3, 5)), np.eye(4))
    bad_labels2_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), 2 * np.eye(4))

    good_mask_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    bad_mask1_img = nibabel.Nifti1Image(np.zeros((2, 3, 5)), np.eye(4))
    bad_mask2_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), 2 * np.eye(4))
    assert_raises(ValueError, region.signals_from_labels, data_img,
                  bad_labels1_img)
    assert_raises(ValueError, region.signals_from_labels, data_img,
                  bad_labels2_img)
    assert_raises(ValueError, region.signals_from_labels, data_img,
                  bad_labels1_img, mask_img=good_mask_img)
    assert_raises(ValueError, region.signals_from_labels, data_img,
                  bad_labels2_img, mask_img=good_mask_img)
    assert_raises(ValueError, region.signals_from_labels, data_img,
                  good_labels_img, mask_img=bad_mask1_img)
    assert_raises(ValueError, region.signals_from_labels, data_img,
                  good_labels_img, mask_img=bad_mask2_img)


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
    region_img = masking.unapply_mask_to_regions(regions_ts, mask_img)
    regions_ts[0, 0] = 0  # change something
    region_broken_img = masking.unapply_mask_to_regions(regions_ts, mask_img)

    region_labels_img = utils.NislImage(
        region._regions_array_to_labels(region_img.get_data()), affine)
    region_labels_broken_img = utils.NislImage(
        region._regions_array_to_labels(region_broken_img.get_data()), affine)

    region_list_img = [utils.NislImage(data, affine)
                       for data
                       in region._regions_array_to_list(region_img.get_data())]
    region_list_broken_img = [
        utils.NislImage(data, affine)
        for data
        in region._regions_array_to_list(region_broken_img.get_data())]

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
