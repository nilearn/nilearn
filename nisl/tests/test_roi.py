"""
Test for roi module.
"""
# License: simplified BSD

import numpy as np
import scipy.signal as spsignal
from .. import roi


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
