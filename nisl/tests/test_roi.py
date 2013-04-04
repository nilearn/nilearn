"""
Test for roi module.
"""
# License: simplified BSD

import numpy as np
from .. import roi


def generate_timeseries(instant_number, feature_number,
                        randgen=np.random.RandomState(0)):
    """Generate some random timeseries. """
    return randgen.randn(instant_number, feature_number)


def generate_hb_regions(feature_number, region_number,
                        randgen=np.random.RandomState(0),):
    """Generate some non-overlapping regions."""

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
        regions[boundaries[n]:boundaries[n + 1], n] = 1

    # Check that no regions overlap
    np.testing.assert_almost_equal(regions.sum(axis=-1),
                                   np.ones(regions.shape[0]))
    return regions


def test_apply_roi():
    instant_number = 101
    voxel_number = 54
    region_number = 11

    # First generate signal based on _non-overlapping_ regions, then do
    # the reverse. Check that the starting signals are recovered.
    ts_roi = generate_timeseries(instant_number, region_number)
    regions = generate_hb_regions(voxel_number, region_number)
    timeseries = roi.unapply_roi(ts_roi, regions)
    recovered = roi.apply_roi(timeseries, regions)

    np.testing.assert_almost_equal(ts_roi, recovered, decimal=14)

    # Extract one timeseries from each region, they must be identical
    # to ROI timeseries.
    indices = regions.argmax(axis=0)
    region_signals = timeseries.T[indices].T
    np.testing.assert_almost_equal(recovered, region_signals)
