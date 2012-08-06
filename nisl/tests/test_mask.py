"""
Test the mask-extracting utilities.
"""

from __future__ import with_statement

from nose.tools import assert_true, assert_false

import numpy as np

from ..masking import _largest_connected_component, series_from_mask, \
    compute_mask


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    a[1:3, 1:3, 1:3] = 1
    yield np.testing.assert_equal, a, _largest_connected_component(a)
    b = a.copy()
    b[5, 5, 5] = 1
    yield np.testing.assert_equal, a, _largest_connected_component(a)


def test_mask():
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mask1 = compute_mask(mean_image)
    mask2 = compute_mask(mean_image, exclude_zeros=True)
    # With an array with no zeros, exclude_zeros should not make
    # any difference
    yield np.testing.assert_array_equal, mask1, mask2
    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30))
    mean_image2[:9, :9] = mean_image
    mask3 = compute_mask(mean_image2, exclude_zeros=True)
    yield np.testing.assert_array_equal, mask1, mask3[:9, :9]
    # However, without exclude_zeros, it does
    mask3 = compute_mask(mean_image2)
    yield assert_false, np.allclose(mask1, mask3[:9, :9])


def test_series_from_mask():
    """ Test the smoothing of the timeseries extraction
    """
    # A delta in 3D
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    mask = np.ones((40, 40, 40), dtype=np.bool)
    for affine in (np.eye(4), np.diag((1, 1, -1, 1)),
                    np.diag((.5, 1, .5, 1))):
        series = series_from_mask(data, affine, mask, smooth=9)
        series = np.reshape(series[:, 0], (40, 40, 40))
        vmax = series.max()
        # We are expecting a full-width at half maximum of
        # 9mm/voxel_size:
        above_half_max = series > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                            axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                    9 / np.abs(affine[axis, axis]))

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.NaN
    series = series_from_mask(data, affine, mask, smooth=9)
    assert_true(np.all(np.isfinite(series)))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
