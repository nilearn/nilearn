"""
Regions of interest extraction and handling.
"""
# License: simplified BSD

#import warnings
import numpy as np
import scipy.linalg as splinalg

#from . import utils


def apply_roi(timeseries, regions, normalize_regions=False):
    """Compute timeseries for regions of interest.

    This function takes timeseries as parameters (masked data).

    This function solves the inverse problem of finding the
    matrix timeseries_roi such that:

    timeseries = np.dot(timeseries_roi, regions.T)

    The direct problem is handled by unapply_roi().

    Parameters
    ==========
    timeseries (2D numpy array)
        Masked data (e.g. output of apply_mask())
        shape: (instant number, voxel number)
    regions (2D numpy array)
        shape: (voxel number, region number)
        Region definitions. One column of this array defines one
        region, given by its weight on each voxel. Voxel numbering
        must match that of `timeseries`.
    normalize_regions (boolean)
        If True, normalize output by
        (regions ** 2).sum(axis=0) / regions.sum(axis=0)
        This factor ensures that if all input timeseries are identical
        in a region, then the corresponding roi-timeseries is exactly the
        same, independent of the region weighting.

    Returns
    =======
    timeseries_roi (2D numpy array)
        Computed timeseries for each region.
        shape: (instant number, region number)
    """

    roi_timeseries = splinalg.lstsq(regions, timeseries.T)[0].T
    if normalize_regions:
        roi_timeseries /= regions.sum(axis=0) / (regions ** 2).sum(axis=0)
    return roi_timeseries


def unapply_roi(timeseries_roi, regions):
    """Recover voxel timeseries from ROI-timeseries.

    See also apply_roi().
    """
    return np.dot(timeseries_roi, regions.T)
