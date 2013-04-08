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

# region array: 4D array, (x, y, z, region number) values are weights (nifti-like)
# region list: list of 3D arrays [(x, y, z)] values are weights
# region labels: 3D array, (x, y, z) values are labels.

# masked regions: 2D array, (region number, voxel number) values are weights.
# apply_mask/unmask to convert to 4D array


def _regions_are_overlapping_masked(regions_masked):
    """Predicate telling if any two regions are overlapping.

    Parameters
    ==========
    regions_masked (numpy.ndarray)
        shape (region number, voxel number). Values are weights.

    """
    count = np.where(regions_masked > 0, 1, 0).sum(axis=1)
    return np.any(count > 1)


def _regions_are_overlapping_array(regions_array):
    regions = np.where(regions_array != 0, 1, 0)
    return regions.sum(axis=-1).max() > 1


def _regions_are_overlapping_list(regions_list):
    """Predicate telling if any two regions are overlapping.

    Parameters
    ==========
    regions_list (list)
        Region definition as a list of 3D arrays

    Returns
    =======
    predicate (boolean)
        True if two regions are overlapping, False otherwise.
    """

    count = np.zeros(regions_list[0].shape)
    for region in regions_list:
        count += region > 0

    return count.max() > 1


def regions_are_overlapping(regions):
    """Predicate telling if any two regions are overlapping."""

    if isinstance(regions, list):
        predicate = _regions_are_overlapping_list(regions)
    elif isinstance(regions, np.ndarray):
        if regions.ndim == 2:
            predicate = _regions_are_overlapping_masked(regions)
        elif regions.ndim == 3:  # labeled array
            predicate = False
        elif regions.ndim == 4:  # arrays of 3D-arrays
            predicate = _regions_are_overlapping_array(regions)
        else:
            raise TypeError("input array may have 2 to 4 dimensions.")

    else:
        raise TypeError("type not understood")

    return predicate


def regions_labels_to_array(region_labels, dtype=np.bool):
    """Convert regions expressed as labels to a 4D array with weights.

    Parameters
    ==========
    region_labels (numpy.ndarray)
        3D array, with integer labels as values

    dtype (numpy dtype)
        dtype of the returned array. Defaults to boolean.

    Returns
    =======
    region_array (numpy.ndarray)
        shape: region_labels.shape + (len(labels),)
    labels (list)
        labels[i] is the label for the region region_array[..., i]

    See also
    ========
    nisl.roi.regions_array_to_labels
    """

    # FIXME: should display a warning if array has not integer values
    labels = np.unique(region_labels)
    regions_array = np.zeros(region_labels.shape + (len(labels), ),
                             dtype=dtype)

    for n, label in enumerate(labels):
        regions_array[..., n] = np.where(region_labels == label, 1, 0)

    return regions_array, list(labels)


def regions_array_to_labels(region_array, labels=None, background_label=0):
    """Convert regions expressed as a 4D array into a labeled array.

    region_array can contain overlapping regions, but return values cannot.
    A point in space is always labeled with the region with highest weight.
    Zero is "outside".

    Parameters
    ==========
    region_array (numpy.ndarray)
        shape: region_labels.shape + (len(labels),)
    labels (list or numpy.ndarray, optional)
        labels[i] is the label for the region region_array[..., i]
        Defaults to range(1, region_array.shape(-1) + 1)
    background_label (number)
        label used for points contained in no region.

    Returns
    =======
    region_labels (numpy.ndarray)
        3D array, with integer labels as values

    See also
    ========
    nisl.roi.regions_labels_to_array

    """
    if labels is not None and (len(labels) != region_array.shape[-1]):
        raise ValueError("number of labels and number of regions do not match")

    region_labels = region_array.argmax(axis=-1)
    if labels is None:
        region_labels += 1
    else:
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        region_labels = labels[region_labels]

    # FIXME: use a different scheme for short dtype (like int8) for which
    # an overflow can occur in sum()
    mask = region_array.sum(axis=-1)  # every weight should be positive
    region_labels[mask < 10. * np.finfo(float).eps] = background_label
    return region_labels


def regions_array_to_list(regions_array, copy=False):
    """Convert a numpy array into an list of arrays with one dimension less.

    By default, data are NOT copied. Arrays in list are views of the
    input array. If a copy is required, use the `copy` keyword.

    Parameters
    ==========
    regions_array (numpy.ndarray)
        arbitrary input array, with two or more dimensions.

    Returns
    =======
    regions_list (list on numpy.ndarray)
        list of slices of regions_array such that:
        regions_list[n] == regions_array[..., n]

    """
    regions_list = []
    for n in xrange(regions_array.shape[-1]):
        if copy:
            regions_list.append(regions_array[..., n].copy())
        else:
            regions_list.append(regions_array[..., n])
    return regions_list


def regions_list_to_array(regions_list):
    """Turn a list of arrays into one array with an extra dimension.

    Data are copied.

    Parameters
    ==========
    regions_list (list on numpy.ndarray)
        Every array in the list must have the same shape.

    Returns
    =======
    regions_array (numpy.ndarray)
        Concatenation of all input arrays, along a new last dimension.
    """
    regions_array = np.ndarray(regions_list[0].shape + (len(regions_list),))
    for n, arr in enumerate(regions_list):
        regions_array[..., n] = arr
    return regions_array


def regions_list_to_labels(regions_list, labels=None, background_label=0):
    """Convert a list of regions into an array of labels.

    If regions overlap, they are made non-overlapping (see
    regions_array_to_labels for details)

    Parameters
    ==========
    regions_list (list of 3D numpy.ndarray)
        Every array must have the same shape.

    labels (list of number)
        labels[n] gives the label to use for region defined by regions_list[n]
        If not provided, defaults to range(1, len(regions_list) + 1)

    background_label (number)
        gives the "no region" label

    Returns
    =======
    region_labels (numpy.ndarray)
        3D numpy array, with integer labels as values
        shape: regions_list[0].shape

    See also
    ========
    regions_array_to_labels
    """

    regions_labels = background_label * np.ones(regions_list[0].shape)
    regions_max = np.zeros(regions_list[0].shape)

    if labels is None:
        labels = range(1, len(regions_list) + 1)

    for region, label in zip(regions_list, labels):
        mask = region > regions_max
        regions_labels[mask] = label
        regions_max[mask] = region[mask]

    return regions_labels


def regions_labels_to_list(regions_labels, background_label=0,
                           dtype=np.bool):
    """Convert an array of labels into a list of region arrays.

    Parameters
    ==========
    regions_labels (numpy.ndarray)
        regions defined using labels in an array (a label is a number)
    background_label (number)
        label used in regions_labels for "background". No output is generated
        for this label. "None" means no background.
    dtype (numpy dtype)
        dtype of returned arrays in list (defaults to numpy.bool)

    Returns
    =======
    regions_list (list on numpy.ndarray)
        Region definition using weights.
    labels (list)
        Label used for each region in input array. labels[n] is the label
        used in regions_labels for regions_list[n].
    """

    # get region labels
    labels = list(np.unique(regions_labels))
    if background_label is not None:
        # FIXME: catch ValueError (raised when background_label is not
        # in labels)
        labels.remove(background_label)

    regions_list = []
    for label in labels:
        region = regions_labels == label
        if dtype != np.bool:
            region = region.astype(dtype)  # copy
        regions_list.append(region)
    return regions_list, labels

