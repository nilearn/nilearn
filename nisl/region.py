"""
Regions of interest extraction and handling.
"""
# Author: Philippe Gervais
# License: simplified BSD

# Vocabulary:
# region array: 4D array, (x, y, z, region number) values are
#               weights (nifti-like)
# region list: list of 3D arrays [(x, y, z)] values are weights
# region labels: 3D array, (x, y, z) values are labels.

# masked regions: 2D array, (region number, voxel number) values are weights.
# apply_mask/unmask to convert to 4D array

import collections

import numpy as np
from scipy import linalg

import nibabel

from . import utils


def apply_regions(voxel_signals, regions, normalize_regions=False):
    """Compute timeseries for regions of interest.

    This function takes timeseries as parameters (masked data).

    This function solves the inverse problem of finding the
    matrix region_signals such that:

    voxel_signals = np.dot(region_signals, regions)

    The direct problem is handled by unapply_regions().

    Parameters
    ==========
    voxel_signals (2D numpy array)
        Masked data (e.g. output of apply_mask())
        shape: (instant number, voxel number)

    regions (2D numpy array)
        shape: (region number, voxel number)
        Region definitions. One row of this array defines one
        region, given by its weight on each voxel. Voxel numbering
        must match that of `timeseries`.

    normalize_regions (boolean)
        If True, normalize output by
        (regions ** 2).sum(axis=0) / regions.sum(axis=0)
        This factor ensures that if all input timeseries are identical
        in a region, then the corresponding regions-timeseries is exactly the
        same, independent of the region weighting.

    Returns
    =======
    region_signals (2D numpy array)
        Computed signals for each region.
        shape: (instant number, region number)
    """
    region_signals = linalg.lstsq(regions.T, voxel_signals.T)[0].T
    if normalize_regions:
        region_signals /= regions.sum(axis=1) / (regions ** 2).sum(axis=1)
    return region_signals


def unapply_regions(region_signals, regions):
    """Recover voxel signals from regions signals.

    Parameters
    ==========
    region_signals (array-like)
        signals for regions. Shape: (instants number, region number)
    regions (array-like)
        regions definition. Shape: (region number, voxel number)

    Returns
    =======
    voxel_series (numpy.ndarray)
        Signals for voxels, masked.
        shape: (instants number, voxel number)

    Notes
    =====
    See also apply_regions().
    """
    # FIXME: turn second argument into niimg
    return np.dot(region_signals, regions)


def _regions_are_overlapping_masked(regions_masked):
    """Predicate telling if any two regions are overlapping.

    Parameters
    ==========
    regions_masked (numpy.ndarray)
        shape (region number, voxel number). Values are weights.

    """
    count = np.where(regions_masked != 0, 1, 0).sum(axis=0)
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
        count += region != 0

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


def _regions_labels_to_array(region_labels, background_label=0, dtype=np.bool):
    """Convert regions expressed as labels to a 4D array with weights.

    Parameters
    ==========
    region_labels (numpy.ndarray)
        3D array, with integer labels as values

    background_label (integer)
        label corresponding to background. No region is output for this label.

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
    nisl.region.regions_array_to_labels
    """

    # TODO: add an option to exclude labels from ouput (useful for background
    # suppression)
    # FIXME: should display a warning if array has not integer values
    labels = list(np.unique(region_labels))
    if background_label is not None:
        try:
            labels.remove(background_label)
        except ValueError:  # value not in list
            pass

    if region_labels.ndim == 4 and region_labels.shape[3] != 1:
        raise ValueError("input array containing labels must be 3D, "
                         "you provided this shape: %s"
                         % str(region_labels.shape))
    regions_array = np.zeros(region_labels.shape + (len(labels), ),
                             dtype=dtype)

    for n, label in enumerate(labels):
        regions_array[..., n] = np.where(region_labels == label, 1, 0)

    return regions_array, labels


def _regions_array_to_labels(region_array, labels=None, background_label=0):
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
    nisl.region.regions_labels_to_array

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


def _regions_array_to_list(regions_array, copy=False):
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


def _regions_list_to_array(regions_list):
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


def _regions_list_to_labels(regions_list, labels=None, background_label=0):
    """Convert a list of regions into an array of labels.

    If regions overlap, they are made non-overlapping (see
    _regions_array_to_labels for details)

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


def _regions_labels_to_list(regions_labels, background_label=0,
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


def regions_to_mask(regions_img, threshold=0., background=0,
                    target_img=None, dtype=np.int8):
    """Merge all regions to give a binary mask.

    A non-zero value in the output means that this point is inside at
    least one region.

    This function can process regions defined as weights or as labels.
    A label image must always be a single image with 3 dimensions. Passing
    a list with only one 3D array does not qualify: it will be considered
    as a single fuzzy region.

    Parameters
    ==========
    regions_img (niimg or list of niimg)
        Regions definition as niimg, in one of the handled formats:
        (4D image, list of 3D images, or 3D label image). All images given
        therein must have the same shape and affine, no resampling is
        performed.

    threshold (float)
        absolute values of weights defining a region must be above this
        threshold to be considered as "inside". Used for fuzzy regions
        definition only (4D and list of 3D arrays). Defaults to zero, as
        it can be exactly represented in floating-point arithmetic.

    background (integer)
        value considered as background for the labeled array case (one 3D
        array) defaults to zero.

    target_img (niimg)
        Image which gives shape and affine to which output must be resampled.
        If None, affine and shape of regions are left unchanged. Resampling is
        performed after mask computation.
        Not implemented yet.

    dtype (numpy.dtype)
        dtype of the output image. This dtype must be storable in a Nifti file.
            (i.e. np.bool is not allowed).

    Returns
    =======
    mask (nibabel.Nifti1Image)
        union of all the regions (binary image)

    See also
    ========
    nisl.masking.intersect_masks
    """

    if isinstance(regions_img, collections.Iterable):
        first = utils.check_niimg(regions_img.__iter__().next())
        affine = first.get_affine()
        shape = utils._get_shape(first)
        if len(shape) != 3:
            raise ValueError("List must contain 3D arrays, {0:d}D "
                             + "array was provided".format(len(shape)))
        output = np.zeros(shape, dtype=dtype)
        del first
        for r in regions_img:  # Load one image at a time to save memory
            niimg = utils.check_niimg(r)
            if utils._get_shape(niimg) != output.shape:
                raise ValueError("Inconsistent shape in input list")
            output[abs(niimg.get_data()) > threshold] = True

    elif isinstance(regions_img, str) or utils.is_a_niimg(regions_img):
        niimg = utils.check_niimg(regions_img)
        shape = utils._get_shape(niimg)
        affine = niimg.get_affine()
        if len(shape) == 4:
            output = np.zeros(shape[:3], dtype=dtype)
            data = niimg.get_data()
            for n in xrange(shape[3]):
                output[abs(data[..., n]) > threshold] = True

        elif len(shape) == 3:  # labels
            output = (niimg.get_data() != background).astype(dtype)

        else:
            raise ValueError(
                "Invalid shape for input array: {0}".format(str(shape)))

    else:
        raise TypeError(
            "Unhandled data type: {0}".format(regions_img.__class__))

    # FIXME: resample if needed
    return nibabel.Nifti1Image(output, affine)
