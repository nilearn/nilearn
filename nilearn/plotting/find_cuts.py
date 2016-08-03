"""
Tools to find activations and cut on maps
"""

# Author: Gael Varoquaux
# License: BSD

import warnings
import numbers
import numpy as np
from scipy import ndimage

# Local imports
from .._utils.ndimage import largest_connected_component
from ..image import new_img_like
from .._utils.extmath import fast_abs_percentile
from .._utils.numpy_conversions import as_ndarray
from .._utils import check_niimg_3d
from .._utils.niimg import _safe_get_data
from ..image.resampling import get_mask_bounds, coord_transform
from ..image.image import _smooth_array
from ..masking import apply_mask, unmask, intersect_masks

################################################################################
# Functions for automatic choice of cuts coordinates
################################################################################


def find_xyz_cut_coords(img, mask_img=None, activation_threshold='auto'):
    """ Find the center of mass of the largest activation connected component.

        Parameters
        -----------
        img : 3D Nifti1Image
            The brain map.
        mask_img : 3D Nifti1Image, optional
            An optional brain mask. If provided, it must not be empty.
        activation_threshold : 'auto' or float, optional (default 'auto')
            The lower threshold to the positive activation. If 'auto', the
            activation threshold is computed using the 80% percentile of
            the absolute value of activation.

        Returns
        -------
        x : float
            the x world coordinate.
        y : float
            the y world coordinate.
        z : float
            the z world coordinate.
    """
    # if a pseudo-4D image or several images were passed (cf. #922),
    # we reduce to a single 3D image to find the coordinates
    img = check_niimg_3d(img)
    data = _safe_get_data(img)

    # We have 3 potential masks here:
    # - a mask provided by the user
    # - the activation threshold
    # - if data is a masked_array, the associated mask

    mask_imgs = []
    if mask_img is not None:
        mask_imgs.append[mask_img]

    # Account for numerical noise
    if activation_threshold is None:
        activation_threshold = 1e-7

    # If threshold is auto, it is computed later on masked data
    if activation_threshold is not None and activation_threshold != 'auto':
        mask_imgs.append(new_img_like(img, np.abs(data) >=
                                      activation_threshold))

    if hasattr(data, 'mask'):
        mask_imgs.append(new_img_like(
            np.logical_not(data.mask.astype(np.int8))))

    if len(mask_imgs) > 0:
        mask = intersect_masks(mask_imgs, threshold=1)
        data = unmask(apply_mask([img], mask)[0], mask)
    else:
        # Get rid of potential memmapping
        data = as_ndarray(data)
        # XXX Is that necessary given _saf_get_data?
        data = data.copy()

    # Testing min and max is faster than np.all(my_map == 0)
    if (data.max() == 0) and (data.min() == 0):
        raise ValueError('No non-zero values found (or all masked). '
                         'Cannot compute center of mass.')

    # If thresholding is auto, we compute it now
    non_zero = np.abs(data) > 0.
    if activation_threshold == 'auto':
        activation_threshold = fast_abs_percentile(
            data[data != 0].ravel(), 80)
        non_zero = np.abs(data) >= activation_threshold - 1.e-15
        if non_zero.max() == 0:
            raise ValueError('All voxels were masked by the auto threshold. '
                             'Please disable it or provide a custom value.')

    largest = largest_connected_component(non_zero)
    del non_zero
    slice_x, slice_y, slice_z = ndimage.find_objects(largest)[0]
    data = data[slice_x, slice_y, slice_z]
    largest = largest[slice_x, slice_y, slice_z]
    data *= largest

    # For the second threshold, we use a mean, as it is much faster,
    # althought it is less robust
    # XXX Can somebody explain what this code does?
    second_threshold = np.abs(np.mean(data[largest]))
    second_mask = (np.abs(data) > second_threshold)
    if second_mask.sum() > 50:
        data *= largest_connected_component(second_mask)
    c_x, c_y, c_z = ndimage.center_of_mass(np.abs(data))

    # Return as a list of scalars
    return np.asarray(coord_transform(
        c_x + slice_x.start,
        c_y + slice_y.start,
        c_z + slice_z.start,
        img.get_affine())).tolist()


def _get_auto_mask_bounds(img):
    """ Compute the bounds of the data with an automaticaly computed mask
    """
    data = _safe_get_data(img)
    affine = img.get_affine()
    if hasattr(data, 'mask'):
        # Masked array
        mask = np.logical_not(data.mask)
        data = np.asarray(data)
    else:
        # The mask will be anything that is fairly different
        # from the values in the corners
        edge_value = float(data[0, 0, 0] + data[0, -1, 0]
                            + data[-1, 0, 0] + data[0, 0, -1]
                            + data[-1, -1, 0] + data[-1, 0, -1]
                            + data[0, -1, -1] + data[-1, -1, -1]
                        )
        edge_value /= 6
        mask = np.abs(data - edge_value) > .005*data.ptp()
    xmin, xmax, ymin, ymax, zmin, zmax = \
            get_mask_bounds(new_img_like(img, mask, affine))
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)


def _transform_cut_coords(cut_coords, direction, affine):
    """Transforms cut_coords back in image space

    Parameters
    ----------
    cut_coords: 1D array of length n_cuts
        The coordinates to be transformed.

    direction: string, optional (default "z")
        sectional direction; possible values are "x", "y", or "z"

    affine: 2D array of shape (4, 4)
        The affine for the image.

    Returns
    -------
    cut_coords: 1D array of length n_cuts
       The original cut_coords transformed image space.
    """
    # make kwargs
    axis = 'xyz'.index(direction)
    kwargs = {}
    for name in 'xyz':
        kwargs[name] = np.zeros(len(cut_coords))
    kwargs[direction] = cut_coords
    kwargs['affine'] = affine

    # We need atleast_1d to make sure that when n_cuts is 1 we do
    # get an iterable
    cut_coords = coord_transform(**kwargs)[axis]
    return np.atleast_1d(cut_coords)


def find_cut_slices(img, direction='z', n_cuts=7, spacing='auto'):
    """ Find 'good' cross-section slicing positions along a given axis.

    Parameters
    ----------
    img: 3D Nifti1Image
        the brain map
    direction: string, optional (default "z")
        sectional direction; possible values are "x", "y", or "z"
    n_cuts: int, optional (default 7)
        number of cuts in the plot
    spacing: 'auto' or int, optional (default 'auto')
        minimum spacing between cuts (in voxels, not milimeters)
        if 'auto', the spacing is .5 / n_cuts * img_length

    Returns
    -------
    cut_coords: 1D array of length n_cuts
        the computed cut_coords

    Notes
    -----
    This code works by iteratively locating peak activations that are
    separated by a distance of at least 'spacing'. If n_cuts is very
    large and all the activated regions are covered, cuts with a spacing
    less than 'spacing' will be returned.
    """

    # misc
    if not direction in 'xyz':
        raise ValueError(
            "'direction' must be one of 'x', 'y', or 'z'. Got '%s'" % (
                direction))
    axis = 'xyz'.index(direction)
    affine = img.get_affine()
    orig_data = np.abs(_safe_get_data(img))
    this_shape = orig_data.shape[axis]

    if not isinstance(n_cuts, numbers.Number):
        raise ValueError("The number of cuts (n_cuts) must be an integer "
                         "greater than or equal to 1. "
                         "You provided a value of n_cuts=%s. " % n_cuts)

    # BF issue #575: Return all the slices along and axis if this axis
    # is the display mode and there are at least as many requested
    # n_slices as there are slices.
    if n_cuts > this_shape:
        warnings.warn('Too many cuts requested for the data: '
                      'n_cuts=%i, data size=%i' % (n_cuts, this_shape))
        return _transform_cut_coords(np.arange(this_shape), direction, affine)

    data = orig_data.copy()
    if data.dtype.kind == 'i':
        data = data.astype(np.float)

    data = _smooth_array(data, affine, fwhm='fast')

    # to control floating point error problems
    # during given input value "n_cuts"
    epsilon = np.finfo(np.float32).eps
    difference = abs(round(n_cuts) - n_cuts)
    if round(n_cuts) < 1. or difference > epsilon:
        message = ("Image has %d slices in direction %s. "
                   "Therefore, the number of cuts must be between 1 and %d. "
                   "You provided n_cuts=%s " % (
                       this_shape, direction, this_shape, n_cuts))
        raise ValueError(message)
    else:
        n_cuts = int(round(n_cuts))

    if spacing == 'auto':
        spacing = max(int(.5 / n_cuts * data.shape[axis]), 1)

    slices = [slice(None, None), slice(None, None), slice(None, None)]

    cut_coords = list()

    for _ in range(n_cuts):
        # Find a peak
        max_along_axis = np.unravel_index(np.abs(data).argmax(),
                                          data.shape)[axis]

        # cancel out the surroundings of the peak
        start = max(0, max_along_axis - spacing)
        stop = max_along_axis + spacing
        slices[axis] = slice(start, stop)
        # We don't actually fully zero the neighborhood, to avoid ending
        # up with fully zeros if n_cuts is too big: we can do multiple
        # passes on the data
        data[slices] *= 1.e-3

        cut_coords.append(max_along_axis)

    # We sometimes get duplicated cuts, so we add cuts at the beginning
    # and the end
    cut_coords = np.unique(cut_coords).tolist()
    while len(cut_coords) < n_cuts:
        # Candidates for new cuts:
        slice_below = min(cut_coords) - 2
        slice_above = max(cut_coords) + 2
        candidates = [slice_above]
        # One slice where there is the biggest gap in the existing
        # cut_coords
        if len(cut_coords) > 1:
            middle_idx = np.argmax(np.diff(cut_coords))
            slice_middle = int(.5 * (cut_coords[middle_idx]
                                    + cut_coords[middle_idx + 1]))
            if not slice_middle in cut_coords:
                candidates.append(slice_middle)
        if slice_below >= 0:
            # We need positive slice to avoid having negative
            # indices, which would work, but not the way we think of them
            candidates.append(slice_below)
        best_weight = -10
        for candidate in candidates:
            if candidate >= this_shape:
                this_weight = 0
            else:
                this_weight = np.sum(np.rollaxis(orig_data, axis)[candidate])
            if this_weight > best_weight:
                best_candidate = candidate
                best_weight = this_weight

        cut_coords.append(best_candidate)
        cut_coords = np.unique(cut_coords).tolist()

    cut_coords = np.array(cut_coords)
    cut_coords.sort()

    return _transform_cut_coords(cut_coords, direction, affine)
