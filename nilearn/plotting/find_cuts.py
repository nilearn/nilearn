"""
Tools to find activations and cut on maps
"""

# Author: Gael Varoquaux
# License: BSD

import numpy as np
from scipy import ndimage

import nibabel

# Local imports
from .._utils.ndimage import largest_connected_component
from .._utils.extmath import fast_abs_percentile
from .._utils.numpy_conversions import as_ndarray
from ..image.resampling import get_mask_bounds, coord_transform
from ..image.image import _smooth_array

################################################################################
# Functions for automatic choice of cuts coordinates
################################################################################


def find_xyz_cut_coords(img, mask=None, activation_threshold=None):
    """ Find the center of the largest activation connected component.

        Parameters
        -----------
        img : 3D Nifti1Image
            The brain map.
        mask : 3D ndarray, boolean, optional
            An optional brain mask.
        activation_threshold : float, optional
            The lower threshold to the positive activation. If None, the
            activation threshold is computed using the 80% percentile of
            the absolute value of the map.

        Returns
        -------
        x : float
            the x world coordinate.
        y : float
            the y world coordinate.
        z : float
            the z world coordinate.
    """
    data = img.get_data()
    # To speed up computations, we work with partial views of the array,
    # and keep track of the offset
    offset = np.zeros(3)
    # Deal with masked arrays:
    if hasattr(data, 'mask'):
        not_mask = np.logical_not(data.mask)
        if mask is None:
            mask = not_mask
        else:
            mask *= not_mask
        data = np.asarray(data)
    # Get rid of potential memmapping
    data = as_ndarray(data)
    my_map = data.copy()
    if mask is not None:
        slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
        my_map = my_map[slice_x, slice_y, slice_z]
        mask = mask[slice_x, slice_y, slice_z]
        my_map *= mask
        offset += [slice_x.start, slice_y.start, slice_z.start]
    # Testing min and max is faster than np.all(my_map == 0)
    if (my_map.max() == 0) and (my_map.min() == 0):
        return .5 * np.array(data.shape)
    if activation_threshold is None:
        activation_threshold = fast_abs_percentile(my_map[my_map !=0].ravel(),
                                                   80)
    mask = np.abs(my_map) > activation_threshold - 1.e-15
    mask = largest_connected_component(mask)
    slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
    my_map = my_map[slice_x, slice_y, slice_z]
    mask = mask[slice_x, slice_y, slice_z]
    my_map *= mask
    offset += [slice_x.start, slice_y.start, slice_z.start]
    # For the second threshold, we use a mean, as it is much faster,
    # althought it is less robust
    second_threshold = np.abs(np.mean(my_map[mask]))
    second_mask = (np.abs(my_map)>second_threshold)
    if second_mask.sum() > 50:
        my_map *= largest_connected_component(second_mask)
    cut_coords = ndimage.center_of_mass(np.abs(my_map))
    x_map, y_map, z_map = cut_coords + offset

    return coord_transform(x_map, y_map, z_map,
                           img.get_affine())



################################################################################

def _get_auto_mask_bounds(img):
    """ Compute the bounds of the data with an automaticaly computed mask
    """
    data = img.get_data().copy()
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
    # Nifti1Image cannot contain bools
    mask = mask.astype(np.int)
    xmin, xmax, ymin, ymax, zmin, zmax = \
            get_mask_bounds(nibabel.Nifti1Image(mask, affine))
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)

def find_cut_slices(img, direction='z', n_cuts=12, spacing='auto'):
    """ Find 'good' cross-section slicing positions along a given axis.

    Parameters
    ----------
    img: 3D Nifti1Image
        the brain map
    direction: string, optional (default "z")
        sectional direction; possible values are "x", "y", or "z"
    n_cuts: int, optional (default 12)
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

    assert direction in 'xyz'

    axis = 'xyz'.index(direction)

    affine = img.get_affine()
    orig_data = np.abs(img.get_data())
    this_shape = orig_data.shape[axis]
    if n_cuts > this_shape:
        raise ValueError('Too many cuts requested for the data: '
                         'n_cuts=%i, data size=%i' % (n_cuts, this_shape))

    data = orig_data.copy()
    if data.dtype.kind == 'i':
        data = data.astype(np.float)

    data = _smooth_array(data, affine, fwhm='fast')

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

    # Transform this back in image space
    kwargs = dict()
    for name in 'xyz':
        kwargs[name] = np.zeros(len(cut_coords))
    kwargs[direction] = cut_coords
    kwargs['affine'] = affine

    cut_coords = coord_transform(**kwargs)[axis]
    # We need to atleast_1d to make sure that when n_cuts is 1 we do
    # get an iterable
    return np.atleast_1d(cut_coords)
