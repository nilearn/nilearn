# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Misc tools to find activations and cut on maps
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

import warnings

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import stats, ndimage

# Local imports
from nilearn._utils.ndimage import largest_connected_component
from nilearn.image.resampling import get_bounds


################################################################################
# Functions for automatic choice of cuts coordinates
################################################################################


def coord_transform(x, y, z, affine):
    """ Convert the x, y, z coordinates from one image space to another
        space.

        Parameters
        ----------
        x : number or ndarray
            The x coordinates in the input space
        y : number or ndarray
            The y coordinates in the input space
        z : number or ndarray
            The z coordinates in the input space
        affine : 2D 4x4 ndarray
            affine that maps from input to output space.

        Returns
        -------
        x : number or ndarray
            The x coordinates in the output space
        y : number or ndarray
            The y coordinates in the output space
        z : number or ndarray
            The z coordinates in the output space

        Warning: The x, y and z have their Talairach ordering, not 3D
        numy image ordering.
    """
    coords = np.c_[np.atleast_1d(x).flat,
                   np.atleast_1d(y).flat,
                   np.atleast_1d(z).flat,
                   np.ones_like(np.atleast_1d(z).flat)].T
    x, y, z, _ = np.dot(affine, coords)
    return x.squeeze(), y.squeeze(), z.squeeze()


def find_cut_coords(map, mask=None, activation_threshold=None):
    """ Find the center of the largest activation connect component.

        Parameters
        -----------
        map : 3D ndarray
            The activation map, as a 3D numpy array.
        mask : 3D ndarray, boolean, optional
            An optional brain mask.
        activation_threshold : float, optional
            The lower threshold to the positive activation. If None, the
            activation threshold is computed using find_activation.

        Returns
        -------
        x: float
            the x coordinate in voxels.
        y: float
            the y coordinate in voxels.
        z: float
            the z coordinate in voxels.
    """
    # To speed up computations, we work with partial views of the array,
    # and keep track of the offset
    offset = np.zeros(3)
    # Deal with masked arrays:
    if hasattr(map, 'mask'):
        not_mask = np.logical_not(map.mask)
        if mask is None:
            mask = not_mask
        else:
            mask *= not_mask
        map = np.asarray(map)
    my_map = map.copy()
    if mask is not None:
        slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
        my_map = my_map[slice_x, slice_y, slice_z]
        mask = mask[slice_x, slice_y, slice_z]
        my_map *= mask
        offset += [slice_x.start, slice_y.start, slice_z.start]
    # Testing min and max is faster than np.all(my_map == 0)
    if (my_map.max() == 0) and (my_map.min() == 0):
        return .5*np.array(map.shape)
    if activation_threshold is None:
        activation_threshold = stats.scoreatpercentile(
                                    np.abs(my_map[my_map !=0]).ravel(), 80)
    mask = np.abs(my_map) > activation_threshold-1.e-15
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
    return cut_coords + offset


################################################################################

def get_mask_bounds(mask, affine):
    """ Return the world-space bounds occupied by a mask given an affine.

        Notes
        -----

        The mask should have only one connect component.

        The affine should be diagonal or diagonal-permuted.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(mask.shape, affine)
    slices = ndimage.find_objects(mask)
    if len(slices) == 0:
        warnings.warn("empty mask", stacklevel=2)
    else:
        x_slice, y_slice, z_slice = slices[0]
        x_width, y_width, z_width = mask.shape
        xmin, xmax = (xmin + x_slice.start*(xmax - xmin)/x_width,
                    xmin + x_slice.stop *(xmax - xmin)/x_width)
        ymin, ymax = (ymin + y_slice.start*(ymax - ymin)/y_width,
                    ymin + y_slice.stop *(ymax - ymin)/y_width)
        zmin, zmax = (zmin + z_slice.start*(zmax - zmin)/z_width,
                    zmin + z_slice.stop *(zmax - zmin)/z_width)

    return xmin, xmax, ymin, ymax, zmin, zmax


def get_cut_coords(map3d, slicer='z', n_cuts=12, delta_axis=3):
    """
    Heuristically computes 'good' cross-section cut_coords for plot_img(...)
    call.

    Parameters
    ----------
    map3d: 3D array
        the data under consideration
    slicer: string, optional (default "z")
        sectional slicer; possible values are "x", "y", or "z"
    n_cuts: int, optional (default 12)
        number of cuts in the plot
    delta_axis: int, optional (default 3)
        spacing between cuts

    Returns
    -------
    cut_coords: 1D array of length n_cuts
        the computed cut_coords

    Raises
    ------
    AssertionError

    """

    assert slicer in 'xyz'

    axis = 'xyz'.index(slicer)

    axis_axis_max = np.unravel_index(
        np.abs(map3d).argmax(), map3d.shape)[axis]
    axis_axis_min = np.unravel_index(
        (-np.abs(map3d)).argmin(), map3d.shape)[axis]
    axis_axis_min, axis_axis_max = (min(axis_axis_min, axis_axis_max),
                              max(axis_axis_max, axis_axis_min))
    axis_axis_min = min(axis_axis_min, axis_axis_max - delta_axis * n_cuts)

    cut_coords = np.linspace(axis_axis_min, axis_axis_max, n_cuts)

    return cut_coords
