"""
Misc tools to find activations and cut on maps
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import ndimage

# Local imports
from .._utils.ndimage import largest_connected_component
from .._utils.fast_maths import fast_abs_percentile


################################################################################
# Functions for automatic choice of cuts coordinates
################################################################################


def find_xyz_cut_coords(map, mask=None, activation_threshold=None):
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
    return cut_coords + offset


################################################################################

def find_cut_slices(map3d, direction='z', n_cuts=12, delta_axis=3):
    """
    Heuristically computes 'good' cross-section cut_coords for plot_img(...)
    call.

    Parameters
    ----------
    map3d: 3D array
        the data under consideration
    direction: string, optional (default "z")
        sectional direction; possible values are "x", "y", or "z"
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

    assert direction in 'xyz'

    axis = 'xyz'.index(direction)

    axis_axis_max = np.unravel_index(
        np.abs(map3d).argmax(), map3d.shape)[axis]
    axis_axis_min = np.unravel_index(
        (-np.abs(map3d)).argmin(), map3d.shape)[axis]
    axis_axis_min, axis_axis_max = (min(axis_axis_min, axis_axis_max),
                              max(axis_axis_max, axis_axis_min))
    axis_axis_min = min(axis_axis_min, axis_axis_max - delta_axis * n_cuts)

    cut_coords = np.linspace(axis_axis_min, axis_axis_max, n_cuts)

    return cut_coords
