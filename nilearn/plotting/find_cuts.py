"""Tools to find activations and cut on maps."""

# Author: Gael Varoquaux

import numbers
import warnings

import numpy as np
from scipy.ndimage import center_of_mass, find_objects, label

from nilearn.image import get_data

from .._utils import check_niimg_3d, check_niimg_4d
from .._utils.extmath import fast_abs_percentile
from .._utils.ndimage import largest_connected_component
from .._utils.niimg import safe_get_data
from .._utils.numpy_conversions import as_ndarray

# Local imports
from ..image import iter_img, new_img_like, reorder_img
from ..image.image import smooth_array
from ..image.resampling import coord_transform, get_mask_bounds

###############################################################################
# Functions for automatic choice of cuts coordinates
###############################################################################

DEFAULT_CUT_COORDS = (0.0, 0.0, 0.0)


def find_xyz_cut_coords(img, mask_img=None, activation_threshold=None):
    """Find the center of the largest activation connected component.

    Parameters
    ----------
    img : 3D Nifti1Image
        The brain map.

    mask_img : 3D Nifti1Image or None, default=None
        An optional brain mask, provided mask_img should not be empty.

    activation_threshold : :obj:`float` or None, default=None
        The lower threshold to the positive activation. If None, the
        activation threshold is computed using the 80% percentile of
        the absolute value of the map.

    Returns
    -------
    x : :obj:`float`
        The x world coordinate.

    y : :obj:`float`
        The y world coordinate.

    z : :obj:`float`
        The z world coordinate.

    """
    # if a pseudo-4D image or several images were passed (cf. #922),
    # we reduce to a single 3D image to find the coordinates
    img = check_niimg_3d(img)
    data = safe_get_data(img)

    # when given image is empty, return (0., 0., 0.)
    if np.all(data == 0.0):
        warnings.warn(
            "Given img is empty. "
            f"Returning default cut_coords={DEFAULT_CUT_COORDS} instead."
        )
        x_map, y_map, z_map = DEFAULT_CUT_COORDS
        return np.asarray(
            coord_transform(x_map, y_map, z_map, img.affine)
        ).tolist()

    # Retrieve optional mask
    if mask_img is not None:
        mask_img = check_niimg_3d(mask_img)
        mask = safe_get_data(mask_img)
        if not np.allclose(mask_img.affine, img.affine):
            raise ValueError(
                f"Mask affine:\n{mask_img.affine}\n "
                f"is different from img affine:\n{img.affine}"
            )
    else:
        mask = None

    # To speed up computations, we work with partial views of the array,
    # and keep track of the offset
    offset = np.zeros(3)

    # Deal with masked arrays:
    if hasattr(data, "mask"):
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
        # check against empty mask
        if mask.sum() == 0.0:
            warnings.warn(
                "Could not determine cut coords: "
                "Provided mask is empty. "
                "Returning center of mass instead."
            )
            cut_coords = center_of_mass(np.abs(my_map)) + offset
            x_map, y_map, z_map = cut_coords
            return np.asarray(
                coord_transform(x_map, y_map, z_map, img.affine)
            ).tolist()
        slice_x, slice_y, slice_z = find_objects(mask.astype(int))[0]
        my_map = my_map[slice_x, slice_y, slice_z]
        mask = mask[slice_x, slice_y, slice_z]
        my_map *= mask
        offset += [slice_x.start, slice_y.start, slice_z.start]
    # Testing min and max is faster than np.all(my_map == 0)
    if my_map.max() == my_map.min() == 0:
        warnings.warn(
            "Could not determine cut coords: "
            "All values were masked. "
            "Returning center of mass of unmasked data instead."
        )
        # Call center of mass on initial data since my_map is zero.
        # Therefore, do not add offset to cut_coords.
        cut_coords = center_of_mass(np.abs(data))
        x_map, y_map, z_map = cut_coords
        return np.asarray(
            coord_transform(x_map, y_map, z_map, img.affine)
        ).tolist()
    if activation_threshold is None:
        activation_threshold = fast_abs_percentile(
            my_map[my_map != 0].ravel(), 80
        )
    try:
        eps = 2 * np.finfo(activation_threshold).eps
    except ValueError:
        # The above will fail for exact types, eg integers
        eps = 1e-15

    mask = np.abs(my_map) > (activation_threshold - eps)
    # mask may be zero everywhere in rare cases
    if mask.max() == 0:
        warnings.warn(
            "Could not determine cut coords: "
            "All voxels were masked by the thresholding. "
            "Returning the center of mass instead."
        )
        cut_coords = center_of_mass(np.abs(my_map)) + offset
        x_map, y_map, z_map = cut_coords
        return np.asarray(
            coord_transform(x_map, y_map, z_map, img.affine)
        ).tolist()

    mask = largest_connected_component(mask)
    slice_x, slice_y, slice_z = find_objects(mask.astype(int))[0]
    my_map = my_map[slice_x, slice_y, slice_z]
    mask = mask[slice_x, slice_y, slice_z]
    my_map *= mask
    offset += [slice_x.start, slice_y.start, slice_z.start]

    # For the second threshold, we use a mean, as it is much faster,
    # although it is less robust
    second_threshold = np.abs(np.mean(my_map[mask]))
    second_mask = np.abs(my_map) > second_threshold
    if second_mask.sum() > 50:
        my_map *= largest_connected_component(second_mask)
    cut_coords = center_of_mass(np.abs(my_map))
    x_map, y_map, z_map = cut_coords + offset

    # Return as a list of scalars
    return np.asarray(
        coord_transform(x_map, y_map, z_map, img.affine)
    ).tolist()


def _get_auto_mask_bounds(img):
    """Compute the bounds of the data with an automatically computed mask."""
    data = safe_get_data(img)
    affine = img.affine
    if hasattr(data, "mask"):
        # Masked array
        mask = np.logical_not(data.mask)
    else:
        # The mask will be anything that is fairly different
        # from the values in the corners
        edge_value = float(
            data[0, 0, 0]
            + data[0, -1, 0]
            + data[-1, 0, 0]
            + data[0, 0, -1]
            + data[-1, -1, 0]
            + data[-1, 0, -1]
            + data[0, -1, -1]
            + data[-1, -1, -1]
        )
        edge_value /= 6
        mask = np.abs(data - edge_value) > 0.005 * np.ptp(data)
    xmin, xmax, ymin, ymax, zmin, zmax = get_mask_bounds(
        new_img_like(img, mask, affine)
    )
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)


def _transform_cut_coords(cut_coords, direction, affine):
    """Transform cut_coords back in image space.

    Parameters
    ----------
    cut_coords : 1D array of length n_cuts
        The coordinates to be transformed.

    direction : :obj:`str`
        Sectional direction; possible values are "x", "y", or "z".

    affine : 2D array of shape (4, 4)
        The affine for the image.

    Returns
    -------
    cut_coords : 1D array of length n_cuts
       The original cut_coords transformed image space.

    """
    # make kwargs
    axis = "xyz".index(direction)
    kwargs = {name: np.zeros(len(cut_coords)) for name in "xyz"}
    kwargs[direction] = cut_coords
    kwargs["affine"] = affine

    # We need atleast_1d to make sure that when n_cuts is 1 we do
    # get an iterable
    cut_coords = coord_transform(**kwargs)[axis]
    return np.atleast_1d(cut_coords)


def find_cut_slices(img, direction="z", n_cuts=7, spacing="auto"):
    """Find 'good' cross-section slicing positions along a given axis.

    Parameters
    ----------
    img : 3D Niimg-like object
        See :ref:`extracting_data`.
        The brain map.

    direction : :obj:`str`, default='z'
        Sectional direction; possible values are "x", "y", or "z".

    n_cuts : :obj:`int`, default=7
        Number of cuts in the plot.

    spacing : 'auto' or :obj:`int`, default='auto'
        Minimum spacing between cuts (in voxels, not millimeters)
        if 'auto', the spacing is .5 / n_cuts * img_length.

    Returns
    -------
    cut_coords : 1D array of length n_cuts
        The computed cut_coords.

    Notes
    -----
    This code works by iteratively locating peak activations that are
    separated by a distance of at least 'spacing'. If n_cuts is very
    large and all the activated regions are covered, cuts with a spacing
    less than 'spacing' will be returned.

    Warnings
    --------
    If a non-diagonal img is given. This function automatically reorders
    img to get it back to diagonal. This is to avoid finding same cuts in
    the slices.

    """
    # misc
    if direction not in "xyz":
        raise ValueError(
            f"'direction' must be one of 'x', 'y', or 'z'. Got '{direction}'"
        )
    axis = "xyz".index(direction)
    img = check_niimg_3d(img)
    affine = img.affine
    if not np.all(np.diag(affine)[:3]):
        warnings.warn(
            "A non-diagonal affine is found in the given "
            "image. Reordering the image to get diagonal affine "
            "for finding cuts in the slices.",
            stacklevel=2,
        )
        # resample is set to avoid issues with an image having a non-diagonal
        # affine and rotation.
        img = reorder_img(img, resample="nearest", copy_header=True)
        affine = img.affine
    # note: orig_data is a copy of img._data_cache thanks to np.abs
    orig_data = np.abs(safe_get_data(img))
    this_shape = orig_data.shape[axis]

    if not isinstance(n_cuts, numbers.Number):
        raise ValueError(
            "The number of cuts (n_cuts) must be an integer "
            "greater than or equal to 1. "
            f"You provided a value of n_cuts={n_cuts}."
        )

    # BF issue #575: Return all the slices along and axis if this axis
    # is the display mode and there are at least as many requested
    # n_slices as there are slices.
    if n_cuts > this_shape:
        warnings.warn(
            "Too many cuts requested for the data: "
            f"n_cuts={n_cuts}, data size={this_shape}."
        )
        return _transform_cut_coords(np.arange(this_shape), direction, affine)

    # To smooth data that might be np.int or np.uint,
    # first convert it to float.
    data = orig_data.copy()
    if data.dtype.kind in ("i", "u"):
        data = data.astype(np.float64)

    data = smooth_array(data, affine, fwhm="fast")

    # to control floating point error problems
    # during given input value "n_cuts"
    epsilon = np.finfo(np.float32).eps
    difference = abs(round(n_cuts) - n_cuts)
    if round(n_cuts) < 1.0 or difference > epsilon:
        message = (
            f"Image has {this_shape} slices in direction {direction}. "
            "Therefore, the number of cuts "
            f"must be between 1 and {this_shape}. "
            f"You provided n_cuts={n_cuts}."
        )
        raise ValueError(message)
    else:
        n_cuts = int(round(n_cuts))

    if spacing == "auto":
        spacing = max(int(0.5 / n_cuts * data.shape[axis]), 1)

    slices = [slice(None, None), slice(None, None), slice(None, None)]

    cut_coords = []

    for _ in range(n_cuts):
        # Find a peak
        max_along_axis = np.unravel_index(np.abs(data).argmax(), data.shape)[
            axis
        ]

        # cancel out the surroundings of the peak
        start = max(0, max_along_axis - spacing)
        stop = max_along_axis + spacing
        slices[axis] = slice(start, stop)
        # We don't actually fully zero the neighborhood, to avoid ending
        # up with fully zeros if n_cuts is too big: we can do multiple
        # passes on the data
        data[tuple(slices)] *= 1.0e-3

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
            slice_middle = int(
                0.5 * (cut_coords[middle_idx] + cut_coords[middle_idx + 1])
            )
            if slice_middle not in cut_coords:
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


def find_parcellation_cut_coords(
    labels_img,
    background_label=0,
    return_label_names=False,
    label_hemisphere="left",
):
    """Return coordinates of center of mass of 3D :term:`parcellation` atlas.

    Parameters
    ----------
    labels_img : 3D Nifti1Image
        A brain :term:`parcellation` atlas with specific mask labels for each
        parcellated region.

    background_label : :obj:`int`, default=0
        Label value used in labels_img to represent background.

    return_label_names : :obj:`bool`, default=False
        Returns list of labels.

    label_hemisphere : 'left' or 'right', default='left'
        Choice of hemisphere to compute label center coords for.
        Applies only in cases where atlas labels are lateralized.
        Eg. Yeo or Harvard Oxford atlas.

    Returns
    -------
    coords : numpy.ndarray of shape (n_labels, 3)
        Label regions cut coordinates in image space (mm).

    labels_list : :obj:`list`, optional
        Label region. Returned only when return_label_names is True.

    See Also
    --------
    nilearn.plotting.find_probabilistic_atlas_cut_coords : For coordinates
        extraction on probabilistic atlases (4D) (Eg. MSDL atlas)

    """
    # check label_hemisphere input
    if label_hemisphere not in ["left", "right"]:
        raise ValueError(
            f"Invalid label_hemisphere name:{label_hemisphere}.\n"
            "Should be one of these 'left' or 'right'."
        )
    # Grab data and affine
    labels_img = reorder_img(check_niimg_3d(labels_img), copy_header=True)
    labels_data = get_data(labels_img)
    labels_affine = labels_img.affine

    # Grab number of unique values in 3d image
    unique_labels = np.unique(labels_data)
    unique_labels = np.delete(unique_labels, background_label)

    # Loop over parcellation labels, grab center of mass and dump into coords
    # list
    coord_list = []
    label_list = []

    for cur_label in unique_labels:
        cur_img = labels_data == cur_label

        # Grab hemispheres separately
        x, y, z = coord_transform(0, 0, 0, np.linalg.inv(labels_affine))
        left_hemi = get_data(labels_img).copy() == cur_label
        right_hemi = get_data(labels_img).copy() == cur_label
        left_hemi[int(x) :] = 0
        right_hemi[: int(x)] = 0

        # Two connected component in both hemispheres
        left_hemi_has_values = np.any(left_hemi)
        right_hemi_all_zero = not np.any(right_hemi)
        if left_hemi_has_values or right_hemi_all_zero:
            if label_hemisphere == "left":
                cur_img = left_hemi.astype(int)
            elif label_hemisphere == "right":
                cur_img = right_hemi.astype(int)

        # Take the largest connected component
        labels, _ = label(cur_img)
        label_count = np.bincount(labels.ravel().astype(int))
        label_count[0] = 0
        component = labels == label_count.argmax()

        # Get parcellation center of mass
        x, y, z = center_of_mass(component)

        # Dump label region and coordinates into a dictionary
        label_list.append(cur_label)
        coord_list.append((x, y, z))

        # Transform coordinates
        coords = [
            coord_transform(i[0], i[1], i[2], labels_affine)
            for i in coord_list
        ]

    if return_label_names:
        return np.array(coords), label_list
    else:
        return np.array(coords)


def find_probabilistic_atlas_cut_coords(maps_img):
    """Return coordinates of center \
       :term:`probabilistic atlas<Probabilistic atlas>` 4D image.

    Parameters
    ----------
    maps_img : 4D Nifti1Image
        A probabilistic brain atlas with probabilistic masks in the fourth
        dimension.

    Returns
    -------
    coords : :class:`numpy.ndarray` of shape (n_maps, 3)
        Label regions cut coordinates in image space (mm).

    See Also
    --------
    nilearn.plotting.find_parcellation_cut_coords : For coordinates
        extraction on parcellations denoted with labels (3D)
        (Eg. Harvard Oxford atlas)

    """
    maps_img = check_niimg_4d(maps_img)
    maps_imgs = iter_img(maps_img)
    coords = [find_xyz_cut_coords(img) for img in maps_imgs]
    return np.array(coords)
