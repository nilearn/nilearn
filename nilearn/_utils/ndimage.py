"""
N-dimensional image manipulation
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import numpy as np
from scipy import ndimage
from .._utils.compat import _basestring
###############################################################################
# Operating on connected components
###############################################################################


def largest_connected_component(volume):
    """Return the largest connected component of a 3D array.

    Parameters
    -----------
    volume: numpy.ndarray
        3D boolean array indicating a volume.

    Returns
    --------
    volume: numpy.ndarray
        3D boolean array with only one connected component.

    See Also
    --------
    nilearn.image.largest_connected_component_img : To simply operate the
        same manipulation directly on Nifti images.

    Notes
    -----

    **Handling big-endian in given numpy.ndarray**
    This function changes the existing byte-ordering information to new byte
    order, if the given volume has non-native data type. This operation
    is done inplace to avoid big-endian issues with scipy ndimage module.

    """
    if hasattr(volume, "get_data") \
       or isinstance(volume, _basestring):
        raise ValueError('Please enter a valid numpy array. For images use\
                         largest_connected_component_img')
    # Get the new byteorder to handle issues like "Big-endian buffer not
    # supported on little-endian compiler" with scipy ndimage label.
    if not volume.dtype.isnative:
        volume.dtype = volume.dtype.newbyteorder('N')

    # We use asarray to be able to work with masked arrays.
    volume = np.asarray(volume)
    labels, label_nb = ndimage.label(volume)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return volume.astype(np.bool)
    label_count = np.bincount(labels.ravel().astype(np.int))
    # discard the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()


def get_border_data(data, border_size):
    return np.concatenate([
        data[:border_size, :, :].ravel(),
        data[-border_size:, :, :].ravel(),
        data[:, :border_size, :].ravel(),
        data[:, -border_size:, :].ravel(),
        data[:, :, :border_size].ravel(),
        data[:, :, -border_size:].ravel(),
    ])


def _peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                    num_peaks=np.inf):
    """Find peaks in an image, and return them as coordinates or a boolean array.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    NOTE: If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    Parameters
    ----------
    image : ndarray of floats
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`). To find the maximum number of peaks, use
        `min_distance=1`.
    threshold_abs : float
        Minimum intensity of peaks.
    threshold_rel : float
        Minimum intensity of peaks calculated as `max(image) * threshold_rel`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.

    Returns
    -------
    output : ndarray or ndarray of bools
        Boolean array shaped like `image`, with peaks represented by True
        values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in a image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison between
    dilated and original image, peak_local_max function returns the
    coordinates of peaks where dilated image = original.

    This code is mostly adapted from scikit image 0.11.3 release.
    Location of file in scikit image: peak_local_max function in
    skimage.feature.peak
    """
    out = np.zeros_like(image, dtype=np.bool)

    if np.all(image == image.flat[0]):
        return out

    image = image.copy()

    size = 2 * min_distance + 1
    image_max = ndimage.maximum_filter(image, size=size, mode='constant')

    mask = (image == image_max)
    image *= mask

    # find top peak candidates above a threshold
    peak_threshold = max(np.max(image.ravel()) * threshold_rel, threshold_abs)

    # get coordinates of peaks
    coordinates = np.argwhere(image > peak_threshold)

    if coordinates.shape[0] > num_peaks:
        intensities = image.flat[np.ravel_multi_index(coordinates.transpose(),
                                                      image.shape)]
        idx_maxsort = np.argsort(intensities)[::-1]
        coordinates = coordinates[idx_maxsort][:num_peaks]

    nd_indices = tuple(coordinates.T)
    out[nd_indices] = True
    return out
