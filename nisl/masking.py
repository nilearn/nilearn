"""
Utilities to compute a brain mask from EPI images
"""

import numpy as np
from scipy import ndimage


###############################################################################
# Operating on connect component
###############################################################################


def _largest_connected_component(mask):
    """
    Return the largest connected component of a 3D mask array.

    Parameters
    -----------
    mask: 3D boolean array
        3D array indicating a mask.

    Returns
    --------
    mask: 3D boolean array
        3D array indicating a mask, with only one connected component.
    """
    # We use asarray to be able to work with masked arrays.
    mask = np.asarray(mask)
    labels, label_nb = ndimage.label(mask)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return mask.astype(np.bool)
    label_count = np.bincount(labels.ravel())
    # discard 0 the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()


###############################################################################
# Utilities to compute masks
###############################################################################


def compute_mask(epi_img, lower_cutoff=0.2, upper_cutoff=0.9,
                 connected=True, exclude_zeros=False):
    """
    Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    epi_img : 3D ndarray
        EPI image, used to compute the mask.
    lower_cutoff : float, optional
        lower fraction of the histogram to be discarded.
    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.
    connected: boolean, optional
        if connected is True, only the largest connect component is kept.
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask
    """
    if len(epi_img.shape) == 4:
        epi_img = epi_img.mean(axis=-1)
    sorted_input = np.sort(np.ravel(epi_img))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = np.floor(lower_cutoff * len(sorted_input))
    upper_cutoff = np.floor(upper_cutoff * len(sorted_input))

    delta = sorted_input[lower_cutoff + 1:upper_cutoff + 1] \
            - sorted_input[lower_cutoff:upper_cutoff]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + lower_cutoff]
                        + sorted_input[ia + lower_cutoff + 1])

    mask = (epi_img >= threshold)

    if connected:
        mask = _largest_connected_component(mask)

    return mask.astype(bool)
