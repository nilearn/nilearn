import numpy as np
from scipy import ndimage


###############################################################################
# Operating on connect component
###############################################################################


def largest_cc(mask):
    """ Return the largest connected component of a 3D mask array.

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
# Utilities to calculate masks
###############################################################################


def compute_mask(mean_volume, m=0.2, M=0.9, cc=True,
                    exclude_zeros=False):
    """
    Compute a mask file from fMRI data in 3D or 4D ndarrays.

    Compute and write the mask of an image based on the grey level
    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    m and M of the total image histogram.

    In case of failure, it is usually advisable to increase m.

    Parameters
    ----------
    mean_volume : 3D ndarray
        mean EPI image, used to compute the threshold for the mask.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask
    """
    sorted_input = np.sort(mean_volume.reshape(-1))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    limite_inf = np.floor(m * len(sorted_input))
    limite_sup = np.floor(M * len(sorted_input))

    delta = sorted_input[limite_inf + 1:limite_sup + 1] \
            - sorted_input[limite_inf:limite_sup]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + limite_inf]
                        + sorted_input[ia + limite_inf + 1])

    mask = (mean_volume >= threshold)

    if cc:
        mask = largest_cc(mask)

    return mask.astype(bool)
