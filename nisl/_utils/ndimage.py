"""
N-dimensional image manipulation
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import numpy as np
from scipy import ndimage


###############################################################################
# Operating on connected components
###############################################################################

def largest_connected_component(volume):
    """Return the largest connected component of a 3D array.

    Parameters
    -----------
    volume: numpy.array
        3D boolean array indicating a volume.

    Returns
    --------
    volume: numpy.array
        3D boolean array with only one connected component.
    """
    # We use asarray to be able to work with masked arrays.
    volume = np.asarray(volume)
    labels, label_nb = ndimage.label(volume)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return volume.astype(np.bool)
    label_count = np.bincount(labels.ravel())
    # discard the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()


