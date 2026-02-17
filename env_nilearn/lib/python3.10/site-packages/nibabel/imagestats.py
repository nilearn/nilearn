# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Functions for computing image statistics"""

import numpy as np

from nibabel.imageclasses import spatial_axes_first


def count_nonzero_voxels(img):
    """
    Count number of non-zero voxels

    Parameters
    ----------
    img : ``SpatialImage``
        All voxels of the mask should be of value 1, background should have value 0.

    Returns
    -------
    count : int
        Number of non-zero voxels

    """
    return np.count_nonzero(img.dataobj)


def mask_volume(img):
    """Compute volume of mask image.

    Equivalent to "fslstats /path/file.nii -V"

    Parameters
    ----------
    img : ``SpatialImage``
        All voxels of the mask should be of value 1, background should have value 0.


    Returns
    -------
    volume : float
        Volume of mask expressed in mm3.

    Examples
    --------
    >>> import numpy as np
    >>> import nibabel as nb
    >>> mask_data = np.zeros((20, 20, 20), dtype='u1')
    >>> mask_data[5:15, 5:15, 5:15] = 1
    >>> nb.imagestats.mask_volume(nb.Nifti1Image(mask_data, np.eye(4)))
    1000.0
    """
    if not spatial_axes_first(img):
        raise ValueError('Cannot calculate voxel volume for image with unknown spatial axes')
    voxel_volume_mm3 = np.prod(img.header.get_zooms()[:3])
    mask_volume_vx = count_nonzero_voxels(img)
    mask_volume_mm3 = mask_volume_vx * voxel_volume_mm3

    return mask_volume_mm3
