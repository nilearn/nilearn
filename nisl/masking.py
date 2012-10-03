"""
Utilities to compute a brain mask from EPI images
"""
# Author: Gael Varoquaux
# License: simplified BSD

import numpy as np
from scipy import ndimage

from . import utils

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


def compute_epi_mask(mean_epi, lower_cutoff=0.2, upper_cutoff=0.9,
                 connected=True, opening=True, exclude_zeros=False,
                 ensure_finite=True, verbose=0):
    """
    Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    mean_epi: 3D ndarray
        EPI image, used to compute the mask.
    lower_cutoff : float, optional
        lower fraction of the histogram to be discarded.
    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.
    connected: boolean, optional
        if connected is True, only the largest connect component is kept.
    opening: boolean, optional
        if opening is True, an morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
    ensure_finite: boolean
        If ensure_finite is True, the non-finite values (NaNs and infs)
        found in the images will be replaced by zeros
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    verbose: integer, optional

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask
    """
    if verbose > 0:
       print "EPI mask computation"
    if len(mean_epi.shape) == 4:
        mean_epi = mean_epi.mean(axis=-1)
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = np.floor(lower_cutoff * len(sorted_input))
    upper_cutoff = np.floor(upper_cutoff * len(sorted_input))

    delta = sorted_input[lower_cutoff + 1:upper_cutoff + 1] \
            - sorted_input[lower_cutoff:upper_cutoff]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + lower_cutoff]
                        + sorted_input[ia + lower_cutoff + 1])

    mask = (mean_epi >= threshold)

    if connected:
        mask = _largest_connected_component(mask)
    if opening:
        mask = ndimage.binary_opening(mask.astype(np.int), iterations=2)
    return mask.astype(bool)

###############################################################################
# Time series extraction
###############################################################################


def apply_mask(niimgs, mask_img, dtype=np.float32,
                     smooth=False, ensure_finite=True):
    """ Read the time series from the given sessions filenames, using the mask.

        Parameters
        -----------
        niimgs: list of 3D nifti file names, or 4D nifti filename.
            Files are grouped by session.
        mask: 3d ndarray
            3D mask array: true where a voxel should be used.
        smooth: False or float, optional
            If smooth is not False, it gives the size, in voxel of the
            spatial smoothing to apply to the signal.
        ensure_finite: boolean
            If ensure_finite is True, the non-finite values (NaNs and infs)
            found in the images will be replaced by zeros

        Returns
        --------
        session_series: ndarray
            3D array of time course: (session, voxel, time)
        header: header object
            The header of the first file.

        Notes
        -----
        When using smoothing, ensure_finite should be True: as elsewhere non
        finite values will spread accross the image.
    """
    mask = utils.check_niimg(mask_img)
    mask = mask_img.get_data().astype(np.bool)
    if smooth:
        # Convert from a sigma to a FWHM:
        smooth /= np.sqrt(8 * np.log(2))

    niimgs = utils.check_niimgs(niimgs)
    series = niimgs.get_data()
    affine = niimgs.get_affine()
    # We have 4D data
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        series[np.logical_not(np.isfinite(series))] = 0
    series = series.astype(dtype)
    affine = affine[:3, :3]
    # del data
    if isinstance(series, np.memmap):
        series = np.asarray(series).copy()
    if smooth:
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        smooth_sigma = smooth / vox_size
        for this_volume in np.rollaxis(series, -1):
            this_volume[...] = ndimage.gaussian_filter(this_volume,
                                                    smooth_sigma)
    series = series[mask]
    return series
