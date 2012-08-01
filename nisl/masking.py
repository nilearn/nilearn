"""
Utilities to compute a brain mask from EPI images
"""

import numpy as np
from scipy import ndimage
import nibabel

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

###############################################################################
# Time series extraction
###############################################################################


def series_from_mask(filenames, mask, dtype=np.float32,
                     smooth=False, ensure_finite=True):
    """ Read the time series from the given sessions filenames, using the mask.

        Parameters
        -----------
        filenames: list of 3D nifti file names, or 4D nifti filename.
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
    assert len(filenames) != 0, (
        'filenames should be a file name or a list of file names, '
        '%s (type %s) was passed' % (filenames, type(filenames)))
    mask = mask.astype(np.bool)
    if smooth:
        # Convert from a sigma to a FWHM:
        smooth /= np.sqrt(8 * np.log(2))
    if isinstance(filenames, basestring):
        # We have a 4D nifti file
        data_file = nibabel.load(filenames)
        header = data_file.get_header()
        series = data_file.get_data()
        if ensure_finite:
            # SPM tends to put NaNs in the data outside the brain
            series[np.logical_not(np.isfinite(series))] = 0
        series = series.astype(dtype)
        affine = data_file.get_affine()[:3, :3]
        del data_file
        if isinstance(series, np.memmap):
            series = np.asarray(series).copy()
        if smooth:
            vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
            smooth_sigma = smooth / vox_size
            for this_volume in np.rollaxis(series, -1):
                this_volume[...] = ndimage.gaussian_filter(this_volume,
                                                        smooth_sigma)
        series = series[mask]
    else:
        nb_time_points = len(list(filenames))
        series = np.zeros((mask.sum(), nb_time_points), dtype=dtype)
        for index, filename in enumerate(filenames):
            data_file = nibabel.load(filename)
            data = data_file.get_data()
            if ensure_finite:
                # SPM tends to put NaNs in the data outside the brain
                data[np.logical_not(np.isfinite(data))] = 0
            data = data.astype(dtype)
            if smooth is not False:
                affine = data_file.get_affine()[:3, :3]
                vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
                smooth_sigma = smooth / vox_size
                data = ndimage.gaussian_filter(data, smooth_sigma)

            series[:, index] = data[mask]
            # Free memory early
            del data
            if index == 0:
                header = data_file.get_header()

    return series, header
