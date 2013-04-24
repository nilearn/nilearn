"""
Utilities to compute a brain mask from EPI images
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import numpy as np
from scipy import ndimage
from nibabel import Nifti1Image
from sklearn.externals.joblib import Parallel, delayed
from . import utils, resampling


def extrapolate_out_mask(data, mask, iterations=1):
    """ Extrapolate values outside of the mask.
    """
    if iterations > 1:
        data, mask = extrapolate_out_mask(data, mask,
                                          iterations=iterations - 1)
    new_mask = ndimage.binary_dilation(mask)
    larger_mask = np.zeros(np.array(mask.shape) + 2, dtype=np.bool)
    larger_mask[1:-1, 1:-1, 1:-1] = mask
    # Use nans as missing value: ugly
    masked_data = np.zeros(larger_mask.shape)
    masked_data[1:-1, 1:-1, 1:-1] = data.copy()
    masked_data[np.logical_not(larger_mask)] = np.nan
    outer_shell = larger_mask.copy()
    outer_shell[1:-1, 1:-1, 1:-1] = new_mask - mask
    outer_shell_x, outer_shell_y, outer_shell_z = np.where(outer_shell)
    extrapolation = list()
    for i, j, k in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
                    (1, 0, 0), (-1, 0, 0)]:
        this_x = outer_shell_x + i
        this_y = outer_shell_y + j
        this_z = outer_shell_z + k
        extrapolation.append(masked_data[this_x, this_y, this_z])

    extrapolation = np.array(extrapolation)
    extrapolation = (np.nansum(extrapolation, axis=0)
                     / np.sum(np.isfinite(extrapolation), axis=0))
    extrapolation[np.logical_not(np.isfinite(extrapolation))] = 0
    new_data = np.zeros_like(masked_data)
    new_data[outer_shell] = extrapolation
    new_data[larger_mask] = masked_data[larger_mask]
    return new_data[1:-1, 1:-1, 1:-1], new_mask


###############################################################################
# Utilities to compute masks
###############################################################################


def compute_epi_mask(mean_epi_img, lower_cutoff=0.2, upper_cutoff=0.9,
                     connected=True, opening=2, exclude_zeros=False,
                     ensure_finite=True, verbose=0):
    """
    Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    mean_epi: 3D or 4D array or nifti-like image
        EPI image, used to compute the mask.

    lower_cutoff : float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: boolean, optional
        if connected is True, only the largest connect component is kept.

    opening: boolean or integer, optional
        if opening is True, an morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
        If opening is an integer 'n', it is performed via 'n' erosion
        followed by 'n' dilations.

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
    mask : 3D nifti-like image
        The brain mask
    """
    if verbose > 0:
        print "EPI mask computation"
    # We suppose that it is a niimg
    # XXX make a is_a_niimgs function ?
    mean_epi_img = utils.check_niimgs(mean_epi_img, accept_3d=True)
    mean_epi = mean_epi_img.get_data()
    if mean_epi.ndim == 4:
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

    mask = mean_epi >= threshold

    if opening:
        opening = int(opening)
        mask = ndimage.binary_erosion(mask, iterations=opening)
    if connected:
        mask = utils.largest_connected_component(mask)
    if opening:
        mask = ndimage.binary_dilation(mask, iterations=opening)
    return Nifti1Image(mask.astype(int), mean_epi_img.get_affine())


def intersect_masks(input_masks, threshold=0.5, connected=True):
    """ Compute intersection of several masks

    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs

    Parameters
    ----------
    input_masks: list of 3D nifti-like images
        3D individual masks

    threshold: float within [0, 1], optional
        gives the level of the intersection.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    connected: bool, optional
        If true, extract the main connected component

    Returns
    -------
        grp_mask, 3D nifti-like image of shape the image shape
    """
    if len(input_masks) == 0:
        raise ValueError('No mask provided for intersection')
    grp_mask = None
    ref_affine = input_masks[0].get_affine()
    ref_shape = input_masks[0].shape
    if threshold > 1:
        raise ValueError('The threshold should be < 1')
    if threshold < 0:
        raise ValueError('The threshold should be > 0')
    threshold = min(threshold, 1 - 1.e-7)

    for this_mask in input_masks:
        if np.any(this_mask.get_affine() != ref_affine):
            raise ValueError("All masks should have the same affine")
        if np.any(this_mask.shape != ref_shape):
            raise ValueError("All masks should have the same shape")
        this_mask = this_mask.get_data().copy().astype(int)
        # Convert the mask in [0, 1] values
        if not len(np.unique(this_mask)) == 2:
            raise ValueError('This mask is not made of 2 values: %s'
                             '. Cannot interpret as true or false'
                             % np.unique(this_mask)
                            )
        this_mask -= this_mask.min()
        this_mask = this_mask != 0
        this_mask = this_mask.astype(np.int)

        if grp_mask is None:
            grp_mask = this_mask
        else:
            # If this_mask is floating point and grp_mask is integer, numpy 2
            # casting rules raise an error for in-place addition. Hence we do
            # it long-hand.
            # XXX should the masks be coerced to int before addition?
            grp_mask += this_mask

    grp_mask = grp_mask > (threshold * len(list(input_masks)))

    if np.any(grp_mask > 0) and connected:
        grp_mask = utils.largest_connected_component(grp_mask)
    grp_mask = grp_mask.astype(int)
    return Nifti1Image(grp_mask, ref_affine)


def compute_multi_epi_mask(session_epi, lower_cutoff=0.2, upper_cutoff=0.9,
                           connected=True, opening=2, threshold=0.5,
                           target_affine=None, target_shape=None,
                           exclude_zeros=False, n_jobs=1, verbose=0):
    """ Compute a common mask for several sessions or subjects of fMRI data.

    Uses the mask-finding algorithms to extract masks for each session
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.


    Parameters
    ----------
    session_files: list 3D or 4D array or Niimgs
        A list of arrays, each item being a subject or a session.

    threshold: float, optional
        the inter-session threshold: the fraction of the
        total number of session in for which a voxel must be in the
        mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    lower_cutoff: float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: boolean, optional
        if connected is True, only the largest connect component is kept.

    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    Returns
    -------
    mask : 3D nifti-like image
        The brain mask
    """
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_epi_mask)(session,
                                  lower_cutoff=lower_cutoff,
                                  upper_cutoff=upper_cutoff,
                                  connected=connected,
                                  opening=opening,
                                  exclude_zeros=exclude_zeros)
        for session in session_epi)

    # Resample if needed
    if target_affine is not None or target_shape is not None:
        masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(resampling.resample_img)
                    (mask, target_affine=target_affine,
                     target_shape=target_shape)
                for mask in masks)

    mask = intersect_masks(masks, connected=connected)
    return mask


###############################################################################
# Time series extraction
###############################################################################

def apply_mask(niimgs, mask_img, dtype=np.float32,
               smooth=None, ensure_finite=True):
    """ Extract time series using specified mask

    Read the time series from the given nifti images or filepaths,
    using the mask.

    Parameters
    -----------
    niimgs (list 4D (ot list of 3D) nifti images)
        Images to be masked.

    mask (3d boolean numpy array)
        3D mask array: true where a voxel should be used.

    smooth (float)
        (optional) Gives the size of the spatial smoothing to apply to
        the signal, in voxels. Implies ensure_finite=True

    ensure_finite (boolean)
        If ensure_finite is True (default), the non-finite values (NaNs and
        infs) found in the images will be replaced by zeros.

    Returns
    --------
    session_series (ndarray)
        2D array of timeseries with shape (time, voxel)

    Notes
    -----
    When using smoothing, ensure_finite is set to True, as non finite
    values will spread accross the image.

    """
    if smooth is not None:
        ensure_finite = True

    mask = utils.check_niimg(mask_img)
    mask = mask.get_data().astype(np.bool)

    niimgs_img = utils.check_niimgs(niimgs)
    affine = niimgs_img.get_affine()[:3, :3]

    data = niimgs_img.get_data()
    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward, especially if smoothing is applied.
    series = utils.as_ndarray(data, dtype=dtype, order="C")
    del data, niimgs_img  # frees a lot of memory

    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        series[np.logical_not(np.isfinite(series))] = 0

    if smooth is not None:
        # Convert from a sigma to a FWHM:
        # Do not use /=, smooth may be a numpy scalar
        smooth = smooth / np.sqrt(8 * np.log(2))
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        smooth_sigma = smooth / vox_size
        for n, s in enumerate(smooth_sigma):
            ndimage.gaussian_filter1d(series, s, output=series, axis=n)

    return series[mask].T


def unmask_3d(X, mask_img):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ==========
    X: numpy array
        Masked data. shape: (samples,)
    mask_img: nifti-like image
        Mask. mask.ndim must be equal to 3.
    
    Return
    ======
    data: 3D nifti-like image
        Unmasked data.
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2])
    """

    mask_img = utils.check_niimg(mask_img)
    mask = mask_img.get_data().astype(bool)
    if X.ndim != 1:
        raise ValueError("X must be a 1-dimensional array")

    data = np.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        dtype=X.dtype)
    data[mask] = X
    return Nifti1Image(data, mask_img.get_affine())


def unmask_nd(X, mask_img):
    """Take masked data and bring them back to n-dimension

    Parameters
    ==========
    X: numpy array
        Masked data. shape: (samples, features)
    mask_img: nifti-like image
        Mask. mask.ndim must be equal to 3.

    Return
    ======
    data: 4D nifti-like image
        Unmasked data.
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
    """

    # Much faster than nisl unmask, and uses three times less memory !
    mask_img = utils.check_niimg(mask_img)
    mask = mask_img.get_data().astype(bool)
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")

    data = np.zeros(mask.shape + (X.shape[0],), dtype=X.dtype)
    data[mask, :] = X.T
    return Nifti1Image(data, mask_img.get_affine())


def unmask(X, mask):
    """Take masked data and bring them back into 3D/4D

    Parameters
    ==========
    X: numpy array (or list of)
        Masked data. shape: (samples #, features #).
        If X is one-dimensional, it is assumed that samples# == 1.
    mask: nifti-like image
        Mask. mask.ndim must be equal to 3, in all cases..

    Return
    ======
    data: nifti-like image (or list of)
        Unmasked data. Depending on the shape of X, data can have
        different shapes:
        - X.ndim = 2:
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
        - X.ndim == 1:
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2])
    """

    if isinstance(X, list):
        ret = []
        for x in X:
            ret.append(unmask(x, mask))  # 1-level recursion
        return ret

    if X.ndim == 2:
        return unmask_nd(X, mask)
    elif X.ndim == 1:
        return unmask_3d(X, mask)
