"""
Utilities to compute a brain mask from EPI images
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import numpy as np
from scipy import ndimage
from sklearn.externals.joblib import Parallel, delayed

import nibabel

from . import utils
from . import region


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


def compute_epi_mask(mean_epi, lower_cutoff=0.2, upper_cutoff=0.9,
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
    mask : 3D boolean ndarray
        The brain mask
    """
    if verbose > 0:
        print "EPI mask computation"
    if not isinstance(mean_epi, np.ndarray):
        # We suppose that it is a niimg
        # XXX make a is_a_niimgs function ?
        mean_epi = utils.check_niimgs(mean_epi, accept_3d=True).get_data()
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

    mask = (mean_epi >= threshold)

    if opening:
        opening = int(opening)
        mask = ndimage.binary_erosion(mask.astype(np.int),
                                      iterations=opening)
    if connected:
        mask = utils.largest_connected_component(mask)
    if opening:
        mask = ndimage.binary_dilation(mask.astype(np.int),
                                      iterations=opening)
    return mask.astype(bool)


def intersect_masks(input_masks, threshold=0.5, connected=True):
    """ Compute intersection of several masks

    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs

    Parameters
    ----------
    input_masks: list of ndarrays
        3D individual masks

    threshold: float within [0, 1], optional
        gives the level of the intersection.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    connected: bool, optional
        If true, extract the main connected component

    Returns
    -------
        grp_mask, boolean array of shape the image shape
    """
    grp_mask = None
    if threshold > 1:
        raise ValueError('The threshold should be smaller than 1')
    if threshold < 0:
        raise ValueError('The threshold should be greater than 0')
    threshold = min(threshold, 1 - 1.e-7)

    for this_mask in input_masks:
        this_mask = this_mask.copy().astype(np.int)
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

    return grp_mask > 0


def compute_multi_epi_mask(session_epi, lower_cutoff=0.2, upper_cutoff=0.9,
                           connected=True, opening=2, threshold=0.5,
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
    mask : 3D boolean ndarray
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

    mask = intersect_masks(masks, connected=connected)
    return mask


###############################################################################
# Time series extraction
###############################################################################

def apply_mask(niimgs, mask_img, input_type="mri", dtype=np.float32,
               smooth=None, ensure_finite=True):
    """Extract time series using specified mask

    Read the time series from the given nifti images or filepaths,
    using the mask.

    Parameters
    -----------
    niimgs (list 4D (ot list of 3D) nifti images)
        Images to be masked.

    mask_img (niimg)
        3D mask array: True where a voxel should be used.

    input_type (str)
        What niimgs represents. Possible values:  "mri" (default), "regions".
        "mri": niimgs are supposed to be (f)MRI images. Output is always
        intensity in voxel versus image number. "regions": niimgs are supposed
        to define regions (fuzzy or not). Output is always intensity in voxel
        versus region number, even if regions are defined with labels in a
        single 3D image.

    smooth (float)
        (optional) Gives the size of the spatial smoothing to apply to
        the signal, in voxels. Implies ensure_finite=True.

    ensure_finite (boolean)
        If ensure_finite is True (default), the non-finite values (NaNs and
        infs) found in the images will be replaced by zeros.

    Returns
    --------
    session_series (numpy.ndarray)
        2D array of series with shape (image number, voxel number)

    Notes
    -----
    When using smoothing, ensure_finite is set to True, as non-finite
    values would spread accross the image.

    """
    if input_type == "mri":
        output = _apply_mask_fmri(niimgs, mask_img, dtype=dtype, smooth=smooth,
                         ensure_finite=ensure_finite)
    elif input_type == "regions":
        output = _apply_mask_regions(niimgs, mask_img)
    else:
        raise ValueError("Unhandled input type: %s" % input_type)

    return output


def _apply_mask_regions(regions_img_in, mask_img):
    """Convert region definition to timeseries-like representation.

    Parameters
    ==========
    regions_img (niimgs)
        regions definition. Three formats are accepted, with the following
        meanings:
        - 4D volume or list of 3D volume: each slice/3D volume defines a single
        region. Values are interpreted as weights.
        - single 3D volume: values are interpreted as labels, each value
        defining a single region. No overlapping is possible in this case.

    mask_img (niimg)
        mask definition. Value are interpreted as boolean: every non-zero value
        is equivalent to "True", every other to "False". Only voxels containing
        "True" are kept.
        mask.shape must match regions.shape[:3]

    Returns
    =======
    masked_regions (numpy.ndarray)
        Regions in a timeseries-like format. Values are weights.
        shape is (voxel number, region number), where voxel number is the total
        number of voxels inside the mask.

    See also
    ========
    nisl.masking.apply_mask
    nisl.region.apply_regions
    """

    regions_img = utils.check_niimgs(regions_img_in, accept_3d=True)

    if np.any(abs(regions_img.get_affine() - mask_img.get_affine()) > 1e-7):
        raise ValueError("regions and mask affine are different")

    data = regions_img.get_data()
    if data.shape[3] == 1:  # labeled case
        # FIXME: use sparse matrices here.
        regions_img = nibabel.Nifti1Image(
            region._regions_labels_to_array(data[..., 0], dtype=np.int8)[0],
            regions_img.get_affine())

    return _apply_mask_fmri(regions_img, mask_img)


def _apply_mask_fmri(niimgs, mask_img, dtype=np.float32,
                     smooth=None, ensure_finite=True):
    if smooth is not None:
        ensure_finite = True

    mask_img = utils.check_niimg(mask_img)
    mask = mask_img.get_data().astype(np.bool)
    del mask_img

    niimgs_img = utils.check_niimgs(niimgs)
    affine = niimgs_img.get_affine()[:3, :3]

    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward, especially if smoothing is applied.
    data = niimgs_img.get_data()
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


def unapply_mask_to_regions(region_ts, mask_img):
    """Convert regions as timeseries into regions as volume.

    This function is the inverse of apply_mask_regions()

    Parameters
    ==========
    region_ts (array-like)
        shape is (region number, voxel number)
    mask_img (niimg)
        Data mask, must have 3 dimensions.
        The number of non-zero elements in mask must match regions_ts.shape[0]
    affine (array-like, optional)
        Gives the affine that should be returned. This value is not used by
        this function, only in its output.

    Returns
    =======
    region_img (niimg)
        Regions definition as a 4D volume. The affine used in this object is
        that of mask_img.

    See also
    ========
    nisl.region.apply_mask_regions
    """

    mask_img = utils.check_niimg(mask_img)
    region_ts = np.asarray(region_ts)
    if region_ts.ndim != 2:
        raise ValueError("region_ts is not 2D")
    if region_ts.shape[1] != (mask_img.get_data() > 0).sum():
        raise ValueError("Mask definition and number of slices do not match")

    return nibabel.Nifti1Image(unmask(region_ts,
                                      mask_img.get_data().astype(np.bool)),
                               mask_img.get_affine())


def unmask_3D(X, mask):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ==========
    X: numpy array
        Masked data. shape: (samples,)
    mask: numpy array (boolean)
        Mask. mask.ndim must be equal to 3.
    """

    if mask.dtype != np.bool:
        raise ValueError("mask must be a boolean array")
    if X.ndim != 1:
        raise ValueError("X must be a 1-dimensional array")

    data = np.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        dtype=X.dtype)
    data[mask] = X
    return data


def unmask_nD(X, mask):
    """Take masked data and bring them back to n-dimension

    Parameters
    ==========
    X: numpy array
        Masked data. shape: (samples, features)
    mask: numpy array (boolean)
        Mask. mask.ndim must be equal to 3.

    Returns
    =======
    data: 4D numpy array
        Unmasked data.
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
    """

    # Much faster than nisl unmask, and uses three times less memory !
    if mask.dtype != np.bool:
        raise ValueError("mask must be a boolean array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")

    data = np.zeros(mask.shape + (X.shape[0],), dtype=X.dtype)
    data[mask, :] = X.T
    return data


def unmask(X, mask):
    """Take masked data and bring them back into 3D/4D

    This function can be applied to a list of masked data.

    Parameters
    ==========
    X (numpy array, or list of)
        Masked data. shape: (samples number, voxels number).
        If X is one-dimensional, it is assumed that samples number equals one.
    mask  (array-like with boolean values)
        Mask. mask.ndim must be equal to 3, in all cases..

    Returns
    =======
    data (numpy array, or list of)
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
        return unmask_nD(X, mask)
    elif X.ndim == 1:
        return unmask_3D(X, mask)
