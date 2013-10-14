"""
Utilities to compute a brain mask from EPI images
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import numpy as np
from scipy import ndimage
from nibabel import Nifti1Image
from sklearn.externals.joblib import Parallel, delayed

from . import _utils
from ._utils.ndimage import largest_connected_component
from ._utils.cache_mixin import cache


def _load_mask_img(mask_img, allow_empty=False):
    ''' Check that a mask is valid, ie with two values including 0 and load it.

    Parameters
    ----------
    mask_img: nifti-like image
        The mask to check

    allow_empty: boolean, optional
        Allow loading an empty mask (full of 0 values)

    Returns
    -------
    mask: numpy.ndarray
        boolean version of the mask
    '''
    mask_img = _utils.check_niimg(mask_img)
    mask = mask_img.get_data()
    values = np.unique(mask)

    if len(values) == 1:
        # We accept a single value if it is not 0 (full true mask).
        if values[0] == 0 and not allow_empty:
            raise ValueError('Given mask is invalid because it masks all data')
    elif len(values) == 2:
        # If there are 2 different values, one of them must be 0 (background)
        if not 0 in values:
            raise ValueError('Background of the mask must be represented with'
                             '0. Given mask contains: %s.' % values)
    elif len(values) != 2:
        # If there are more than 2 values, the mask is invalid
        raise ValueError('Given mask is not made of 2 values: %s'
                         '. Cannot interpret as true or false'
                         % values)

    mask = _utils.as_ndarray(mask, dtype=bool)
    return mask, mask_img.get_affine()


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


def compute_epi_mask(epi_img, lower_cutoff=0.2, upper_cutoff=0.9,
                     connected=True, opening=2, exclude_zeros=False,
                     ensure_finite=True,
                     target_affine=None, target_shape=None,
                     memory=None, verbose=0,):
    """
    Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    epi_img: nifti-like image
        EPI image, used to compute the mask. 3D and 4D images are accepted.
        If a 3D image is given, we suggest to use the mean image

    lower_cutoff: float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: bool, optional
        if connected is True, only the largest connect component is kept.

    opening: bool or int, optional
        if opening is True, an morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
        If opening is an integer 'n', it is performed via 'n' erosion
        followed by 'n' dilations.

    ensure_finite: bool
        If ensure_finite is True, the non-finite values (NaNs and infs)
        found in the images will be replaced by zeros

    exclude_zeros: bool, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    memory: instance of joblib.Memory or string
        Used to cache the function call.

    verbose: int, optional

    Returns
    -------
    mask: nibabel.Nifti1Image
        The brain mask (3D image)
    """
    if verbose > 0:
        print "EPI mask computation"
    # We suppose that it is a niimg
    # XXX make a is_a_niimgs function ?

    # Delayed import to avoid circular imports
    from . import image
    input_repr = _utils._repr_niimgs(epi_img)
    epi_img = cache(image.resample_img, memory, ignore=['copy'])(
                                epi_img,
                                target_affine=target_affine,
                                target_shape=target_shape)

    epi_img = _utils.check_niimgs(epi_img, accept_3d=True)
    mean_epi = epi_img.get_data()
    if not mean_epi.ndim in (3, 4):
        raise ValueError('compute_epi_mask expects 3D or 4D '
            'images, but %i dimensions were given (%s)'
            % (mean_epi.ndim, input_repr))
    if mean_epi.ndim == 4:
        mean_epi = mean_epi.mean(axis=-1)
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = int(np.floor(lower_cutoff * len(sorted_input)))
    upper_cutoff = int(np.floor(upper_cutoff * len(sorted_input)))

    delta = sorted_input[lower_cutoff + 1:upper_cutoff + 1] \
        - sorted_input[lower_cutoff:upper_cutoff]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + lower_cutoff]
                       + sorted_input[ia + lower_cutoff + 1])

    mask = mean_epi >= threshold

    if opening:
        opening = int(opening)
        mask = ndimage.binary_erosion(mask, iterations=opening)
    if connected and mask.any():
        mask = largest_connected_component(mask)
    if opening:
        mask = ndimage.binary_dilation(mask, iterations=opening)
    return Nifti1Image(_utils.as_ndarray(mask, dtype=np.int8),
                       epi_img.get_affine())


def intersect_masks(mask_imgs, threshold=0.5, connected=True):
    """ Compute intersection of several masks

    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs

    Parameters
    ----------
    masks_imgs: list of 3D nifti-like images
        3D individual masks with same shape and affine.

    threshold: float, optional
        Gives the level of the intersection, must be within [0, 1].
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    connected: bool, optional
        If true, extract the main connected component

    Returns
    -------
        grp_mask: 3D nifti-like image
            intersection of all masks.
    """
    if len(mask_imgs) == 0:
        raise ValueError('No mask provided for intersection')
    grp_mask = None
    first_mask, ref_affine = _load_mask_img(mask_imgs[0], allow_empty=True)
    ref_shape = first_mask.shape
    if threshold > 1:
        raise ValueError('The threshold should be smaller than 1')
    if threshold < 0:
        raise ValueError('The threshold should be greater than 0')
    threshold = min(threshold, 1 - 1.e-7)

    for this_mask in mask_imgs:
        mask, affine = _load_mask_img(this_mask, allow_empty=True)
        if np.any(affine != ref_affine):
            raise ValueError("All masks should have the same affine")
        if np.any(mask.shape != ref_shape):
            raise ValueError("All masks should have the same shape")

        if grp_mask is None:
            # We use int here because there may be a lot of masks to merge
            grp_mask = _utils.as_ndarray(mask, dtype=int)
        else:
            # If this_mask is floating point and grp_mask is integer, numpy 2
            # casting rules raise an error for in-place addition. Hence we do
            # it long-hand.
            # XXX should the masks be coerced to int before addition?
            grp_mask += mask

    grp_mask = grp_mask > (threshold * len(list(mask_imgs)))

    if np.any(grp_mask > 0) and connected:
        grp_mask = largest_connected_component(grp_mask)
    grp_mask = _utils.as_ndarray(grp_mask, dtype=np.int8)
    return Nifti1Image(grp_mask, ref_affine)


def compute_multi_epi_mask(epi_imgs, lower_cutoff=0.2, upper_cutoff=0.9,
                           connected=True, opening=2, threshold=0.5,
                           target_affine=None, target_shape=None,
                           exclude_zeros=False, n_jobs=1,
                           memory=None, verbose=0):
    """ Compute a common mask for several sessions or subjects of fMRI data.

    Uses the mask-finding algorithms to extract masks for each session
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.

    Parameters
    ----------
    epi_imgs: list of Niimgs
        A list of arrays, each item being a subject or a session.
        3D and 4D images are accepted.
        If 3D images is given, we suggest to use the mean image of each
        session

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

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    memory: instance of joblib.Memory or string
        Used to cache the function call.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    Returns
    -------
    mask : 3D nifti-like image
        The brain mask.
    """
    if len(epi_imgs) == 0:
        raise TypeError('An empty object - %r - was passed instead of an '
                        'image or a list of images' % epi_imgs)
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_epi_mask)(epi_img,
                                  lower_cutoff=lower_cutoff,
                                  upper_cutoff=upper_cutoff,
                                  connected=connected,
                                  opening=opening,
                                  exclude_zeros=exclude_zeros,
                                  target_affine=target_affine,
                                  target_shape=target_shape,
                                  memory=memory)
        for epi_img in epi_imgs)

    mask = intersect_masks(masks, connected=connected, threshold=threshold)
    return mask


###############################################################################
# Time series extraction
###############################################################################

def apply_mask(niimgs, mask_img, dtype=np.float32,
               smoothing_fwhm=None, ensure_finite=True):
    """Extract signals from images using specified mask.

    Read the time series from the given nifti images or filepaths,
    using the mask.

    Parameters
    -----------
    niimgs: list of 4D nifti images
        Images to be masked. list of lists of 3D images are also accepted.

    mask_img: niimg
        3D mask array: True where a voxel should be used.

    smoothing_fwhm: float
        (optional) Gives the size of the spatial smoothing to apply to
        the signal, in voxels. Implies ensure_finite=True.

    ensure_finite: bool
        If ensure_finite is True (default), the non-finite values (NaNs and
        infs) found in the images will be replaced by zeros.

    Returns
    --------
    session_series: numpy.ndarray
        2D array of series with shape (image number, voxel number)

    Notes
    -----
    When using smoothing, ensure_finite is set to True, as non-finite
    values would spread accross the image.
    """
    mask, mask_affine = _load_mask_img(mask_img)
    mask_img = Nifti1Image(_utils.as_ndarray(mask, dtype=np.int8),
                           mask_affine)
    return _apply_mask_fmri(niimgs, mask_img, dtype=dtype,
                            smoothing_fwhm=smoothing_fwhm,
                            ensure_finite=ensure_finite)


def _apply_mask_fmri(niimgs, mask_img, dtype=np.float32,
                     smoothing_fwhm=None, ensure_finite=True):
    """Same as apply_mask().

    The only difference with apply_mask is that some costly checks on mask_img
    are not performed: mask_img is assumed to contain only two different
    values (this is checked for in apply_mask, not in this function).
    """

    mask_img = _utils.check_niimg(mask_img)
    mask_affine = mask_img.get_affine()
    mask_data = _utils.as_ndarray(mask_img.get_data(),
                                             dtype=np.bool)

    if smoothing_fwhm is not None:
        ensure_finite = True

    niimgs_img = _utils.check_niimgs(niimgs)
    affine = niimgs_img.get_affine()[:3, :3]

    if not np.allclose(mask_affine, niimgs_img.get_affine()):
        raise ValueError('Mask affine: \n%s\n is different from img affine:'
                         '\n%s' % (str(mask_affine),
                                   str(niimgs_img.get_affine())))

    if not mask_data.shape == niimgs_img.shape[:3]:
        raise ValueError('Mask shape: %s is different from img shape:%s'
                         % (str(mask_data.shape), str(niimgs_img.shape[:3])))

    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward, especially if smoothing is applied.
    data = niimgs_img.get_data()
    series = _utils.as_ndarray(data, dtype=dtype, order="C",
                                          copy=True)
    del data, niimgs_img  # frees a lot of memory

    _smooth_array(series, affine, fwhm=smoothing_fwhm,
                  ensure_finite=ensure_finite, copy=False)
    return series[mask_data].T


def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of arr.

    Parameters
    ==========
    arr: numpy.ndarray
        4D array, with image number as last dimension. 3D arrays are also
        accepted.

    affine: numpy.ndarray
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).

    fwhm: scalar or numpy.ndarray
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a scalar is given, width is identical on all three directions.
        A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed)

    ensure_finite: bool
        if True, replace every non-finite values (like NaNs) by zero before
        filtering.

    copy: bool
        if True, input array is not modified. False by default: the filtering
        is performed in-place.

    Returns
    =======
    filtered_arr: numpy.ndarray
        arr, filtered.

    Notes
    =====
    This function is most efficient with arr in C order.
    """

    if copy:
        arr = arr.copy()

    # Keep only the scale part.
    affine = affine[:3, :3]

    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0

    if fwhm is not None:
        # Convert from a FWHM to a sigma:
        # Do not use /=, fwhm may be a numpy scalar
        fwhm = fwhm / np.sqrt(8 * np.log(2))
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        sigma = fwhm / vox_size
        for n, s in enumerate(sigma):
            ndimage.gaussian_filter1d(arr, s, output=arr, axis=n)

    return arr


def _unmask_3d(X, mask, order="C"):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ==========
    X: numpy.ndarray
        Masked data. shape: (samples,)

    mask: numpy.ndarray
        Mask. mask.ndim must be equal to 3, and dtype *must* be bool.
    """

    if mask.dtype != np.bool:
        raise ValueError("mask must be a boolean array")
    if X.ndim != 1:
        raise ValueError("X must be a 1-dimensional array")

    data = np.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        dtype=X.dtype, order=order)
    data[mask] = X
    return data


def _unmask_nd(X, mask, order="C"):
    """Take masked data and bring them back to n-dimension

    Parameters
    ==========
    X: numpy.ndarray
        Masked data. shape: (samples, features)

    mask: numpy.ndarray
        Mask. mask.ndim must be equal to 3, and dtype equal to bool.

    Returns
    =======
    data: numpy.ndarray
        Unmasked data.
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
    """

    if mask.dtype != np.bool:
        raise ValueError("mask must be a boolean array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")

    data = np.zeros(mask.shape + (X.shape[0],), dtype=X.dtype, order=order)
    data[mask, :] = X.T
    return data


def unmask(X, mask_img, order="F"):
    """Take masked data and bring them back into 3D/4D

    This function can be applied to a list of masked data.

    Parameters
    ==========
    X: numpy.ndarray (or list of)
        Masked data. shape: (samples #, features #).
        If X is one-dimensional, it is assumed that samples# == 1.
    mask_img: nifti-like image
        Mask. Must be 3-dimensional.

    Returns
    =======
    data: nifti-like image (or list of)
        Unmasked data. Depending on the shape of X, data can have
        different shapes:

        - X.ndim == 2:
          Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
        - X.ndim == 1:
          Shape: (mask.shape[0], mask.shape[1], mask.shape[2])
    """

    if isinstance(X, list):
        ret = []
        for x in X:
            ret.append(unmask(x, mask_img, order=order))  # 1-level recursion
        return ret

    mask, affine = _load_mask_img(mask_img)

    if X.ndim == 2:
        unmasked = _unmask_nd(X, mask, order=order)
    elif X.ndim == 1:
        unmasked = _unmask_3d(X, mask, order=order)
    return Nifti1Image(unmasked, affine)
