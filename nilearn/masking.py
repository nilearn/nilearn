"""
Utilities to compute and operate on brain masks
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import warnings
import numbers

import numpy as np
from scipy import ndimage
from sklearn.externals.joblib import Parallel, delayed

from . import _utils
from .image import new_img_like
from ._utils.cache_mixin import cache
from ._utils.ndimage import largest_connected_component, get_border_data
from ._utils.niimg import _safe_get_data
from ._utils.compat import get_affine


class MaskWarning(UserWarning):
    "A class to always raise warnings"


warnings.simplefilter("always", MaskWarning)


def _load_mask_img(mask_img, allow_empty=False):
    """Check that a mask is valid, ie with two values including 0 and load it.

    Parameters
    ----------
    mask_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask to check

    allow_empty: boolean, optional
        Allow loading an empty mask (full of 0 values)

    Returns
    -------
    mask: numpy.ndarray
        boolean version of the mask
    """
    mask_img = _utils.check_niimg_3d(mask_img)
    mask = mask_img.get_data()
    values = np.unique(mask)

    if len(values) == 1:
        # We accept a single value if it is not 0 (full true mask).
        if values[0] == 0 and not allow_empty:
            raise ValueError(
                'The mask is invalid as it is empty: it masks all data.')
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
    return mask, get_affine(mask_img)


def _extrapolate_out_mask(data, mask, iterations=1):
    """ Extrapolate values outside of the mask.
    """
    if iterations > 1:
        data, mask = _extrapolate_out_mask(data, mask,
                                          iterations=iterations - 1)
    new_mask = ndimage.binary_dilation(mask)
    larger_mask = np.zeros(np.array(mask.shape) + 2, dtype=np.bool)
    larger_mask[1:-1, 1:-1, 1:-1] = mask
    # Use nans as missing value: ugly
    masked_data = np.zeros(larger_mask.shape + data.shape[3:])
    masked_data[1:-1, 1:-1, 1:-1] = data.copy()
    masked_data[np.logical_not(larger_mask)] = np.nan
    outer_shell = larger_mask.copy()
    outer_shell[1:-1, 1:-1, 1:-1] = np.logical_xor(new_mask, mask)
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


#
# Utilities to compute masks
#

def intersect_masks(mask_imgs, threshold=0.5, connected=True):
    """ Compute intersection of several masks

    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs

    Parameters
    ----------
    mask_imgs: list of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html
        3D individual masks with same shape and affine.

    threshold: float, optional
        Gives the level of the intersection, must be within [0, 1].
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    connected: bool, optional
        If true, extract the main connected component

    Returns
    -------
        grp_mask: 3D nibabel.Nifti1Image
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
    return new_img_like(_utils.check_niimg_3d(mask_imgs[0]), grp_mask, ref_affine)


def _post_process_mask(mask, affine, opening=2, connected=True,
                       warning_msg=""):
    if opening:
        opening = int(opening)
        mask = ndimage.binary_erosion(mask, iterations=opening)
    mask_any = mask.any()
    if not mask_any:
        warnings.warn("Computed an empty mask. %s" % warning_msg,
            MaskWarning, stacklevel=2)
    if connected and mask_any:
        mask = largest_connected_component(mask)
    if opening:
        mask = ndimage.binary_dilation(mask, iterations=2 * opening)
        mask = ndimage.binary_erosion(mask, iterations=opening)
    return mask, affine


def compute_epi_mask(epi_img, lower_cutoff=0.2, upper_cutoff=0.85,
                     connected=True, opening=2, exclude_zeros=False,
                     ensure_finite=True,
                     target_affine=None, target_shape=None,
                     memory=None, verbose=0,):
    """Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    epi_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        EPI image, used to compute the mask. 3D and 4D images are accepted.
        If a 3D image is given, we suggest to use the mean image

    lower_cutoff: float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: bool, optional
        if connected is True, only the largest connect component is kept.

    opening: bool or int, optional
        if opening is True, a morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
        If opening is an integer `n`, it is performed via `n` erosions.
        After estimation of the largest connected constituent, 2`n` closing
        operations are performed followed by `n` erosions. This corresponds
        to 1 opening operation of order `n` followed by a closing operator
        of order `n`.
        Note that turning off opening (opening=False) will also prevent
        any smoothing applied to the image during the mask computation.

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
        Used to cache the function call: if this is a string, it
        specifies the directory where the cache will be stored.

    verbose: int, optional
        Controls the amount of verbosity: higher numbers give
        more messages

    Returns
    -------
    mask: nibabel.Nifti1Image
        The brain mask (3D image)
    """
    if verbose > 0:
        print("EPI mask computation")

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean
    mean_epi, affine = cache(_compute_mean, memory)(epi_img,
                                     target_affine=target_affine,
                                     target_shape=target_shape,
                                     smooth=(1 if opening else False))

    if ensure_finite:
        # Get rid of memmapping
        mean_epi = _utils.as_ndarray(mean_epi)
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = int(np.floor(lower_cutoff * len(sorted_input)))
    upper_cutoff = min(int(np.floor(upper_cutoff * len(sorted_input))),
                       len(sorted_input) - 1)

    delta = sorted_input[lower_cutoff + 1:upper_cutoff + 1] \
        - sorted_input[lower_cutoff:upper_cutoff]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + lower_cutoff]
                       + sorted_input[ia + lower_cutoff + 1])

    mask = mean_epi >= threshold

    mask, affine = _post_process_mask(mask, affine, opening=opening,
        connected=connected, warning_msg="Are you sure that input "
            "data are EPI images not detrended. ")
    return new_img_like(epi_img, mask, affine)


def compute_multi_epi_mask(epi_imgs, lower_cutoff=0.2, upper_cutoff=0.85,
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
    epi_imgs: list of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html
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
    mask : 3D nibabel.Nifti1Image
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


def compute_background_mask(data_imgs, border_size=2,
                     connected=False, opening=False,
                     target_affine=None, target_shape=None,
                     memory=None, verbose=0):
    """ Compute a brain mask for the images by guessing the value of the
    background from the border of the image.

    Parameters
    ----------
    data_imgs: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Images used to compute the mask. 3D and 4D images are accepted.
        If a 3D image is given, we suggest to use the mean image

    border_size: integer, optional
        The size, in voxel of the border used on the side of the image
        to determine the value of the background.

    connected: bool, optional
        if connected is True, only the largest connect component is kept.

    opening: bool or int, optional
        if opening is True, a morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
        If opening is an integer `n`, it is performed via `n` erosions.
        After estimation of the largest connected constituent, 2`n` closing
        operations are performed followed by `n` erosions. This corresponds
        to 1 opening operation of order `n` followed by a closing operator
        of order `n`.

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
        print("Background mask computation")

    data_imgs = _utils.check_niimg(data_imgs)

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean
    data, affine = cache(_compute_mean, memory)(data_imgs,
                target_affine=target_affine, target_shape=target_shape,
                smooth=False)

    background = np.median(get_border_data(data, border_size))
    if np.isnan(background):
        # We absolutely need to catter for NaNs as a background:
        # SPM does that by default
        mask = np.logical_not(np.isnan(data))
    else:
        mask = data != background

    mask, affine = _post_process_mask(mask, affine, opening=opening,
        connected=connected, warning_msg="Are you sure that input "
            "images have a homogeneous background.")
    return new_img_like(data_imgs, mask, affine)


def compute_multi_background_mask(data_imgs, border_size=2, upper_cutoff=0.85,
                           connected=True, opening=2, threshold=0.5,
                           target_affine=None, target_shape=None,
                           exclude_zeros=False, n_jobs=1,
                           memory=None, verbose=0):
    """ Compute a common mask for several sessions or subjects of data.

    Uses the mask-finding algorithms to extract masks for each session
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.

    Parameters
    ----------
    data_imgs: list of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html
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

    border_size: integer, optional
        The size, in voxel of the border used on the side of the image
        to determine the value of the background.

    connected: boolean, optional
        if connected is True, only the largest connect component is kept.

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
    mask : 3D nibabel.Nifti1Image
        The brain mask.
    """
    if len(data_imgs) == 0:
        raise TypeError('An empty object - %r - was passed instead of an '
                        'image or a list of images' % data_imgs)
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_background_mask)(img,
                                  border_size=border_size,
                                  connected=connected,
                                  opening=opening,
                                  target_affine=target_affine,
                                  target_shape=target_shape,
                                  memory=memory)
        for img in data_imgs)

    mask = intersect_masks(masks, connected=connected, threshold=threshold)
    return mask


#
# Time series extraction
#

def apply_mask(imgs, mask_img, dtype='f',
               smoothing_fwhm=None, ensure_finite=True):
    """Extract signals from images using specified mask.

    Read the time series from the given Niimg-like object, using the mask.

    Parameters
    -----------
    imgs: list of 4D Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html
        Images to be masked. list of lists of 3D images are also accepted.

    mask_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        3D mask array: True where a voxel should be used.

    dtype: numpy dtype or 'f'
        The dtype of the output, if 'f', any float output is acceptable
        and if the data is stored on the disk as floats the data type
        will not be changed.

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
    mask_img = _utils.check_niimg_3d(mask_img)
    mask, mask_affine = _load_mask_img(mask_img)
    mask_img = new_img_like(mask_img, mask, mask_affine)
    return _apply_mask_fmri(imgs, mask_img, dtype=dtype,
                            smoothing_fwhm=smoothing_fwhm,
                            ensure_finite=ensure_finite)


def _apply_mask_fmri(imgs, mask_img, dtype='f',
                     smoothing_fwhm=None, ensure_finite=True):
    """Same as apply_mask().

    The only difference with apply_mask is that some costly checks on mask_img
    are not performed: mask_img is assumed to contain only two different
    values (this is checked for in apply_mask, not in this function).
    """

    mask_img = _utils.check_niimg_3d(mask_img)
    mask_affine = get_affine(mask_img)
    mask_data = _utils.as_ndarray(mask_img.get_data(),
                                  dtype=np.bool)

    if smoothing_fwhm is not None:
        ensure_finite = True

    imgs_img = _utils.check_niimg(imgs)
    affine = get_affine(imgs_img)[:3, :3]

    if not np.allclose(mask_affine, get_affine(imgs_img)):
        raise ValueError('Mask affine: \n%s\n is different from img affine:'
                         '\n%s' % (str(mask_affine),
                                   str(get_affine(imgs_img))))

    if not mask_data.shape == imgs_img.shape[:3]:
        raise ValueError('Mask shape: %s is different from img shape:%s'
                         % (str(mask_data.shape), str(imgs_img.shape[:3])))

    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward, especially if smoothing is applied.
    series = _safe_get_data(imgs_img)

    if dtype == 'f':
        if series.dtype.kind == 'f':
            dtype = series.dtype
        else:
            dtype = np.float32
    series = _utils.as_ndarray(series, dtype=dtype, order="C",
                               copy=True)
    del imgs_img  # frees a lot of memory

    # Delayed import to avoid circular imports
    from .image.image import _smooth_array
    _smooth_array(series, affine, fwhm=smoothing_fwhm,
                  ensure_finite=ensure_finite, copy=False)
    return series[mask_data].T


def _unmask_3d(X, mask, order="C"):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ----------
    X: numpy.ndarray
        Masked data. shape: (features,)

    mask: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask. mask.ndim must be equal to 3, and dtype *must* be bool.
    """

    if mask.dtype != np.bool:
        raise TypeError("mask must be a boolean array")
    if X.ndim != 1:
        raise TypeError("X must be a 1-dimensional array")
    n_features = mask.sum()
    if X.shape[0] != n_features:
        raise TypeError('X must be of shape (samples, %d).' % n_features)

    data = np.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        dtype=X.dtype, order=order)
    data[mask] = X
    return data


def _unmask_4d(X, mask, order="C"):
    """Take masked data and bring them back to 4D.

    Parameters
    ----------
    X: numpy.ndarray
        Masked data. shape: (samples, features)

    mask: numpy.ndarray
        Mask. mask.ndim must be equal to 4, and dtype *must* be bool.

    Returns
    -------
    data: numpy.ndarray
        Unmasked data.
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
    """

    if mask.dtype != np.bool:
        raise TypeError("mask must be a boolean array")
    if X.ndim != 2:
        raise TypeError("X must be a 2-dimensional array")
    n_features = mask.sum()
    if X.shape[1] != n_features:
        raise TypeError('X must be of shape (samples, %d).' % n_features)

    data = np.zeros(mask.shape + (X.shape[0],), dtype=X.dtype, order=order)
    data[mask, :] = X.T
    return data


def unmask(X, mask_img, order="F"):
    """Take masked data and bring them back into 3D/4D

    This function can be applied to a list of masked data.

    Parameters
    ----------
    X: numpy.ndarray (or list of)
        Masked data. shape: (samples #, features #).
        If X is one-dimensional, it is assumed that samples# == 1.
    mask_img: niimg: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Must be 3-dimensional.

    Returns
    -------
    data: nibabel.Nift1Image object
        Unmasked data. Depending on the shape of X, data can have
        different shapes:

        - X.ndim == 2:
          Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
        - X.ndim == 1:
          Shape: (mask.shape[0], mask.shape[1], mask.shape[2])
    """
    # Handle lists. This can be a list of other lists / arrays, or a list or
    # numbers. In the latter case skip.
    if isinstance(X, list) and not isinstance(X[0], numbers.Number):
        ret = []
        for x in X:
            ret.append(unmask(x, mask_img, order=order))  # 1-level recursion
        return ret

    # The code after this block assumes that X is an ndarray; ensure this
    X = np.asanyarray(X)

    mask_img = _utils.check_niimg_3d(mask_img)
    mask, affine = _load_mask_img(mask_img)

    if np.ndim(X) == 2:
        unmasked = _unmask_4d(X, mask, order=order)
    elif np.ndim(X) == 1:
        unmasked = _unmask_3d(X, mask, order=order)
    else:
        raise TypeError("Masked data X must be 2D or 1D array; "
                        "got shape: %s" % str(X.shape))

    return new_img_like(mask_img, unmasked, affine)
