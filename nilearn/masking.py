"""Utilities to compute and operate on brain masks."""

# Authors: Gael Varoquaux, Alexandre Abraham, Philippe Gervais, Ana Luisa Pinho
import numbers
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation, binary_erosion

from . import _utils
from ._utils import fill_doc, logger
from ._utils.cache_mixin import cache
from ._utils.ndimage import get_border_data, largest_connected_component
from ._utils.niimg import safe_get_data
from .datasets import (
    load_mni152_gm_template,
    load_mni152_template,
    load_mni152_wm_template,
)
from .image import get_data, new_img_like, resampling

__all__ = [
    "apply_mask",
    "compute_background_mask",
    "compute_brain_mask",
    "compute_epi_mask",
    "compute_multi_background_mask",
    "compute_multi_brain_mask",
    "compute_multi_epi_mask",
    "intersect_masks",
    "unmask",
]


class _MaskWarning(UserWarning):
    """A class to always raise warnings."""


warnings.simplefilter("always", _MaskWarning)


def load_mask_img(mask_img, allow_empty=False):
    """Check that a mask is valid, ie with two values including 0 and load it.

    Parameters
    ----------
    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        The mask to check.

    allow_empty : :obj:`bool`, default=False
        Allow loading an empty mask (full of 0 values).

    Returns
    -------
    mask : :class:`numpy.ndarray`
        Boolean version of the mask.

    mask_affine : None or (4,4) array-like
        Affine of the mask.
    """
    mask_img = _utils.check_niimg_3d(mask_img)
    mask = safe_get_data(mask_img, ensure_finite=True)
    values = np.unique(mask)

    if len(values) == 1:
        # We accept a single value if it is not 0 (full true mask).
        if values[0] == 0 and not allow_empty:
            raise ValueError(
                "The mask is invalid as it is empty: it masks all data."
            )
    elif len(values) == 2:
        # If there are 2 different values, one of them must be 0 (background)
        if 0 not in values:
            raise ValueError(
                "Background of the mask must be represented with 0. "
                f"Given mask contains: {values}."
            )
    else:
        # If there are more than 2 values, the mask is invalid
        raise ValueError(
            f"Given mask is not made of 2 values: {values}. "
            "Cannot interpret as true or false."
        )

    mask = _utils.as_ndarray(mask, dtype=bool)
    return mask, mask_img.affine


def extrapolate_out_mask(data, mask, iterations=1):
    """Extrapolate values outside of the mask."""
    if iterations > 1:
        data, mask = extrapolate_out_mask(
            data, mask, iterations=iterations - 1
        )
    new_mask = binary_dilation(mask)
    larger_mask = np.zeros(np.array(mask.shape) + 2, dtype=bool)
    larger_mask[1:-1, 1:-1, 1:-1] = mask
    # Use nans as missing value: ugly
    masked_data = np.zeros(larger_mask.shape + data.shape[3:])
    masked_data[1:-1, 1:-1, 1:-1] = data.copy()
    masked_data[np.logical_not(larger_mask)] = np.nan
    outer_shell = larger_mask.copy()
    outer_shell[1:-1, 1:-1, 1:-1] = np.logical_xor(new_mask, mask)
    outer_shell_x, outer_shell_y, outer_shell_z = np.where(outer_shell)
    extrapolation = []
    for i, j, k in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        this_x = outer_shell_x + i
        this_y = outer_shell_y + j
        this_z = outer_shell_z + k
        extrapolation.append(masked_data[this_x, this_y, this_z])

    extrapolation = np.array(extrapolation)
    extrapolation = np.nansum(extrapolation, axis=0) / np.sum(
        np.isfinite(extrapolation), axis=0
    )
    extrapolation[np.logical_not(np.isfinite(extrapolation))] = 0
    new_data = np.zeros_like(masked_data)
    new_data[outer_shell] = extrapolation
    new_data[larger_mask] = masked_data[larger_mask]
    return new_data[1:-1, 1:-1, 1:-1], new_mask


#
# Utilities to compute masks
#
@_utils.fill_doc
def intersect_masks(mask_imgs, threshold=0.5, connected=True):
    """Compute intersection of several masks.

    Given a list of input mask images, generate the output image which
    is the threshold-level intersection of the inputs.

    Parameters
    ----------
    mask_imgs : :obj:`list` of Niimg-like objects
        See :ref:`extracting_data`.
        3D individual masks with same shape and affine.

    threshold : :obj:`float`, default=0.5
        Gives the level of the intersection, must be within [0, 1].
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    %(connected)s
        Default=True.

    Returns
    -------
    grp_mask : 3D :class:`nibabel.nifti1.Nifti1Image`
        Intersection of all masks.
    """
    if len(mask_imgs) == 0:
        raise ValueError("No mask provided for intersection")
    grp_mask = None
    first_mask, ref_affine = load_mask_img(mask_imgs[0], allow_empty=True)
    ref_shape = first_mask.shape
    if threshold > 1:
        raise ValueError("The threshold should be smaller than 1")
    if threshold < 0:
        raise ValueError("The threshold should be greater than 0")
    threshold = min(threshold, 1 - 1.0e-7)

    for this_mask in mask_imgs:
        mask, affine = load_mask_img(this_mask, allow_empty=True)
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
    return new_img_like(
        _utils.check_niimg_3d(mask_imgs[0]), grp_mask, ref_affine
    )


def _post_process_mask(
    mask, affine, opening=2, connected=True, warning_msg=""
):
    """Perform post processing on mask.

    Performs opening and keep only largest connected component is
    ``connected=True``.
    """
    if opening:
        opening = int(opening)
        mask = binary_erosion(mask, iterations=opening)
    mask_any = mask.any()
    if not mask_any:
        warnings.warn(
            f"Computed an empty mask. {warning_msg}",
            _MaskWarning,
            stacklevel=2,
        )
    if connected and mask_any:
        mask = largest_connected_component(mask)
    if opening:
        mask = binary_dilation(mask, iterations=2 * opening)
        mask = binary_erosion(mask, iterations=opening)
    return mask, affine


@_utils.fill_doc
def compute_epi_mask(
    epi_img,
    lower_cutoff=0.2,
    upper_cutoff=0.85,
    connected=True,
    opening=2,
    exclude_zeros=False,
    ensure_finite=True,
    target_affine=None,
    target_shape=None,
    memory=None,
    verbose=0,
):
    """Compute a brain mask from :term:`fMRI` data in 3D or \
    4D :class:`numpy.ndarray`.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    ``lower_cutoff`` and ``upper_cutoff`` of the total image histogram.

    .. note::

        In case of failure, it is usually advisable to
        increase ``lower_cutoff``.

    Parameters
    ----------
    epi_img : Niimg-like object
        See :ref:`extracting_data`.
        :term:`EPI` image, used to compute the mask.
        3D and 4D images are accepted.

        .. note::
            If a 3D image is given, we suggest to use the mean image.

    %(lower_cutoff)s
        Default=0.2.
    %(upper_cutoff)s
        Default=0.85.
    %(connected)s
        Default=True.
    %(opening)s
        Default=2.
    ensure_finite : :obj:`bool`, default=True
        If ensure_finite is True, the non-finite values (NaNs and infs)
        found in the images will be replaced by zeros

    exclude_zeros : :obj:`bool`, default=False
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(memory)s
    %(verbose0)s

    Returns
    -------
    mask : :class:`nibabel.nifti1.Nifti1Image`
        The brain mask (3D image).
    """
    logger.log("EPI mask computation", verbose)

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean

    mean_epi, affine = cache(_compute_mean, memory)(
        epi_img,
        target_affine=target_affine,
        target_shape=target_shape,
        smooth=(1 if opening else False),
    )

    if ensure_finite:
        # Get rid of memmapping
        mean_epi = _utils.as_ndarray(mean_epi)
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = int(np.floor(lower_cutoff * len(sorted_input)))
    upper_cutoff = min(
        int(np.floor(upper_cutoff * len(sorted_input))), len(sorted_input) - 1
    )

    delta = (
        sorted_input[lower_cutoff + 1 : upper_cutoff + 1]
        - sorted_input[lower_cutoff:upper_cutoff]
    )
    ia = delta.argmax()
    threshold = 0.5 * (
        sorted_input[ia + lower_cutoff] + sorted_input[ia + lower_cutoff + 1]
    )

    mask = mean_epi >= threshold

    mask, affine = _post_process_mask(
        mask,
        affine,
        opening=opening,
        connected=connected,
        warning_msg="Are you sure that input "
        "data are EPI images not detrended. ",
    )
    return new_img_like(epi_img, mask, affine)


@_utils.fill_doc
def compute_multi_epi_mask(
    epi_imgs,
    lower_cutoff=0.2,
    upper_cutoff=0.85,
    connected=True,
    opening=2,
    threshold=0.5,
    target_affine=None,
    target_shape=None,
    exclude_zeros=False,
    n_jobs=1,
    memory=None,
    verbose=0,
):
    """Compute a common mask for several runs or subjects of :term:`fMRI` data.

    Uses the mask-finding algorithms to extract masks for each run
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.

    Parameters
    ----------
    epi_imgs : :obj:`list` of Niimg-like objects
        See :ref:`extracting_data`.
        A list of arrays, each item being a subject or a run.
        3D and 4D images are accepted.

        .. note::

            If 3D images are given, we suggest to use the mean image
            of each run.

    threshold : :obj:`float`, optional
        The inter-run threshold: the fraction of the
        total number of runs in for which a :term:`voxel` must be
        in the mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    %(lower_cutoff)s
        Default=0.2.
    %(upper_cutoff)s
        Default=0.85.
    %(connected)s
        Default=True.
    %(opening)s
        Default=2.
    exclude_zeros : :obj:`bool`, default=False
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(memory)s
    %(n_jobs)s

    Returns
    -------
    mask : 3D :class:`nibabel.nifti1.Nifti1Image`
        The brain mask.
    """
    if len(epi_imgs) == 0:
        raise TypeError(
            f"An empty object - {epi_imgs:r} - was passed instead of an "
            "image or a list of images"
        )
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_epi_mask)(
            epi_img,
            lower_cutoff=lower_cutoff,
            upper_cutoff=upper_cutoff,
            connected=connected,
            opening=opening,
            exclude_zeros=exclude_zeros,
            target_affine=target_affine,
            target_shape=target_shape,
            memory=memory,
        )
        for epi_img in epi_imgs
    )

    mask = intersect_masks(masks, connected=connected, threshold=threshold)
    return mask


@_utils.fill_doc
def compute_background_mask(
    data_imgs,
    border_size=2,
    connected=False,
    opening=False,
    target_affine=None,
    target_shape=None,
    memory=None,
    verbose=0,
):
    """Compute a brain mask for the images by guessing \
    the value of the background from the border of the image.

    Parameters
    ----------
    data_imgs : Niimg-like object
        See :ref:`extracting_data`.
        Images used to compute the mask. 3D and 4D images are accepted.

        .. note::

            If a 3D image is given, we suggest to use the mean image.

    %(border_size)s
        Default=2.
    %(connected)s
        Default=False.
    %(opening)s
        Default=False.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(memory)s
    %(verbose0)s

    Returns
    -------
    mask : :class:`nibabel.nifti1.Nifti1Image`
        The brain mask (3D image).
    """
    logger.log("Background mask computation", verbose)

    data_imgs = _utils.check_niimg(data_imgs)

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean

    data, affine = cache(_compute_mean, memory)(
        data_imgs,
        target_affine=target_affine,
        target_shape=target_shape,
        smooth=False,
    )

    if np.isnan(get_border_data(data, border_size)).any():
        # We absolutely need to catter for NaNs as a background:
        # SPM does that by default
        mask = np.logical_not(np.isnan(data))
    else:
        background = np.median(get_border_data(data, border_size))
        mask = data != background

    mask, affine = _post_process_mask(
        mask,
        affine,
        opening=opening,
        connected=connected,
        warning_msg="Are you sure that input "
        "images have a homogeneous background.",
    )
    return new_img_like(data_imgs, mask, affine)


@_utils.fill_doc
def compute_multi_background_mask(
    data_imgs,
    border_size=2,
    connected=True,
    opening=2,
    threshold=0.5,
    target_affine=None,
    target_shape=None,
    n_jobs=1,
    memory=None,
    verbose=0,
):
    """Compute a common mask for several runs or subjects of data.

    Uses the mask-finding algorithms to extract masks for each run
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.

    Parameters
    ----------
    data_imgs : :obj:`list` of Niimg-like objects
        See :ref:`extracting_data`.
        A list of arrays, each item being a subject or a run.
        3D and 4D images are accepted.

        .. note::
            If 3D images are given, we suggest to use the mean image
            of each run.

    threshold : :obj:`float`, optional
        The inter-run threshold: the fraction of the
        total number of run in for which a :term:`voxel` must be
        in the mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    %(border_size)s
        Default=2.
    %(connected)s
        Default=True.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(memory)s
    %(n_jobs)s

    Returns
    -------
    mask : 3D :class:`nibabel.nifti1.Nifti1Image`
        The brain mask.
    """
    if len(data_imgs) == 0:
        raise TypeError(
            f"An empty object - {data_imgs:r} - was passed instead of an "
            "image or a list of images"
        )
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_background_mask)(
            img,
            border_size=border_size,
            connected=connected,
            opening=opening,
            target_affine=target_affine,
            target_shape=target_shape,
            memory=memory,
        )
        for img in data_imgs
    )

    mask = intersect_masks(masks, connected=connected, threshold=threshold)
    return mask


@_utils.fill_doc
def compute_brain_mask(
    target_img,
    threshold=0.5,
    connected=True,
    opening=2,
    memory=None,
    verbose=0,
    mask_type="whole-brain",
):
    """Compute the whole-brain, grey-matter or white-matter mask.

    This mask is calculated using MNI152 1mm-resolution template mask onto the
    target image.

    Parameters
    ----------
    target_img : Niimg-like object
        See :ref:`extracting_data`.
        Images used to compute the mask. 3D and 4D images are accepted.
        Only the shape and affine of ``target_img`` will be used here.

    threshold : :obj:`float`, default=0.5
        The value under which the :term:`MNI` template is cut off.
    %(connected)s
        Default=True.
    %(opening)s
        Default=2.
    %(memory)s
    %(verbose0)s
    %(mask_type)s

        .. versionadded:: 0.8.1

    Returns
    -------
    mask : :class:`nibabel.nifti1.Nifti1Image`
        The whole-brain mask (3D image).
    """
    logger.log(f"Template {mask_type} mask computation", verbose)

    target_img = _utils.check_niimg(target_img)

    if mask_type == "whole-brain":
        template = load_mni152_template(resolution=1)
    elif mask_type == "gm":
        template = load_mni152_gm_template(resolution=1)
    elif mask_type == "wm":
        template = load_mni152_wm_template(resolution=1)
    else:
        raise ValueError(
            f"Unknown mask type {mask_type}. "
            "Only 'whole-brain', 'gm' or 'wm' are accepted."
        )

    resampled_template = cache(resampling.resample_to_img, memory)(
        template,
        target_img,
        copy_header=True,
        force_resample=False,  # TODO set to True in 0.13.0
    )

    mask = (get_data(resampled_template) >= threshold).astype("int8")

    warning_message = (
        f"{mask_type} mask is empty, "
        "lower the threshold or check your input FOV"
    )
    mask, affine = _post_process_mask(
        mask,
        target_img.affine,
        opening=opening,
        connected=connected,
        warning_msg=warning_message,
    )

    return new_img_like(target_img, mask, affine)


@_utils.fill_doc
def compute_multi_brain_mask(
    target_imgs,
    threshold=0.5,
    connected=True,
    opening=2,
    memory=None,
    verbose=0,
    mask_type="whole-brain",
    **kwargs,  # noqa: ARG001
):
    """Compute the whole-brain, grey-matter or white-matter mask \
    for a list of images.

    The mask is calculated through the resampling of the corresponding
    MNI152 template mask onto the target image.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    target_imgs : :obj:`list` of Niimg-like object
        See :ref:`extracting_data`.
        Images used to compute the mask. 3D and 4D images are accepted.

        .. note::
            The images in this list must be of same shape and affine.
            The mask is calculated with the first element of the list
            for only the shape/affine of the image is used for this
            masking strategy.

    threshold : :obj:`float`, default=0.5
        The value under which the :term:`MNI` template is cut off.

    %(connected)s
        Default=True.

    %(opening)s
        Default=2.

    %(mask_type)s

    %(memory)s

    %(verbose0)s

    .. note::
        Argument not used but kept to fit the API

    **kwargs : optional arguments
        Arguments such as 'target_affine' are used in the call of other
        masking strategies, which then would raise an error for this function
        which does not need such arguments.

    Returns
    -------
    mask : :class:`nibabel.nifti1.Nifti1Image`
        The brain mask (3D image).

    See Also
    --------
    nilearn.masking.compute_brain_mask
    """
    if len(target_imgs) == 0:
        raise TypeError(
            f"An empty object - {target_imgs:r} - was passed instead of an "
            "image or a list of images"
        )

    # Check images in the list have the same FOV without loading them in memory
    _ = list(_utils.check_niimg(target_imgs, return_iterator=True))

    mask = compute_brain_mask(
        target_imgs[0],
        threshold=threshold,
        connected=connected,
        opening=opening,
        memory=memory,
        verbose=verbose,
        mask_type=mask_type,
    )
    return mask


#
# Time series extraction
#


@fill_doc
def apply_mask(
    imgs, mask_img, dtype="f", smoothing_fwhm=None, ensure_finite=True
):
    """Extract signals from images using specified mask.

    Read the time series from the given Niimg-like object, using the mask.

    Parameters
    ----------
    imgs : :obj:`list` of 4D Niimg-like objects
        See :ref:`extracting_data`.
        Images to be masked. list of lists of 3D images are also accepted.

    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        3D mask array: True where a :term:`voxel` should be used.

    dtype: numpy dtype or 'f'
        The dtype of the output, if 'f', any float output is acceptable
        and if the data is stored on the disk as floats the data type
        will not be changed.
    %(smoothing_fwhm)s

        .. note::

            Implies ensure_finite=True.

    ensure_finite : :obj:`bool`, default=True
        If ensure_finite is True, the non-finite values (NaNs and
        infs) found in the images will be replaced by zeros.

    Returns
    -------
    run_series : :class:`numpy.ndarray`
        2D array of series with shape (image number, :term:`voxel` number)

    Notes
    -----
    When using smoothing, ``ensure_finite`` is set to True, as non-finite
    values would spread across the image.
    """
    mask_img = _utils.check_niimg_3d(mask_img)
    mask, mask_affine = load_mask_img(mask_img)
    mask_img = new_img_like(mask_img, mask, mask_affine)
    return apply_mask_fmri(
        imgs,
        mask_img,
        dtype=dtype,
        smoothing_fwhm=smoothing_fwhm,
        ensure_finite=ensure_finite,
    )


def apply_mask_fmri(
    imgs, mask_img, dtype="f", smoothing_fwhm=None, ensure_finite=True
):
    """Perform similar action to :func:`nilearn.masking.apply_mask`.

    The only difference with :func:`nilearn.masking.apply_mask` is that
    some costly checks on ``mask_img`` are not performed: ``mask_img`` is
    assumed to contain only two different values (this is checked for in
    :func:`nilearn.masking.apply_mask`, not in this function).
    """
    mask_img = _utils.check_niimg_3d(mask_img)
    mask_affine = mask_img.affine
    mask_data = _utils.as_ndarray(get_data(mask_img), dtype=bool)

    if smoothing_fwhm is not None:
        ensure_finite = True

    imgs_img = _utils.check_niimg(imgs)
    affine = imgs_img.affine[:3, :3]

    if not np.allclose(mask_affine, imgs_img.affine):
        raise ValueError(
            f"Mask affine:\n{mask_affine}\n is different from img affine:"
            "\n{imgs_img.affine}"
        )

    if mask_data.shape != imgs_img.shape[:3]:
        raise ValueError(
            f"Mask shape: {mask_data.shape!s} is different "
            f"from img shape:{imgs_img.shape[:3]!s}"
        )

    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward, especially if smoothing is applied.
    series = safe_get_data(imgs_img)

    if dtype == "f":
        dtype = series.dtype if series.dtype.kind == "f" else np.float32

    series = _utils.as_ndarray(series, dtype=dtype, order="C", copy=True)
    del imgs_img  # frees a lot of memory

    # Delayed import to avoid circular imports
    from .image.image import smooth_array

    smooth_array(
        series,
        affine,
        fwhm=smoothing_fwhm,
        ensure_finite=ensure_finite,
        copy=False,
    )
    return series[mask_data].T


def _unmask_3d(X, mask, order="C"):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Masked data. shape: (features,)

    mask : Niimg-like object
        See :ref:`extracting_data`.
        Mask. mask.ndim must be equal to 3, and dtype *must* be bool.
    """
    if mask.dtype != bool:
        raise TypeError("mask must be a boolean array")
    if X.ndim != 1:
        raise TypeError("X must be a 1-dimensional array")
    n_features = mask.sum()
    if X.shape[0] != n_features:
        raise TypeError(f"X must be of shape (samples, {n_features}).")

    data = np.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        dtype=X.dtype,
        order=order,
    )
    data[mask] = X
    return data


def _unmask_4d(X, mask, order="C"):
    """Take masked data and bring them back to 4D.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Masked data. shape: (samples, features)

    mask : :class:`numpy.ndarray`
        Mask. mask.ndim must be equal to 4, and dtype *must* be bool.

    Returns
    -------
    data : :class:`numpy.ndarray`
        Unmasked data.
        Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
    """
    if mask.dtype != bool:
        raise TypeError("mask must be a boolean array")
    if X.ndim != 2:
        raise TypeError("X must be a 2-dimensional array")
    n_features = mask.sum()
    if X.shape[1] != n_features:
        raise TypeError(f"X must be of shape (samples, {n_features}).")

    data = np.zeros((*mask.shape, X.shape[0]), dtype=X.dtype, order=order)
    data[mask, :] = X.T
    return data


def unmask(X, mask_img, order="F"):
    """Take masked data and bring them back into 3D/4D.

    This function can be applied to a list of masked data.

    Parameters
    ----------
    X : :class:`numpy.ndarray` (or :obj:`list` of)
        Masked data. shape: (samples #, features #).
        If X is one-dimensional, it is assumed that samples# == 1.

    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        Must be 3-dimensional.

    Returns
    -------
    data : :class:`nibabel.nifti1.Nifti1Image`
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
        ret = [unmask(x, mask_img, order=order) for x in X]
        return ret

    # The code after this block assumes that X is an ndarray; ensure this
    X = np.asanyarray(X)

    mask_img = _utils.check_niimg_3d(mask_img)
    mask, affine = load_mask_img(mask_img)

    if np.ndim(X) == 2:
        unmasked = _unmask_4d(X, mask, order=order)
    elif np.ndim(X) == 1:
        unmasked = _unmask_3d(X, mask, order=order)
    else:
        raise TypeError(
            f"Masked data X must be 2D or 1D array; got shape: {X.shape!s}"
        )

    return new_img_like(mask_img, unmasked, affine)


def unmask_from_to_3d_array(w, mask):
    """Unmask an image into whole brain, \
    with off-mask :term:`voxels<voxel>` set to 0.

    Used as a stand-alone function in low-level decoding (SpaceNet) and
    clustering (ReNA) functions.

    Parameters
    ----------
    w : :class:`numpy.ndarray`, shape (n_features,)
      The image to be unmasked.

    mask : :class:`numpy.ndarray`
      The mask used in the unmasking operation. It is required that
      ``mask.sum() == n_features``.

    Returns
    -------
    out : 3D :class:`numpy.ndarray` (same shape as `mask`)
        The unmasked version of `w`.
    """
    if mask.sum() != len(w):
        raise ValueError("Expecting mask.sum() == len(w).")
    out = np.zeros(mask.shape, dtype=w.dtype)
    out[mask] = w
    return out
