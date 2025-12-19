"""
Preprocessing functions for images.

See also nilearn.signal.
"""

import collections.abc
import glob
import itertools
import warnings
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
from joblib import Memory, Parallel, delayed
from nibabel import Nifti1Image, Nifti1Pair, load, spatialimages
from nibabel.fileslice import is_fancy
from nibabel.spatialimages import SpatialImage
from numpy.testing import assert_array_equal
from scipy.ndimage import gaussian_filter1d, generate_binary_structure, label
from scipy.stats import scoreatpercentile

from nilearn import EXPAND_PATH_WILDCARDS, signal
from nilearn._utils import logger
from nilearn._utils.cache_mixin import cache
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg import (
    _get_data,
    load_niimg,
    repr_niimgs,
    safe_get_data,
)
from nilearn._utils.numpy_conversions import as_ndarray
from nilearn._utils.param_validation import (
    check_is_of_allowed_type,
    check_params,
    check_threshold,
)
from nilearn._utils.path_finding import resolve_globbing
from nilearn.exceptions import DimensionError
from nilearn.surface.surface import (
    SurfaceImage,
    at_least_2d,
    extract_data,
    find_surface_clusters,
)
from nilearn.surface.surface import get_data as get_surface_data
from nilearn.surface.utils import assert_polymesh_equal, check_polymesh_equal
from nilearn.typing import ClusterThreshold, NiimgLike


def get_data(img):
    """Get the image data as a :class:`numpy.ndarray`.

    Parameters
    ----------
    img : Niimg-like object or iterable of Niimg-like objects
        See :ref:`extracting_data`.

    Returns
    -------
    :class:`numpy.ndarray`
        3D or 4D numpy array depending on the shape of `img`. This function
        preserves the type of the image data.
        If `img` is an in-memory Nifti image
        it returns the image data array itself -- not a copy.

    """
    img = check_niimg(img)
    return _get_data(img)


def high_variance_confounds(
    imgs, n_confounds=5, percentile=2.0, detrend=True, mask_img=None
) -> np.ndarray:
    """Return confounds extracted from input signals with highest variance.

    Parameters
    ----------
    imgs : 4D Niimg-like or 2D SurfaceImage object
        See :ref:`extracting_data`.

    mask_img : Niimg-like or SurfaceImage object, or None, default=None
        If not provided, all voxels / vertices are used.
        If provided, confounds are extracted
        from voxels / vertices inside the mask.
        See :ref:`extracting_data`.

    n_confounds : :obj:`int`, default=5
        Number of confounds to return.

    percentile : :obj:`float`, default=2.0
        Highest-variance signals percentile to keep before computing the
        singular value decomposition, 0. <= `percentile` <= 100.
        `mask_img.sum() * percentile / 100` must be greater than `n_confounds`.

    detrend : :obj:`bool`, default=True
        If True, detrend signals before processing.

    Returns
    -------
    :class:`numpy.ndarray`
        Highest variance confounds. Shape: *(number_of_scans, n_confounds)*.

    Notes
    -----
    This method is related to what has been published in the literature
    as 'CompCor' (Behzadi NeuroImage 2007).

    The implemented algorithm does the following:

    - Computes the sum of squares for each signal (no mean removal).
    - Keeps a given percentile of signals with highest variance (percentile).
    - Computes an SVD of the extracted signals.
    - Returns a given number (n_confounds) of signals from the SVD with
      highest singular values.

    See Also
    --------
    nilearn.signal.high_variance_confounds

    """
    from .. import masking

    check_compatibility_mask_and_images(mask_img, imgs)

    if mask_img is not None:
        if isinstance(imgs, SurfaceImage):
            check_polymesh_equal(mask_img.mesh, imgs.mesh)
        sigs = masking.apply_mask(imgs, mask_img)

    # Load the data only if it doesn't need to be masked
    # Not using apply_mask here saves memory in most cases.
    elif isinstance(imgs, SurfaceImage):
        sigs = as_ndarray(get_surface_data(imgs))
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T
    else:
        imgs = check_niimg_4d(imgs)
        sigs = as_ndarray(get_data(imgs))
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T

    del imgs  # help reduce memory consumption

    return signal.high_variance_confounds(
        sigs, n_confounds=n_confounds, percentile=percentile, detrend=detrend
    )


def _fast_smooth_array(arr):
    """Perform simple smoothing.

    Less computationally expensive than applying a Gaussian filter.

    Only the first three dimensions of the array will be smoothed. The
    filter uses [0.2, 1, 0.2] weights in each direction and use a
    normalization to preserve the local average value.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        4D array, with image number as last dimension. 3D arrays are
        also accepted.

    Returns
    -------
    :class:`numpy.ndarray`
        Smoothed array.

    Notes
    -----
    Rather than calling this function directly, users are encouraged
    to call the high-level function :func:`smooth_img` with
    `fwhm='fast'`.

    """
    neighbor_weight = 0.2
    # 6 neighbors in 3D if not on an edge
    n_neighbors = 6
    # This scale ensures that a uniform array stays uniform
    # except on the array edges
    scale = 1 + n_neighbors * neighbor_weight

    # Need to copy because the smoothing is done in multiple statements
    # and there does not seem to be an easy way to do it in place
    smoothed_arr = arr.copy()
    weighted_arr = neighbor_weight * arr

    smoothed_arr[:-1] += weighted_arr[1:]
    smoothed_arr[1:] += weighted_arr[:-1]
    smoothed_arr[:, :-1] += weighted_arr[:, 1:]
    smoothed_arr[:, 1:] += weighted_arr[:, :-1]
    smoothed_arr[:, :, :-1] += weighted_arr[:, :, 1:]
    smoothed_arr[:, :, 1:] += weighted_arr[:, :, :-1]
    smoothed_arr /= scale

    return smoothed_arr


@fill_doc
def smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of `arr`.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        4D array, with image number as last dimension. 3D arrays are also
        accepted.

    affine : :class:`numpy.ndarray`
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If `fwhm='fast'`, the affine is not used and can be None.
    %(fwhm)s
    ensure_finite : :obj:`bool`, default=True
        If True, replace every non-finite values (like NaNs) by zero before
        filtering.

    copy : :obj:`bool`, default=True
        If True, input array is not modified. True by default: the filtering
        is not performed in-place.

    Returns
    -------
    :class:`numpy.ndarray`
        Filtered `arr`.

    Notes
    -----
    This function is most efficient with arr in C order.

    """
    # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
    # See issue #1537
    if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
        warnings.warn(
            f"The parameter 'fwhm' for smoothing is specified as {fwhm}. "
            "Setting it to None (no smoothing will be performed)",
            stacklevel=find_stack_level(),
        )
        fwhm = None
    if arr.dtype.kind == "i":
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)  # We don't need crazy precision.
    if copy:
        arr = arr.copy()
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0
    if isinstance(fwhm, str) and (fwhm == "fast"):
        arr = _fast_smooth_array(arr)
    elif fwhm is not None:
        fwhm = np.asarray([fwhm]).ravel()
        fwhm = np.asarray([0.0 if elem is None else elem for elem in fwhm])
        affine = affine[:3, :3]  # Keep only the scale part.
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
        vox_size = np.sqrt(np.sum(affine**2, axis=0))
        sigma = fwhm / (fwhm_over_sigma_ratio * vox_size)
        for n, s in enumerate(sigma):
            if s > 0.0:
                gaussian_filter1d(arr, s, output=arr, axis=n)
    return arr


@fill_doc
def smooth_img(imgs, fwhm):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of `arr`.
    In all cases, non-finite values in input image are replaced by zeros.

    Parameters
    ----------
    imgs : Niimg-like object or iterable of Niimg-like objects
        Image(s) to smooth (see :ref:`extracting_data`
        for a detailed description of the valid input types).
    %(fwhm)s

    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image` or list of
        Filtered input image. If `imgs` is an iterable,
        then `filtered_img` is a list.

    """
    # Use hasattr() instead of isinstance to workaround a Python 2.6/2.7 bug
    # See http://bugs.python.org/issue7624
    imgs = stringify_path(imgs)
    if hasattr(imgs, "__iter__") and not isinstance(imgs, str):
        single_img = False
    else:
        single_img = True
        imgs = [imgs]

    ret = []
    for img in imgs:
        img = check_niimg(img)
        affine = img.affine
        filtered = smooth_array(
            get_data(img), affine, fwhm=fwhm, ensure_finite=True, copy=True
        )
        ret.append(new_img_like(img, filtered, affine))

    return ret[0] if single_img else ret


def _crop_img_to(img, slices, copy=True, copy_header=True):
    """Crops an image to a smaller size.

    Crop `img` to size indicated by slices and adjust affine accordingly.

    Parameters
    ----------
    img : Niimg-like object
        Image to be cropped.
        If slices has less entries than `img` has dimensions,
        the slices will be applied to the first `len(slices)` dimensions
        (See :ref:`extracting_data`).

    slices : list of slices
        Defines the range of the crop.
        E.g. [slice(20, 200), slice(40, 150), slice(0, 100)] defines a cube.

    copy : :obj:`bool`, default=True
        Specifies whether cropped data is to be copied or not.

    %(copy_header)s

        .. nilearn_versionadded:: 0.11.0

    Returns
    -------
    Niimg-like object
        Cropped version of the input image.

    offset : :obj:`list`, optional
        List of tuples representing the number of voxels removed
        (before, after) the cropped volumes, i.e.:
        *[(x1_pre, x1_post), (x2_pre, x2_post), ..., (xN_pre, xN_post)]*

    """
    img = check_niimg(img)

    data = get_data(img)
    affine = img.affine

    cropped_data = data[tuple(slices)]
    if copy:
        cropped_data = cropped_data.copy()

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin

    return new_img_like(img, cropped_data, new_affine, copy_header=copy_header)


@fill_doc
def crop_img(
    img, rtol=1e-8, copy=True, pad=True, return_offset=False, copy_header=True
):
    """Crops an image as much as possible.

    Will crop `img`, removing as many zero entries as possible without
    touching non-zero entries.
    Will leave one :term:`voxel` of zero padding
    around the obtained non-zero area in order
    to avoid sampling issues later on.

    Parameters
    ----------
    img : Niimg-like object
        Image to be cropped (see :ref:`extracting_data` for a detailed
        description of the valid input types).

    rtol : :obj:`float`, default=1e-08
        relative tolerance (with respect to maximal absolute value of the
        image), under which values are considered negligible and thus
        croppable.

    copy : :obj:`bool`, default=True
        Specifies whether cropped data is copied or not.

    pad : :obj:`bool`, default=True
        Toggles adding 1-voxel of 0s around the border.

    return_offset : :obj:`bool`, default=False
        Specifies whether to return a tuple of the removed padding.

    %(copy_header)s

        .. nilearn_versionadded:: 0.11.0

    Returns
    -------
    Niimg-like object or :obj:`tuple`
        Cropped version of the input image and, if `return_offset=True`,
        a tuple of tuples representing the number of voxels
        removed (before, after) the cropped volumes, i.e.:
        *[(x1_pre, x1_post), (x2_pre, x2_post), ..., (xN_pre, xN_post)]*

        If the specified image is empty, the original image will be returned.
    """
    check_params(locals())

    img = check_niimg(img)
    data = get_data(img)
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(
        data < -rtol * infinity_norm, data > rtol * infinity_norm
    )

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))

    # Sets full range if no data are found along the axis
    if coords.shape[1] == 0:
        start, end = [0, 0, 0], list(data.shape)
        pad = False
        warnings.warn(
            f"No values above {rtol}. Returning original image.",
            stacklevel=find_stack_level(),
        )
    else:
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    if pad:
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, data.shape[:3])

    slices = list(map(slice, start, end))[:3]
    cropped_im = _crop_img_to(img, slices, copy=copy, copy_header=copy_header)
    return (cropped_im, tuple(slices)) if return_offset else cropped_im


def compute_mean(imgs, target_affine=None, target_shape=None, smooth=False):
    """Compute the mean of the images over time or the 4th dimension.

    See mean_img for details about the API.
    """
    from . import resampling

    input_repr = repr_niimgs(imgs, shorten=True)

    imgs = check_niimg(imgs)
    mean_data = safe_get_data(imgs)
    affine = imgs.affine
    # Free memory ASAP
    del imgs
    if mean_data.ndim not in (3, 4):
        raise ValueError(
            "Computation expects 3D or 4D images, "
            f"but {mean_data.ndim} dimensions were given ({input_repr})"
        )
    if mean_data.ndim == 4:
        mean_data = mean_data.mean(axis=-1)
    else:
        mean_data = mean_data.copy()
    mean_data = resampling.resample_img(
        Nifti1Image(mean_data, affine),
        target_affine=target_affine,
        target_shape=target_shape,
        copy=False,
    )
    affine = mean_data.affine
    mean_data = get_data(mean_data)

    if smooth:
        nan_mask = np.isnan(mean_data)
        mean_data = smooth_array(
            mean_data,
            affine=np.eye(4),
            fwhm=smooth,
            ensure_finite=True,
            copy=False,
        )
        mean_data[nan_mask] = np.nan

    return mean_data, affine


def _compute_surface_mean(imgs: SurfaceImage) -> SurfaceImage:
    """Compute mean of a single surface image over its 2nd dimension."""
    if len(imgs.shape) < 2 or imgs.shape[1] < 2:
        data = imgs.data
    else:
        data = {
            part: np.mean(value, axis=1).astype(float)
            for part, value in imgs.data.parts.items()
        }
    return new_img_like(imgs, data=data)


@fill_doc
def mean_img(
    imgs,
    target_affine=None,
    target_shape=None,
    verbose=0,
    n_jobs=1,
    copy_header=True,
):
    """Compute the mean over images.

    This can be a mean over time or the 4th dimension for a volume,
    or the 2nd dimension for a surface image.

    Note that if list of 4D volume images (or 2D surface images)
    are given,
    the mean of each image is computed separately,
    and the resulting mean is computed after.

    Parameters
    ----------
    imgs : Niimg-like or :obj:`~nilearn.surface.SurfaceImage` object, or \
           iterable of Niimg-like or :obj:`~nilearn.surface.SurfaceImage`.
        Images to be averaged over 'time'
        (see :ref:`extracting_data`
        for a detailed description of the valid input types).

    %(target_affine)s
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    %(target_shape)s
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    %(verbose0)s

    n_jobs : :obj:`int`, default=1
        The number of CPUs to use to do the computation (-1 means
        'all CPUs').
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    %(copy_header)s

        .. nilearn_versionadded:: 0.11.0

    Returns
    -------
    :obj:`~nibabel.nifti1.Nifti1Image` or :obj:`~nilearn.surface.SurfaceImage`
        Mean image.

    See Also
    --------
    nilearn.image.math_img : For more general operations on images.

    """
    check_params(locals())
    is_iterable = isinstance(imgs, collections.abc.Iterable)
    is_surface_img = isinstance(imgs, SurfaceImage) or (
        is_iterable and all(isinstance(x, SurfaceImage) for x in imgs)
    )
    if is_surface_img:
        if not is_iterable:
            imgs = [imgs]
        all_means = concat_imgs([_compute_surface_mean(x) for x in imgs])
        return _compute_surface_mean(all_means)

    imgs = stringify_path(imgs)
    is_str = isinstance(imgs, str)
    is_iterable = isinstance(imgs, collections.abc.Iterable)
    if is_str or not is_iterable:
        imgs = [imgs]

    imgs_iter = iter(imgs)
    first_img = check_niimg(next(imgs_iter))

    # Compute the first mean to retrieve the reference
    # target_affine and target_shape if_needed
    n_imgs = 1
    running_mean, first_affine = compute_mean(
        first_img, target_affine=target_affine, target_shape=target_shape
    )

    if target_affine is None or target_shape is None:
        target_affine = first_affine
        target_shape = running_mean.shape[:3]

    for this_mean in Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_mean)(
            n, target_affine=target_affine, target_shape=target_shape
        )
        for n in imgs_iter
    ):
        n_imgs += 1
        # compute_mean returns (mean_img, affine)
        this_mean = this_mean[0]
        running_mean += this_mean

    running_mean = running_mean / float(n_imgs)
    return new_img_like(
        first_img, running_mean, target_affine, copy_header=copy_header
    )


def swap_img_hemispheres(img):
    """Perform swapping of hemispheres in the indicated NIfTI image.

       Use case: synchronizing ROIs across hemispheres.

    Parameters
    ----------
    img : Niimg-like object
        Images to swap (see :ref:`extracting_data` for a detailed description
        of the valid input types).

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Hemispherically swapped image.

    Notes
    -----
    Assumes that the image is sagitally aligned.

    Should be used with caution (confusion might be caused with
    radio/neuro conventions)

    Note that this does not require a change of the affine matrix.

    """
    from .resampling import reorder_img

    # Check input is really a path to a nifti file or a nifti object
    img = check_niimg_3d(img)

    # get nifti in x-y-z order
    img = reorder_img(img)

    # create swapped nifti object
    out_img = new_img_like(img, get_data(img)[::-1], img.affine)

    return out_img


def index_img(imgs, index):
    """Indexes into a image in the last dimension.

    Common use cases include extracting an image out of `img` or
    creating a 4D (or 2D for surface) image
    whose data is a subset of `img` data.

    Parameters
    ----------
    imgs : 4D Niimg-like object or 2D :obj:`~nilearn.surface.SurfaceImage`
        See :ref:`extracting_data`.

    index : Any type compatible with numpy array indexing
        Used for indexing the data array in the last dimension.

    Returns
    -------
    :obj:`~nibabel.nifti1.Nifti1Image` or :obj:`~nilearn.surface.SurfaceImage`
        Indexed image.

    See Also
    --------
    nilearn.image.concat_imgs
    nilearn.image.iter_img

    Examples
    --------
    First we concatenate two MNI152 images to create a 4D-image::

     >>> from nilearn import datasets
     >>> from nilearn.image import concat_imgs, index_img
     >>> joint_mni_image = concat_imgs([datasets.load_mni152_template(),
     ...                                datasets.load_mni152_template()])
     >>> print(joint_mni_image.shape)
     (197, 233, 189, 2)

    We can now select one slice from the last dimension of this 4D-image::

     >>> single_mni_image = index_img(joint_mni_image, 1)
     >>> print(single_mni_image.shape)
     (197, 233, 189)

    We can also select multiple frames using the `slice` constructor::

     >>> five_mni_images = concat_imgs([datasets.load_mni152_template()] * 5)
     >>> print(five_mni_images.shape)
     (197, 233, 189, 5)

     >>> first_three_images = index_img(five_mni_images,
     ...                                slice(0, 3))
     >>> print(first_three_images.shape)
     (197, 233, 189, 3)

    """
    if isinstance(imgs, SurfaceImage):
        imgs.data._check_ndims(2, var_name="imgs")
        return new_img_like(imgs, data=extract_data(imgs, index))

    imgs = check_niimg_4d(imgs)
    # duck-type for pandas arrays, and select the 'values' attr
    if hasattr(index, "values") and hasattr(index, "iloc"):
        index = index.to_numpy().flatten()
    return _index_img(imgs, index)


def _index_img(img: Nifti1Image, index):
    """Helper function for check_niimg_4d."""  # noqa: D401
    # catch too-large negative indices
    if isinstance(index, int):
        n = img.shape[3]
        if index < -n or index >= n:
            raise IndexError(
                f"Integer index {index} out of range for last dimension"
                f" with size {n}"
            )

    if is_fancy(index):
        data = _get_data(img)[:, :, :, index]
    else:
        # this should be faster
        data = _get_data(img.slicer[..., index])

    return new_img_like(img, data, img.affine)


def iter_img(imgs):
    """Iterate over images.

    Could be along the the 4th dimension for 4D Niimg-like object
    or the 2nd dimension for 2D Surface images..

    Parameters
    ----------
    imgs : 4D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        See :ref:`extracting_data`.

    Returns
    -------
    Iterator of :class:`~nibabel.nifti1.Nifti1Image` \
        or :obj:`~nilearn.surface.SurfaceImage`

    See Also
    --------
    nilearn.image.index_img

    """
    if isinstance(imgs, SurfaceImage):
        output = (
            index_img(imgs, i) for i in range(at_least_2d(imgs).shape[1])
        )
        return output
    return check_niimg_4d(imgs, return_iterator=True)


def _downcast_from_int64_if_possible(data):
    """Try to downcast to int32 if possible.

    If `data` is 64-bit ints and can be converted to (signed) int 32,
    return an int32 copy, otherwise return `data` itself.
    """
    if data.dtype not in (np.int64, np.uint64):
        return data
    img_min, img_max = np.min(data), np.max(data)
    type_info = np.iinfo(np.int32)
    can_cast = type_info.min <= img_min and type_info.max >= img_max
    if can_cast:
        warnings.warn(
            "Data array used to create a new image contains 64-bit ints. "
            "This is likely due to creating the array with numpy and "
            "passing `int` as the `dtype`. Many tools such as FSL and SPM "
            "cannot deal with int64 in Nifti images, so for compatibility the "
            "data has been converted to int32.",
            stacklevel=find_stack_level(),
        )
        return data.astype("int32")
    warnings.warn(
        "Data array used to create a new image contains 64-bit ints, and "
        "some values too large to store in 32-bit ints. The resulting image "
        "thus contains 64-bit ints, which may cause some compatibility issues "
        "with some other tools or an error when saving the image to a "
        "Nifti file.",
        stacklevel=find_stack_level(),
    )
    return data


@fill_doc
def new_img_like(ref_niimg, data, affine=None, copy_header=True):
    """Create a new image of the same class as the reference image.

    Parameters
    ----------
    ref_niimg : Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Reference image. The new image will be of the same type.

    data : :class:`numpy.ndarray`, :obj:`~nilearn.surface.PolyData`, \
           or :obj:`dict` of  \
           :obj:`numpy.ndarray`, \
           :obj:`str`, \
           :obj:`pathlib.Path`

        Data to be stored in the image. If data dtype is a boolean, then data
        is cast to 'uint8' by default.

        .. nilearn_versionchanged:: 0.9.2
            Changed default dtype casting of booleans from 'int8' to 'uint8'.

        If ``ref_niimg`` is a Niimg-like object,
        then data must be a :class:`numpy.ndarray`.

    affine : 4x4 :class:`numpy.ndarray`, default=None
        Transformation matrix.
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    %(copy_header)s

    Returns
    -------
    Niimg-like or :obj:`~nilearn.surface.SurfaceImage` object
        A loaded image with the same file type (and, optionally, header)
        as the reference image.

    """
    if isinstance(ref_niimg, SurfaceImage):
        mesh = ref_niimg.mesh
        return SurfaceImage(
            mesh=deepcopy(mesh),
            data=data,
        )
    # Hand-written loading code to avoid too much memory consumption
    orig_ref_niimg = ref_niimg
    ref_niimg = stringify_path(ref_niimg)
    is_str = isinstance(ref_niimg, str)
    has_get_fdata = hasattr(ref_niimg, "get_fdata")
    has_iter = hasattr(ref_niimg, "__iter__")
    has_affine = hasattr(ref_niimg, "affine")
    if has_iter and not any([is_str, has_get_fdata]):
        ref_niimg = ref_niimg[0]
        ref_niimg = stringify_path(ref_niimg)
        is_str = isinstance(ref_niimg, str)
        has_get_fdata = hasattr(ref_niimg, "get_fdata")
        has_affine = hasattr(ref_niimg, "affine")
    if not (has_get_fdata and has_affine):
        if is_str:
            ref_niimg = load(ref_niimg)
        else:
            raise TypeError(
                "The reference image should be a niimg."
                f" {orig_ref_niimg!r} was passed"
            )

    if affine is None:
        affine = ref_niimg.affine
    if data.dtype == bool:
        data = as_ndarray(data, dtype=np.uint8)
    data = _downcast_from_int64_if_possible(data)
    header = None
    if copy_header and ref_niimg.header is not None:
        header = ref_niimg.header.copy()
        try:
            "something" in header  # noqa: B015
        except TypeError:
            pass
        else:
            if "scl_slope" in header:
                header["scl_slope"] = 0.0
            if "scl_inter" in header:
                header["scl_inter"] = 0.0
            # 'glmax' is removed for Nifti2Image. Modify only if 'glmax' is
            # available in header. See issue #1611
            if "glmax" in header:
                header["glmax"] = 0.0
            if "cal_max" in header:
                header["cal_max"] = np.max(data) if data.size > 0 else 0.0
            if "cal_min" in header:
                header["cal_min"] = np.min(data) if data.size > 0 else 0.0
    klass = ref_niimg.__class__
    if klass is Nifti1Pair:
        # Nifti1Pair is an internal class, without a to_filename,
        # we shouldn't return it
        klass = Nifti1Image
    return klass(data, affine, header=header)


def _apply_cluster_size_threshold(arr, cluster_threshold, copy=True):
    """Apply cluster-extent thresholding to voxel-wise thresholded array.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray` of shape (X, Y, Z)
        3D array that has been thresholded at the voxel level.

    %(cluster_threshold)s

    copy : :obj:`bool`, default=True
        Whether to copy the array before modifying it or not.

    Returns
    -------
    arr : :obj:`numpy.ndarray` of shape (X, Y, Z)
        Cluster-extent thresholded array.

    Notes
    -----
    Clusters are defined in a bi-sided manner;
    both negative and positive clusters are evaluated,
    but this is done separately for each sign.

    Clusters are defined using 6-connectivity, also known as NN1 (in AFNI) or
    "faces" connectivity.
    """
    check_params(locals())
    assert arr.ndim == 3

    if copy:
        arr = arr.copy()

    # Define array for 6-connectivity, aka NN1 or "faces"
    bin_struct = generate_binary_structure(3, 1)

    for sign in np.unique(np.sign(arr)):
        # Binarize using one-sided cluster-defining threshold
        binarized = ((arr * sign) > 0).astype(int)

        # Apply cluster threshold
        label_map = label(binarized, bin_struct)[0]
        clust_ids = sorted(np.unique(label_map)[1:])
        for c_val in clust_ids:
            if np.sum(label_map == c_val) < cluster_threshold:
                arr[label_map == c_val] = 0

    return arr


@fill_doc
def threshold_img(
    img,
    threshold,
    cluster_threshold: ClusterThreshold = 0,
    two_sided: bool = True,
    mask_img=None,
    copy: bool = True,
    copy_header: bool = True,
):
    """Threshold the given input image, mostly statistical or atlas images.

    Thresholding can be done based on direct image intensities or selection
    threshold with given percentile.

    - If ``threshold`` is a :obj:`float`:

      we threshold the image based on image intensities.

      - When ``two_sided`` is True:

        The given value should be within the range of minimum and maximum
        intensity of the input image.
        All intensities in the interval ``[-threshold, threshold]`` will be
        set to zero.

      - When ``two_sided`` is False:

        - If the threshold is negative:

          It should be greater than the minimum intensity of the input data.
          All intensities greater than or equal to the specified threshold will
          be set to zero.
          All other intensities keep their original values.

        - If the threshold is positive:

          then it should be less than the maximum intensity of the input data.
          All intensities less than or equal to the specified threshold will be
          set to zero.
          All other intensities keep their original values.

    - If threshold is :obj:`str`:

      The number part should be in interval ``[0, 100]``.
      We threshold the image based on the score obtained using this percentile
      on the image data.
      The percentile rank is computed using
      :func:`scipy.stats.scoreatpercentile`.

      - When ``two_sided`` is True:

        The score is calculated on the absolute values of data.

      - When ``two_sided`` is False:

        The score is calculated only on the non-negative values of data.

    .. nilearn_versionadded:: 0.2

    .. nilearn_versionchanged:: 0.9.0
        New ``cluster_threshold`` and ``two_sided`` parameters added.

    .. nilearn_versionchanged:: 0.12.0
        Add support for SurfaceImage.

    Parameters
    ----------
    img : a 3D/4D Niimg-like object or a :obj:`~nilearn.surface.SurfaceImage`
        Image containing statistical or atlas maps which should be thresholded.

    threshold : :obj:`float` or :obj:`str`
        Threshold that is used to set certain voxel intensities to zero.
        If threshold is float, it should be within the range of minimum and the
        maximum intensity of the data.
        If `two_sided` is True, threshold cannot be negative.
        If threshold is :obj:`str`,
        the given string should be
        within the range of ``"0%%"`` to ``"100%%"``.

    cluster_threshold : :obj:`float` or :obj:`int`, default=0
        Cluster size threshold, in voxels / vertices.
        In the returned thresholded map,
        sets of connected voxels / vertices (``clusters``)
        with size smaller than this number
        will be removed.

        .. nilearn_versionadded:: 0.9.0

    two_sided : :obj:`bool`, default=True
        Whether the thresholding should yield both positive and negative
        part of the maps.

        .. nilearn_versionadded:: 0.9.0

    mask_img : Niimg-like object or a :obj:`~nilearn.surface.SurfaceImage` \
        or None, default=None
        Mask image applied to mask the input data.
        If None, no masking will be applied.

    copy : :obj:`bool`, default=True
        If True, input array is not modified. True by default: the filtering
        is not performed in-place.

    %(copy_header)s

        .. nilearn_versionadded:: 0.11.0

    Returns
    -------
    :obj:`~nibabel.nifti1.Nifti1Image` \
        or a :obj:`~nilearn.surface.SurfaceImage`
        Thresholded image of the given input image.

    Raises
    ------
    ValueError
        If threshold is of type str but is not a non-negative number followed
        by the percent sign.
        If threshold is a negative float and `two_sided` is True.
    TypeError
        If threshold is neither float nor a string in correct percentile
        format.

    See Also
    --------
    nilearn.glm.threshold_stats_img :
        Threshold a statistical image using the alpha value, optionally with
        false positive control.

    """
    from nilearn.image.resampling import resample_img
    from nilearn.masking import load_mask_img

    check_params(locals())

    if not isinstance(img, (*NiimgLike, SurfaceImage)):
        raise TypeError(
            "'img' should be a 3D/4D Niimg-like object or a SurfaceImage. "
            f"Got {img.__class__.__name__}."
        )

    if mask_img is not None:
        check_compatibility_mask_and_images(mask_img, img)

    if isinstance(img, SurfaceImage) and isinstance(mask_img, SurfaceImage):
        check_polymesh_equal(mask_img.mesh, img.mesh)

    if isinstance(img, NiimgLike):
        img = check_niimg(img)
        img_data = safe_get_data(img, ensure_finite=True, copy_data=copy)
        affine = img.affine
    else:
        if copy:
            img = deepcopy(img)
        img_data = get_surface_data(img, ensure_finite=True)

    img_data_for_cutoff = img_data

    if mask_img is not None:
        # Set as 0 for the values which are outside of the mask
        if isinstance(mask_img, NiimgLike):
            mask_img = check_niimg_3d(mask_img)
            if not check_same_fov(img, mask_img):
                mask_img = resample_img(
                    mask_img,
                    target_affine=affine,
                    target_shape=img.shape[:3],
                    interpolation="nearest",
                )
            mask_data, _ = load_mask_img(mask_img)

            # Take only points that are within the mask to check for threshold
            img_data_for_cutoff = img_data_for_cutoff[mask_data != 0.0]

            img_data[mask_data == 0.0] = 0.0

        else:
            mask_img, _ = load_mask_img(mask_img)

            mask_data = get_surface_data(mask_img)

            # Take only points that are within the mask to check for threshold
            img_data_for_cutoff = img_data_for_cutoff[mask_data != 0.0]

            for hemi in mask_img.data.parts:
                mask = mask_img.data.parts[hemi]
                img.data.parts[hemi][mask == 0.0] = 0.0

    cutoff_threshold = check_threshold(
        threshold,
        img_data_for_cutoff,
        percentile_func=scoreatpercentile,
        name="threshold",
        two_sided=two_sided,
    )

    # Apply threshold
    if isinstance(img, NiimgLike):
        img_data = _apply_threshold(img_data, two_sided, cutoff_threshold)
    else:
        img_data = _apply_threshold(img, two_sided, cutoff_threshold)

    # Perform cluster thresholding, if requested

    # Expand to 4D to support both 3D and 4D nifti
    expand = isinstance(img, NiimgLike) and img_data.ndim == 3
    if expand:
        img_data = img_data[:, :, :, None]
    if cluster_threshold > 0:
        if isinstance(img, NiimgLike):
            for i_vol in range(img_data.shape[3]):
                img_data[..., i_vol] = _apply_cluster_size_threshold(
                    img_data[..., i_vol],
                    cluster_threshold,
                )
        else:
            for hemi in img_data.data.parts:
                clusters, labels = find_surface_clusters(
                    img_data.mesh.parts[hemi], img_data.data.parts[hemi]
                )
                # exclude only labels with size < min_size
                exclude_labels = set(
                    clusters.loc[clusters["size"] < cluster_threshold, "index"]
                )
                exclude = np.isin(labels, list(exclude_labels))
                img_data.data.parts[hemi][exclude] = 0

    if expand:
        # Reduce back to 3D
        img_data = img_data[:, :, :, 0]

    # Reconstitute img object
    if isinstance(img, NiimgLike):
        return new_img_like(img, img_data, affine, copy_header=copy_header)

    return new_img_like(img, img_data.data)


def _apply_threshold(img_data, two_sided, cutoff_threshold):
    """Apply a given threshold to an 'image'.

    If the image is a Surface applies to each part.

    Parameters
    ----------
    img_data: np.ndarray or SurfaceImage

    two_sided : :obj:`bool`, default=True
        Whether the thresholding should yield both positive and negative
        part of the maps.

    cutoff_threshold: :obj:`int`
        Effective threshold returned by check_threshold.

    Returns
    -------
    np.ndarray or SurfaceImage
    """
    if isinstance(img_data, SurfaceImage):
        for hemi, value in img_data.data.parts.items():
            img_data.data.parts[hemi] = _apply_threshold(
                value, two_sided, cutoff_threshold
            )
        return img_data

    if two_sided:
        mask = (-cutoff_threshold <= img_data) & (img_data <= cutoff_threshold)
    elif cutoff_threshold >= 0:
        mask = img_data <= cutoff_threshold
    else:
        mask = img_data >= cutoff_threshold

    img_data[mask] = 0.0

    return img_data


def math_img(formula, copy_header_from=None, **imgs):
    """Interpret a numpy based string formula using niimg in named parameters.

    .. nilearn_versionadded:: 0.2.3

    Parameters
    ----------
    formula : :obj:`str`
        The mathematical formula to apply to image internal data. It can use
        numpy imported as 'np'.

    copy_header_from : :obj:`str` or None, default=None
        Takes the variable name of one of the images in the formula.
        The header of this image will be copied to the result of the formula.
        Note that the result image and the image to copy the header from,
        should have the same number of dimensions.
        If None, the default
        :class:`~nibabel.nifti1.Nifti1Header` is used.

        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

        .. nilearn_versionadded:: 0.10.4

    imgs : images (:class:`~nibabel.nifti1.Nifti1Image` or file names \
           or :obj:`~nilearn.surface.SurfaceImage` object)
        Keyword arguments corresponding to the variables in the formula as
        images.
        All input images should have the same 'geometry':

        - shape and affine for volume data
        - mesh (coordinates and faces) for surface data

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image` \
    or :obj:`~nilearn.surface.SurfaceImage` object
        Result of the formula as an image.
        Note that the dimension of the result image
        can be smaller than the input image.
        For volume image input, the affine is the same as the input images.
        For surface image input, the mesh is the same as the input images.

    See Also
    --------
    nilearn.image.mean_img : To simply compute the mean of multiple images

    Examples
    --------
    Let's load an image using nilearn datasets module::

     >>> from nilearn import datasets
     >>> anatomical_image = datasets.load_mni152_template()

    Now we can use any numpy function on this image::

     >>> from nilearn.image import math_img
     >>> log_img = math_img("np.log(img)", img=anatomical_image)

    We can also apply mathematical operations on several images::

     >>> result_img = math_img("img1 + img2",
     ...                       img1=anatomical_image, img2=log_img)

    The result image will have the same shape and affine as the input images;
    but might have different header information, specifically the TR value,
    see :gh:`2645`.

    .. nilearn_versionadded:: 0.10.4

    We can also copy the header from one of the input images using
    ``copy_header_from``::

     >>> result_img_with_header = math_img("img1 + img2",
     ...                                   img1=anatomical_image, img2=log_img,
     ...                                   copy_header_from="img1")

    Notes
    -----
    This function is the Python equivalent of ImCal in SPM or fslmaths
    in FSL.

    """
    is_surface = all(isinstance(x, SurfaceImage) for x in imgs.values())

    if is_surface:
        first_img = next(iter(imgs.values()))
        for image in imgs.values():
            assert_polymesh_equal(first_img.mesh, image.mesh)

        # Computing input data as a dictionary of numpy arrays.
        data_dict = {k: {} for k in first_img.data.parts}
        for key, img in imgs.items():
            for k, v in img.data.parts.items():
                data_dict[k][key] = v

        # Add a reference to numpy in the kwargs of eval
        # so that numpy functions can be called from there.
        result = {}
        try:
            for k in data_dict:
                data_dict[k]["np"] = np
                result[k] = eval(formula, data_dict[k])
        except Exception as exc:
            exc.args = (
                "Input formula couldn't be processed, "
                f"you provided '{formula}',",
                *exc.args,
            )
            raise

        return new_img_like(first_img, result)

    try:
        niimgs = [check_niimg(image) for image in imgs.values()]
        check_same_fov(*niimgs, raise_error=True)
    except Exception as exc:
        exc.args = (
            "Input images cannot be compared, "
            f"you provided '{imgs.values()}',",
            *exc.args,
        )
        raise

    # Computing input data as a dictionary of numpy arrays. Keep a reference
    # niimg for building the result as a new niimg.
    niimg = None
    data_dict = {}
    for key, img in imgs.items():
        niimg = check_niimg(img)
        data_dict[key] = safe_get_data(niimg)

    # Add a reference to numpy in the kwargs of eval so that numpy functions
    # can be called from there.
    data_dict["np"] = np
    try:
        result = eval(formula, data_dict)
    except Exception as exc:
        exc.args = (
            f"Input formula couldn't be processed, you provided '{formula}',",
            *exc.args,
        )
        raise

    if copy_header_from is None:
        return new_img_like(niimg, result, niimg.affine, copy_header=False)
    niimg = check_niimg(imgs[copy_header_from])
    # only copy the header if the result and the input image to copy the
    # header from have the same shape
    if result.ndim != niimg.ndim:
        raise ValueError(
            "Cannot copy the header. "
            "The result of the formula has a different number of "
            "dimensions than the image to copy the header from."
        )
    return new_img_like(niimg, result, niimg.affine)


@fill_doc
def binarize_img(
    img, threshold=0.0, mask_img=None, two_sided=False, copy_header=True
):
    """Binarize an image such that its values are either 0 or 1.

    .. nilearn_versionadded:: 0.8.1

    Parameters
    ----------
    img : a 3D/4D Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Image which should be binarized.

    threshold : :obj:`float` or :obj:`str`, default=0.0
        If float, we threshold the image based on image intensities meaning
        voxels which have intensities greater than this value will be kept.
        The given value should be within the range of minimum and
        maximum intensity of the input image.
        If string, it should finish with percent sign e.g. "80%%" and we
        threshold based on the score obtained using this percentile on
        the image data. The voxels which have intensities greater than
        this score will be kept. The given string should be
        within the range of "0%%" to "100%%".

    mask_img : Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`, \
               default=None
        Mask image applied to mask the input data.
        If None, no masking will be applied.

    two_sided : :obj:`bool`, default=False
        If `True`, threshold is applied to the absolute value of the image.
        If `False`, threshold is applied to the original value of the image.

        .. nilearn_versionadded:: 0.10.3

        .. nilearn_versionchanged:: 0.13.0dev

            Default was changed to False.

     %(copy_header)s

        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

        .. nilearn_versionadded:: 0.11.0

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
    or :obj:`~nilearn.surface.SurfaceImage`
        Binarized version of the given input image. Output dtype is int8.

    See Also
    --------
    nilearn.image.threshold_img : To simply threshold but not binarize images.

    Examples
    --------
    Let's load an image using nilearn datasets module::

     >>> from nilearn import datasets
     >>> anatomical_image = datasets.load_mni152_template()

    Now we binarize it, generating a pseudo brainmask::

     >>> from nilearn.image import binarize_img
     >>> img = binarize_img(anatomical_image)

    """
    return math_img(
        "img.astype(bool).astype('int8')",
        img=threshold_img(
            img,
            threshold,
            mask_img=mask_img,
            two_sided=two_sided,
            copy_header=copy_header,
        ),
        copy_header_from="img",
    )


@fill_doc
def clean_img(
    imgs,
    runs=None,
    detrend=True,
    standardize=True,
    confounds=None,
    low_pass=None,
    high_pass=None,
    t_r=None,
    ensure_finite=False,
    mask_img=None,
    **kwargs,
):
    """Improve :term:`SNR` on masked :term:`fMRI` signals.

    This function can do several things on the input signals, in
    the following order:

    - detrend
    - low- and high-pass filter
    - remove confounds
    - standardize

    Low-pass filtering improves specificity.

    High-pass filtering should be kept small, to keep some sensitivity.

    Filtering is only meaningful on evenly-sampled signals.

    According to Lindquist et al. (2018), removal of confounds will be done
    orthogonally to temporal filters (low- and/or high-pass filters), if both
    are specified.

    .. nilearn_versionadded:: 0.2.5

    Parameters
    ----------
    imgs : 4D image Niimg-like object or \
           2D :obj:`~nilearn.surface.SurfaceImage`
        The signals in the last dimension are filtered (see
        :ref:`extracting_data` for a detailed description of the valid input
        types).

    runs : :class:`numpy.ndarray`, default=``None``
        Add a run level to the cleaning process. Each run will be
        cleaned independently. Must be a 1D array of n_samples elements.

        .. warning::

            'runs' replaces 'sessions' after release 0.10.0.
            Using 'session' will result in an error after release 0.10.0.


    detrend : :obj:`bool`, default=True
        If detrending should be applied on timeseries
        (before confound removal).

    %(standardize_true)s

    confounds : :class:`numpy.ndarray`, :obj:`str` or :obj:`list` of \
        Confounds timeseries. default=None
        Shape must be (instant number, confound number),
        or just (instant number,)
        The number of time instants in signals and confounds must be
        identical (i.e. signals.shape[0] == confounds.shape[0]).
        If a string is provided, it is assumed to be the name of a csv file
        containing signals as columns, with an optional one-line header.
        If a list is provided, all confounds are removed from the input
        signal, as if all were in the same array.

    %(low_pass)s

    %(high_pass)s

    t_r : :obj:`float`, default=None
        Repetition time, in second (sampling period). Set to None if not
        specified. Mandatory if used together with `low_pass` or `high_pass`.

    ensure_finite : :obj:`bool`, default=False
        If True, the non-finite values (NaNs and infs) found in the images
        will be replaced by zeros.

    mask_img : Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`,\
               default=None
        If provided, signal is only cleaned from voxels inside the mask.
        If mask is provided, it should have same shape and affine as imgs.
        If not provided, all voxels / verrices are used.
        See :ref:`extracting_data`.

    kwargs : :obj:`dict`
        Keyword arguments to be passed to functions called
        within this function.
        Kwargs prefixed with ``'clean__'`` will be passed to
        :func:`~nilearn.signal.clean`.
        Within :func:`~nilearn.signal.clean`, kwargs prefixed with
        ``'butterworth__'`` will be passed to the Butterworth filter
        (i.e., ``clean__butterworth__``).

    Returns
    -------
    Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
        Input images, cleaned. Same shape as `imgs`.

    Notes
    -----
    Confounds removal is based on a projection on the orthogonal
    of the signal space from :footcite:t:`Friston1994`.

    Orthogonalization between temporal filters and confound removal is based on
    suggestions in :footcite:t:`Lindquist2018`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
        nilearn.signal.clean

    """
    check_params(locals())
    # Avoid circular import
    from nilearn import masking

    # Check if t_r is set, otherwise propose t_r from imgs header
    if (low_pass is not None or high_pass is not None) and t_r is None:
        # We raise an error, instead of using the header's t_r as this
        # value is considered to be non-reliable
        raise ValueError(
            "Repetition time (t_r) must be specified for filtering. "
            "You specified None. "
            f"imgs header suggest it to be {imgs.header.get_zooms()[3]}"
        )

    clean_kwargs = {
        k[7:]: v for k, v in kwargs.items() if k.startswith("clean__")
    }

    if isinstance(imgs, SurfaceImage):
        imgs.data._check_ndims(2, "imgs")

        data = {}
        # Clean signal
        for p, v in imgs.data.parts.items():
            data[p] = signal.clean(
                v.T,
                runs=runs,
                detrend=detrend,
                standardize=standardize,
                confounds=confounds,
                low_pass=low_pass,
                high_pass=high_pass,
                t_r=t_r,
                ensure_finite=ensure_finite,
                **clean_kwargs,
            )
            data[p] = data[p].T

        if mask_img is not None:
            mask_img = masking.load_mask_img(mask_img)[0]
            for hemi in mask_img.data.parts:
                mask = mask_img.data.parts[hemi]
                data[hemi][mask == 0.0, ...] = 0.0

        return new_img_like(imgs, data)

    imgs_ = check_niimg_4d(imgs)

    # Prepare signal for cleaning
    if mask_img is not None:
        signals = masking.apply_mask(imgs_, mask_img)
    else:
        signals = get_data(imgs_).reshape(-1, imgs_.shape[-1]).T

    # Clean signal
    data = signal.clean(
        signals,
        runs=runs,
        detrend=detrend,
        standardize=standardize,
        confounds=confounds,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        ensure_finite=ensure_finite,
        **clean_kwargs,
    )

    # Put results back into Niimg-like object
    if mask_img is not None:
        imgs_ = masking.unmask(data, mask_img)
    elif "sample_mask" in clean_kwargs:
        sample_shape = imgs_.shape[:3] + clean_kwargs["sample_mask"].shape
        imgs_ = new_img_like(imgs_, data.T.reshape(sample_shape))
    else:
        imgs_ = new_img_like(imgs_, data.T.reshape(imgs_.shape))

    return imgs_


@fill_doc
def load_img(img, wildcards=True, dtype=None):
    """Load a Niimg-like object from filenames or list of filenames.

    .. nilearn_versionadded:: 0.2.5

    Parameters
    ----------
    img : Niimg-like object
        If string, consider it as a path to NIfTI image and
        call `nibabel.load()`on it.
        The '~' symbol is expanded to the user home folder.
        If it is an object, check if affine attribute is present, raise
        `TypeError` otherwise.
        See :ref:`extracting_data`.

    wildcards : :obj:`bool`, default=True
        Use `img` as a regular expression to get a list of matching input
        filenames.
        If multiple files match, the returned list is sorted using an ascending
        order.
        If no file matches the regular expression, a `ValueError` exception is
        raised.

    %(dtype)s

    Returns
    -------
    3D/4D Niimg-like object
        Result can be :class:`~nibabel.nifti1.Nifti1Image` or the input, as-is.
        It is guaranteed that
        the returned object has an affine attributes and that
        nilearn.image.get_data returns its data.

    """
    return check_niimg(img, wildcards=wildcards, dtype=dtype)


@fill_doc
def concat_imgs(
    niimgs,
    dtype=np.float32,
    ensure_ndim=None,
    memory=None,
    memory_level=0,
    auto_resample=False,
    verbose=0,
):
    """Concatenate a list of images of varying lengths.

    The image list can contain:

    - Niimg-like objects of varying dimensions (i.e., 3D or 4D)
      as well as different 3D shapes and affines,
      as they will be matched to the first image in the list
      if ``auto_resample=True``.

    - surface images of varying dimensions (i.e., 1D or 2D)
      but with same number of vertices

    Parameters
    ----------
    niimgs : iterable of Niimg-like objects, or glob pattern, \
             or :obj:`list` or :obj:`tuple` \
             of :obj:`~nilearn.surface.SurfaceImage` object
        See :ref:`extracting_data`.
        Images to concatenate.

    dtype : numpy dtype, default=np.float32
        The dtype of the returned image.

    ensure_ndim : :obj:`int`, default=None
        Indicate the dimensionality of the expected niimg. An
        error is raised if the niimg is of another dimensionality.
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    auto_resample : :obj:`bool`, default=False
        Converts all images to the space of the first one.
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    %(verbose0)s

    %(memory)s
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    %(memory_level)s
        Ignored for :obj:`~nilearn.surface.SurfaceImage`.

    Returns
    -------
    concatenated : :obj:`~nibabel.nifti1.Nifti1Image` \
                   or :obj:`~nilearn.surface.SurfaceImage`
        A single image.

    See Also
    --------
    nilearn.image.index_img

    """
    check_params(locals())

    if isinstance(niimgs, (tuple, list)) and len(niimgs) == 0:
        raise TypeError("Cannot concatenate empty objects.")

    if isinstance(niimgs, (tuple, list)) and all(
        isinstance(x, SurfaceImage) for x in niimgs
    ):
        if len(niimgs) == 1:
            return niimgs[0]

        for i, img in enumerate(niimgs):
            check_polymesh_equal(img.mesh, niimgs[0].mesh)
            niimgs[i] = at_least_2d(img)

        if dtype is None:
            dtype = extract_data(niimgs[0]).dtype

        output_data = {}
        for part in niimgs[0].data.parts:
            tmp = [img.data.parts[part] for img in niimgs]
            output_data[part] = np.concatenate(tmp, axis=1).astype(dtype)

        return new_img_like(niimgs[0], data=output_data)

    if memory is None:
        memory = Memory(location=None)

    target_fov = "first" if auto_resample else None

    # We remove one to the dimensionality because of the list is one dimension.
    ndim = None
    if ensure_ndim is not None:
        ndim = ensure_ndim - 1

    # If niimgs is a string, use glob to expand it to the matching filenames.
    niimgs = resolve_globbing(niimgs)

    # First niimg is extracted to get information and for new_img_like
    first_niimg = None

    iterator, literator = itertools.tee(iter(niimgs))
    try:
        first_niimg = check_niimg(next(literator), ensure_ndim=ndim)
    except StopIteration:
        raise TypeError("Cannot concatenate empty objects")
    except DimensionError as exc:
        # Keep track of the additional dimension in the error
        exc.increment_stack_counter()
        raise

    # If no particular dimensionality is asked, we force consistency wrt the
    # first image
    if ndim is None:
        ndim = len(first_niimg.shape)

    if ndim not in [3, 4]:
        raise TypeError(
            "Concatenated images must be 3D or 4D. You gave a "
            f"list of {ndim}D images"
        )

    lengths = [first_niimg.shape[-1] if ndim == 4 else 1]
    for niimg in literator:
        # We check the dimensionality of the niimg
        try:
            niimg = check_niimg(niimg, ensure_ndim=ndim)
        except DimensionError as exc:
            # Keep track of the additional dimension in the error
            exc.increment_stack_counter()
            raise
        lengths.append(niimg.shape[-1] if ndim == 4 else 1)

    target_shape = first_niimg.shape[:3]
    if dtype is None:
        dtype = _get_data(first_niimg).dtype
    data = np.ndarray((*target_shape, sum(lengths)), order="F", dtype=dtype)
    cur_4d_index = 0
    for index, (size, niimg) in enumerate(
        zip(
            lengths,
            iter_check_niimg(
                iterator,
                atleast_4d=True,
                target_fov=target_fov,
                memory=memory,
                memory_level=memory_level,
            ),
            strict=False,
        )
    ):
        nii_str = (
            f"image {niimg}" if isinstance(niimg, str) else f"image #{index}"
        )
        logger.log(f"Concatenating {index + 1}: {nii_str}", verbose)

        data[..., cur_4d_index : cur_4d_index + size] = _get_data(niimg)
        cur_4d_index += size

    return new_img_like(first_niimg, data, first_niimg.affine)


def largest_connected_component_img(imgs):
    """Return the largest connected component of an image or list of images.

    .. nilearn_versionadded:: 0.3.1

    Parameters
    ----------
    imgs : Niimg-like object or iterable of Niimg-like objects (3D)
        Image(s) to extract the largest connected component from.
        See :ref:`extracting_data`.

    Returns
    -------
    3D Niimg-like object or list of
        Image or list of images containing the largest connected component.

    Notes
    -----
    **Handling big-endian in given Nifti image**
    This function changes the existing byte-ordering information to new byte
    order, if the dtype in given Nifti image has non-native data type.
    This operation is done internally to avoid big-endian issues with
    scipy ndimage module.

    """
    from .._utils.ndimage import largest_connected_component

    imgs = stringify_path(imgs)
    if hasattr(imgs, "__iter__") and not isinstance(imgs, str):
        single_img = False
    else:
        single_img = True
        imgs = [imgs]

    ret = []
    for img in imgs:
        img = check_niimg_3d(img)
        affine = img.affine
        largest_component = largest_connected_component(safe_get_data(img))
        ret.append(new_img_like(img, largest_component, affine))

    return ret[0] if single_img else ret


def copy_img(img):
    """Copy an image to a nibabel.Nifti1Image.

    Parameters
    ----------
    img : image
        nibabel SpatialImage object to copy.

    Returns
    -------
    img_copy : image
        copy of input (data, affine and header)
    """
    check_is_of_allowed_type(img, (spatialimages.SpatialImage,), "img")
    return new_img_like(
        img, safe_get_data(img, copy_data=True), img.affine.copy()
    )


def _check_fov(img, affine, shape) -> bool:
    """Return True if img's field of view correspond to given \
    shape and affine, False elsewhere.
    """
    return img.shape[:3] == shape and np.allclose(img.affine, affine)


def check_same_fov(*args, **kwargs) -> bool:
    """Return True if provided images have the same field of view (shape and \
    affine) and return False or raise an error elsewhere, depending on the \
    `raise_error` argument.

    This function can take an unlimited number of
    images as arguments or keyword arguments and raise a user-friendly
    ValueError if asked.

    Parameters
    ----------
    args : images
        Images to be checked. Images passed without keywords will be labeled
        as img_#1 in the error message (replace 1 with the appropriate index).

    kwargs : images
        Images to be checked. In case of error, images will be referenced by
        their keyword name in the error message.

    raise_error : :obj:`bool`, optional
        If True, an error will be raised in case of error.

    """
    raise_error = kwargs.pop("raise_error", False)
    for i, arg in enumerate(args):
        kwargs[f"img_#{i}"] = arg
    errors = []
    for (a_name, a_img), (b_name, b_img) in itertools.combinations(
        kwargs.items(), 2
    ):
        if a_img.shape[:3] != b_img.shape[:3]:
            errors.append((a_name, b_name, "shape"))
        if not np.allclose(a_img.affine, b_img.affine):
            errors.append((a_name, b_name, "affine"))
    if errors and raise_error:
        raise ValueError(
            "Following field of view errors were detected:\n"
            + "\n".join(
                [
                    f"- {e[0]} and {e[1]} do not have the same {e[2]}"
                    for e in errors
                ]
            )
        )
    return not errors


def check_imgs_equal(img1, img2) -> bool:
    """Check if 2 NiftiImages have same fov and data."""
    if not check_same_fov(img1, img2, raise_error=False):
        return False

    data_img1 = safe_get_data(img1)
    data_img2 = safe_get_data(img2)

    try:
        assert_array_equal(data_img1, data_img2)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e


@fill_doc
def iter_check_niimg(
    niimgs,
    ensure_ndim=None,
    atleast_4d=False,
    target_fov=None,
    dtype=None,
    memory=None,
    memory_level=0,
):
    """Iterate over a list of niimgs and do sanity checks and resampling.

    Parameters
    ----------
    niimgs : :obj:`list` of niimg or glob pattern or itertools.tee instance
        Images to iterate over.

    ensure_ndim : :obj:`int`, default=None
        If specified, an error is raised if the data does not have the
        required dimension.

    atleast_4d : :obj:`bool`, default=False
        If True, any 3D image is converted to a 4D single scan.

    target_fov : :obj:`tuple` of affine and shape, or None, default=None
       If specified, images are resampled to this field of view.

    %(dtype)s

    %(memory)s
        default=None
        If ``None`` is passed will default to ``Memory(location=None)``.

    %(memory_level)s

    See Also
    --------
        check_niimg, check_niimg_3d, check_niimg_4d

    """
    # avoid circular import
    from nilearn.image.resampling import resample_img

    if memory is None:
        memory = Memory(location=None)
    # If niimgs is a string, use glob to expand it to the matching filenames.
    niimgs = resolve_globbing(niimgs)

    ref_fov = None
    resample_to_first_img = False
    ndim_minus_one = ensure_ndim - 1 if ensure_ndim is not None else None
    if target_fov is not None and target_fov != "first":
        ref_fov = target_fov

    i = -1
    for i, niimg in enumerate(niimgs):
        if isinstance(niimg, SurfaceImage):
            # TODO do some checks
            yield niimg

        else:
            try:
                niimg = check_niimg(
                    niimg,
                    ensure_ndim=ndim_minus_one,
                    atleast_4d=atleast_4d,
                    dtype=dtype,
                )
                if i == 0:
                    ndim_minus_one = len(niimg.shape)
                    if ref_fov is None:
                        ref_fov = (niimg.affine, niimg.shape[:3])
                        resample_to_first_img = True

                if not _check_fov(niimg, ref_fov[0], ref_fov[1]):
                    if target_fov is None:
                        raise ValueError(
                            f"Field of view of image #{i} is different from "
                            "reference FOV.\n"
                            f"Reference affine:\n{ref_fov[0]!r}\n"
                            f"Image affine:\n{niimg.affine!r}\n"
                            f"Reference shape:\n{ref_fov[1]!r}\n"
                            f"Image shape:\n{niimg.shape!r}\n"
                        )

                    if resample_to_first_img:
                        warnings.warn(
                            "Affine is different across subjects."
                            " Realignment on first subject "
                            "affine forced",
                            stacklevel=find_stack_level(),
                        )
                    niimg = cache(
                        resample_img,
                        memory,
                        func_memory_level=2,
                        memory_level=memory_level,
                    )(
                        niimg,
                        target_affine=ref_fov[0],
                        target_shape=ref_fov[1],
                    )
                yield niimg
            except DimensionError as exc:
                # Keep track of the additional dimension in the error
                exc.increment_stack_counter()
                raise
            except TypeError as exc:
                img_name = (
                    f" ({niimg}) " if isinstance(niimg, (str, Path)) else ""
                )

                exc.args = (
                    f"Error encountered while loading image #{i}{img_name}",
                    *exc.args,
                )
                raise

    # Raising an error if input generator is empty.
    if i == -1:
        raise ValueError("Input niimgs list is empty.")


# ensure_ndim = 3 always returns a NiftiImage
@overload
def check_niimg(
    niimg,
    ensure_ndim: Literal[3] = ...,
    atleast_4d: Literal[False] = ...,
    dtype=...,
    return_iterator: Literal[False] = ...,
    wildcards=...,
) -> Nifti1Image: ...


# ensure_ndim = 4 with return_iterator=False always returns a NiftiImage
@overload
def check_niimg(
    niimg,
    ensure_ndim: Literal[4] = ...,
    atleast_4d=...,
    dtype=...,
    return_iterator: Literal[False] = ...,
    wildcards=...,
) -> Nifti1Image: ...


# ensure_ndim = 4 with return_iterator=True
# always returns an iterator NiftiImage
@overload
def check_niimg(
    niimg,
    ensure_ndim: Literal[4] = ...,
    atleast_4d=...,
    dtype=...,
    return_iterator: Literal[True] = ...,
    wildcards=...,
) -> Iterable[Nifti1Image]: ...


@fill_doc
def check_niimg(
    niimg: Any,
    ensure_ndim: Literal[3, 4] | None = None,
    atleast_4d: bool = False,
    dtype: Any = None,
    return_iterator: bool = False,
    wildcards: bool = True,
) -> Nifti1Image | Iterable[Nifti1Image]:
    """Check that niimg is a proper 3D/4D niimg.

    Turn filenames into objects.

    Parameters
    ----------
    niimg : Niimg-like object
        See :ref:`extracting_data`.
        If niimg is a string or :obj:`pathlib.Path`,
        consider it as a path to Nifti image
        and call nibabel.load on it.
        The ``'~'`` symbol is expanded to the user home folder.
        If it is an object, check if the affine attribute present
        and that :func:`nilearn.image.get_data` returns a result,
        raise :obj:`TypeError` otherwise.

    ensure_ndim : {3, 4, None}, default=None
        Indicate the dimensionality of the expected niimg.
        An error is raised if the niimg is of another dimensionality.

    atleast_4d : :obj:`bool`, default=False
        Indicates if a 3d image should be turned into a single-scan 4d niimg.

    %(dtype)s

    return_iterator : :obj:`bool`, default=False
        Returns an iterator on the content of the niimg file input.

    wildcards : :obj:`bool`, default=True
        Use niimg as a regular expression
        to get a list of matching input filenames.
        If multiple files match,
        the returned list is sorted using an ascending order.
        If no file matches the regular expression,
        a :obj:`ValueError` exception is raised.

    Returns
    -------
    result : 3D/4D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is.
        It is guaranteed that the returned object
        has an affine attribute
        and that its data can be retrieved
        with :func:`nilearn.image.get_data`.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite
    for any data processing method in Nilearn
    because it checks if data have a correct format
    and loads them if necessary.

    Its application is idempotent.

    See Also
    --------
        check_niimg_3d, check_niimg_4d

    """
    if not (
        isinstance(niimg, (NiimgLike, SpatialImage))
        or (hasattr(niimg, "__iter__"))
    ):
        raise TypeError(
            "input should be a NiftiLike object "
            "or an iterable of NiftiLike object. "
            f"Got: {niimg.__class__.__name__}"
        )

    if hasattr(niimg, "__iter__"):
        for x in niimg:
            if not (
                isinstance(x, (NiimgLike, SpatialImage))
                or hasattr(x, "__iter__")
            ):
                raise TypeError(
                    "iterable inputs should contain "
                    "NiftiLike objects or iterables. "
                    f"Got: {x.__class__.__name__}"
                )

    niimg = stringify_path(niimg)

    if isinstance(niimg, str):
        if wildcards and EXPAND_PATH_WILDCARDS:
            # Expand user path
            expanded_niimg = str(Path(niimg).expanduser())
            # Ascending sorting
            filenames = sorted(glob.glob(expanded_niimg))

            # processing filenames matching globbing expression
            if len(filenames) >= 1 and glob.has_magic(niimg):
                niimg = filenames  # iterable case
            # niimg is an existing filename
            elif [expanded_niimg] == filenames:
                niimg = filenames[0]
            # No files found by glob
            elif glob.has_magic(niimg):
                # No files matching the glob expression, warn the user
                message = (
                    "No files matching the entered niimg expression: "
                    f"'{niimg}'.\n"
                    "You may have left wildcards usage activated: "
                    "please set the global constant "
                    "'nilearn.EXPAND_PATH_WILDCARDS' to False "
                    "to deactivate this behavior."
                )
                raise ValueError(message)
            else:
                raise ValueError(f"File not found: '{niimg}'")
        elif not Path(niimg).exists():
            raise ValueError(f"File not found: '{niimg}'")

    # in case of an iterable
    if hasattr(niimg, "__iter__") and not isinstance(niimg, str):
        if return_iterator:
            return iter_check_niimg(
                niimg, ensure_ndim=ensure_ndim, dtype=dtype
            )
        return concat_imgs(niimg, ensure_ndim=ensure_ndim, dtype=dtype)

    # Otherwise, it should be a filename or a SpatialImage, we load it
    niimg = load_niimg(niimg, dtype=dtype)

    if ensure_ndim == 3 and len(niimg.shape) == 4 and niimg.shape[3] == 1:
        # "squeeze" the image.
        data = safe_get_data(niimg)
        affine = niimg.affine
        niimg = new_img_like(niimg, data[:, :, :, 0], affine)

    if len(niimg.shape) == 3:
        # This a rare edge case where return_iterator is True
        # but the input image is 3D.
        if return_iterator and not atleast_4d:
            raise DimensionError(len(niimg.shape), required_dimension=4)

        if atleast_4d:
            data = _get_data(niimg).view()
            data.shape = (*data.shape, 1)
            niimg = new_img_like(niimg, data, niimg.affine)

    if ensure_ndim is not None and len(niimg.shape) != ensure_ndim:
        raise DimensionError(len(niimg.shape), ensure_ndim)

    if return_iterator:
        return (_index_img(niimg, i) for i in range(niimg.shape[3]))

    return niimg


@fill_doc
def check_niimg_3d(niimg: Any, dtype: Any = None) -> Nifti1Image:
    """Check that niimg is a proper 3D niimg-like object and load it.

    Parameters
    ----------
    niimg : Niimg-like object
        See :ref:`extracting_data`.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it.
        If it is an object, check if the affine attribute present and that
        :func:`nilearn.image.get_data` returns a result,
        raise :obj:`TypeError` otherwise.

    %(dtype)s

    Returns
    -------
    result : 3D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is.
        It is guaranteed that the returned object has an affine attribute
        and that its data can be retrieved
        with :func:`nilearn.image.get_data`.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite
    for any data processing method in Nilearn
    because it checks if data have a correct format
    and loads them if necessary.

    Its application is idempotent.

    """
    return check_niimg(niimg, ensure_ndim=3, dtype=dtype)


@overload
def check_niimg_4d(
    niimg: Any,
    return_iterator: Literal[False] = ...,
    dtype: Any = ...,
) -> Nifti1Image: ...


@overload
def check_niimg_4d(
    niimg: Any,
    return_iterator: Literal[True] = ...,
    dtype: Any = ...,
) -> Iterable[Nifti1Image]: ...


@fill_doc
def check_niimg_4d(
    niimg: Any,
    return_iterator: Literal[False, True] = False,
    dtype: Any = None,
):
    """Check that niimg is a proper 4D niimg-like object and load it.

    Parameters
    ----------
    niimg : 4D Niimg-like object
        See :ref:`extracting_data`.
        If ``niimgs`` is an iterable, checks if data is really 4D.
        Then, considering that it is a list of niimg and load them one by one.
        If ``niimgs`` is a string,
        consider it as a path to Nifti image
        and call nibabel.load on it.
        If it is an object, check if the affine attribute present
        and that :func:`nilearn.image.get_data` returns a result,
        raise :obj:`TypeError` otherwise.

    return_iterator : :obj:`bool`, default=False
        If True, an iterator of 3D images is returned.
        This reduces the memory usage when ``niimgs`` contains 3D images.
        If False, a single 4D image is returned.
        When ``niimgs`` contains 3D images they are concatenated together.

    %(dtype)s.

    Returns
    -------
    niimg: 4D nibabel.Nifti1Image or iterator of 3D nibabel.Nifti1Image

    Notes
    -----
    This function is the equivalent to :func:`nilearn.image.check_niimg_3d()`
    for Niimg-like objects with a run level.

    Its application is idempotent.

    """
    ensure_ndim: Literal[4] = 4
    return check_niimg(
        niimg,
        ensure_ndim=ensure_ndim,
        return_iterator=return_iterator,
        dtype=dtype,
    )


def get_indices_from_image(image) -> np.ndarray:
    """Return unique values in a label image."""
    if isinstance(image, NiimgLike):
        img = check_niimg(image)
        data = safe_get_data(img)
    elif isinstance(image, SurfaceImage):
        data = get_surface_data(image)
    elif isinstance(image, np.ndarray):
        data = image
    else:
        raise TypeError(
            "Image to extract indices from must be one of: "
            "Niimg-Like, SurfaceImage, numpy array. "
            f"Got {image.__class__.__name__}"
        )

    labels_present = np.unique(data)

    return labels_present[np.isfinite(labels_present)]
