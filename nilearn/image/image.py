"""
Preprocessing functions for images.

See also nilearn.signal.
"""

# Authors: Philippe Gervais, Alexandre Abraham

import collections.abc
import copy
import itertools
import warnings

import numpy as np
from joblib import Memory, Parallel, delayed
from nibabel import Nifti1Image, Nifti1Pair, load, spatialimages
from scipy.ndimage import gaussian_filter1d, generate_binary_structure, label
from scipy.stats import scoreatpercentile

from .. import signal
from .._utils import (
    as_ndarray,
    check_niimg,
    check_niimg_3d,
    check_niimg_4d,
    fill_doc,
    logger,
    repr_niimgs,
)
from .._utils.exceptions import DimensionError
from .._utils.helpers import (
    check_copy_header,
    stringify_path,
)
from .._utils.niimg import _get_data, safe_get_data
from .._utils.niimg_conversions import (
    _index_img,
    check_same_fov,
    iter_check_niimg,
)
from .._utils.param_validation import check_threshold
from .._utils.path_finding import resolve_globbing


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
):
    """Return confounds extracted from input signals with highest variance.

    Parameters
    ----------
    imgs : Niimg-like object
        4D image.
        See :ref:`extracting_data`.

    mask_img : Niimg-like object
        If not provided, all voxels are used.
        If provided, confounds are extracted from voxels inside the mask.
        See :ref:`extracting_data`.

    n_confounds : :obj:`int`, default=5
        Number of confounds to return.

    percentile : :obj:`float`, default=2
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

    if mask_img is not None:
        sigs = masking.apply_mask(imgs, mask_img)
    else:
        # Load the data only if it doesn't need to be masked
        imgs = check_niimg_4d(imgs)
        sigs = as_ndarray(get_data(imgs))
        # Not using apply_mask here saves memory in most cases.
        del imgs  # help reduce memory consumption
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T

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
            "Setting it to None (no smoothing will be performed)"
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
        ret.append(new_img_like(img, filtered, affine, copy_header=True))

    return ret[0] if single_img else ret


def _crop_img_to(img, slices, copy=True, copy_header=False):
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

    copy_header : :obj:`bool`
        Whether to copy the header of the input image to the output.
        If None, the default behavior is to not copy the header.

        .. versionadded:: 0.11.0

        This parameter will be set to True by default in 0.13.0.

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


def crop_img(
    img, rtol=1e-8, copy=True, pad=True, return_offset=False, copy_header=False
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

    rtol : :obj:`float`, default=1e-8
        relative tolerance (with respect to maximal absolute value of the
        image), under which values are considered negligeable and thus
        croppable.

    copy : :obj:`bool`, default=True
        Specifies whether cropped data is copied or not.

    pad : :obj:`bool`, default=True
        Toggles adding 1-voxel of 0s around the border.

    return_offset : :obj:`bool`, default=False
        Specifies whether to return a tuple of the removed padding.

    copy_header : :obj:`bool`, default=False
        Whether to copy the header of the input image to the output.

        .. versionadded:: 0.11.0

        This parameter will be set to True by default in 0.13.0.

    Returns
    -------
    Niimg-like object or :obj:`tuple`
        Cropped version of the input image and, if `return_offset=True`,
        a tuple of tuples representing the number of voxels
        removed (before, after) the cropped volumes, i.e.:
        *[(x1_pre, x1_post), (x2_pre, x2_post), ..., (xN_pre, xN_post)]*

    """
    # TODO: remove this warning in 0.13.0
    check_copy_header(copy_header)

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
    else:
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    if pad:
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)][:3]
    cropped_im = _crop_img_to(img, slices, copy=copy, copy_header=copy_header)
    return (cropped_im, tuple(slices)) if return_offset else cropped_im


def _pad_array(array, pad_sizes):
    """Pad an array with zeros.

    Pads an array with zeros as specified in `pad_sizes`.

    Parameters
    ----------
    array : :class:`numpy.ndarray`
        Array to pad.

    pad_sizes : :obj:`list`
        Padding quantity specified as
        *[x1minpad, x1maxpad, x2minpad,x2maxpad, x3minpad, ...]*.

    Returns
    -------
    :class:`numpy.ndarray`
        Padded array.

    Raises
    ------
    ValueError
        Inconsistent min/max padding quantities.

    """
    if len(pad_sizes) % 2 != 0:
        raise ValueError(
            "Please specify as many max paddings as min"
            f" paddings. You have specified {len(pad_sizes)} arguments"
        )

    all_paddings = np.zeros([array.ndim, 2], dtype=np.int64)
    all_paddings[: len(pad_sizes) // 2] = np.array(pad_sizes).reshape(-1, 2)

    lower_paddings, upper_paddings = all_paddings.T
    new_shape = np.array(array.shape) + upper_paddings + lower_paddings

    padded = np.zeros(new_shape, dtype=array.dtype)
    source_slices = [
        slice(max(-lp, 0), min(s + up, s))
        for lp, up, s in zip(lower_paddings, upper_paddings, array.shape)
    ]
    target_slices = [
        slice(max(lp, 0), min(s - up, s))
        for lp, up, s in zip(lower_paddings, upper_paddings, new_shape)
    ]

    padded[tuple(target_slices)] = array[tuple(source_slices)].copy()
    return padded


def _compute_mean(imgs, target_affine=None, target_shape=None, smooth=False):
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
    # TODO switch to force_resample=True
    # when bumping to version > 0.13
    mean_data = resampling.resample_img(
        Nifti1Image(mean_data, affine),
        target_affine=target_affine,
        target_shape=target_shape,
        copy=False,
        copy_header=True,
        force_resample=False,
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


def mean_img(
    imgs,
    target_affine=None,
    target_shape=None,
    verbose=0,
    n_jobs=1,
    copy_header=False,
):
    """Compute the mean of the images over time or the 4th dimension.

    Note that if list of 4D images are given, the mean of each 4D image is
    computed separately, and the resulting mean is computed after.

    Parameters
    ----------
    imgs : Niimg-like object or iterable of Niimg-like objects
        Images to be averaged over time (see :ref:`extracting_data`
        for a detailed description of the valid input types).

    target_affine : :class:`numpy.ndarray`, optional
        If specified, the image is resampled corresponding to this new affine.
        target_affine can be a 3x3 or a 4x4 matrix.

    target_shape : :obj:`tuple` or :obj:`list`, optional
        If specified, the image will be resized to match this new shape.
        len(target_shape) must be equal to 3.
        A target_affine has to be specified jointly with target_shape.

    verbose : :obj:`int`, default=0
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages).

    n_jobs : :obj:`int`, default=1
        The number of CPUs to use to do the computation (-1 means
        'all CPUs').

    copy_header : :obj:`bool`, default=False
        Whether to copy the header of the input image to the output.

        .. versionadded:: 0.11.0

        This parameter will be set to True by default in 0.13.0.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Mean image.

    See Also
    --------
    nilearn.image.math_img : For more general operations on images.

    """
    # TODO: remove this warning in 0.13.0
    check_copy_header(copy_header)

    imgs = stringify_path(imgs)
    is_str = isinstance(imgs, str)
    is_iterable = isinstance(imgs, collections.abc.Iterable)
    if is_str or not is_iterable:
        imgs = [
            imgs,
        ]

    imgs_iter = iter(imgs)
    first_img = check_niimg(next(imgs_iter))

    # Compute the first mean to retrieve the reference
    # target_affine and target_shape if_needed
    n_imgs = 1
    running_mean, first_affine = _compute_mean(
        first_img, target_affine=target_affine, target_shape=target_shape
    )

    if target_affine is None or target_shape is None:
        target_affine = first_affine
        target_shape = running_mean.shape[:3]

    for this_mean in Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_compute_mean)(
            n, target_affine=target_affine, target_shape=target_shape
        )
        for n in imgs_iter
    ):
        n_imgs += 1
        # _compute_mean returns (mean_img, affine)
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
    img = reorder_img(img, copy_header=True)

    # create swapped nifti object
    out_img = new_img_like(
        img, get_data(img)[::-1], img.affine, copy_header=True
    )

    return out_img


def index_img(imgs, index):
    """Indexes into a 4D Niimg-like object in the fourth dimension.

    Common use cases include extracting a 3D image out of `img` or
    creating a 4D image whose data is a subset of `img` data.

    Parameters
    ----------
    imgs : 4D Niimg-like object
        See :ref:`extracting_data`.

    index : Any type compatible with numpy array indexing
        Used for indexing the 4D data array in the fourth dimension.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
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
    imgs = check_niimg_4d(imgs)
    # duck-type for pandas arrays, and select the 'values' attr
    if hasattr(index, "values") and hasattr(index, "iloc"):
        index = index.to_numpy().flatten()
    return _index_img(imgs, index)


def iter_img(imgs):
    """Iterate over a 4D Niimg-like object in the fourth dimension.

    Parameters
    ----------
    imgs : 4D Niimg-like object
        See :ref:`extracting_data`.

    Returns
    -------
    Iterator of 3D :class:`~nibabel.nifti1.Nifti1Image`

    See Also
    --------
    nilearn.image.index_img

    """
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
            stacklevel=3,
        )
        return data.astype("int32")
    warnings.warn(
        "Data array used to create a new image contains 64-bit ints, and "
        "some values too large to store in 32-bit ints. The resulting image "
        "thus contains 64-bit ints, which may cause some compatibility issues "
        "with some other tools or an error when saving the image to a "
        "Nifti file.",
        stacklevel=3,
    )
    return data


def new_img_like(ref_niimg, data, affine=None, copy_header=False):
    """Create a new image of the same class as the reference image.

    Parameters
    ----------
    ref_niimg : Niimg-like object
        Reference image. The new image will be of the same type.

    data : :class:`numpy.ndarray`
        Data to be stored in the image. If data dtype is a boolean, then data
        is cast to 'uint8' by default.

        .. versionchanged:: 0.9.2
            Changed default dtype casting of booleans from 'int8' to 'uint8'.

    affine : 4x4 :class:`numpy.ndarray`, optional
        Transformation matrix.

    copy_header : :obj:`bool`, default=False
        Indicated if the header of the reference image should be used to
        create the new image.

    Returns
    -------
    Niimg-like object
        A loaded image with the same file type (and, optionally, header)
        as the reference image.

    """
    # Hand-written loading code to avoid too much memory consumption
    orig_ref_niimg = ref_niimg
    ref_niimg = stringify_path(ref_niimg)
    is_str = isinstance(ref_niimg, str)
    has_get_data = hasattr(ref_niimg, "get_data")
    has_get_fdata = hasattr(ref_niimg, "get_fdata")
    has_iter = hasattr(ref_niimg, "__iter__")
    has_affine = hasattr(ref_niimg, "affine")
    if has_iter and not any([is_str, has_get_data, has_get_fdata]):
        ref_niimg = ref_niimg[0]
        ref_niimg = stringify_path(ref_niimg)
        is_str = isinstance(ref_niimg, str)
        has_get_data = hasattr(ref_niimg, "get_data")
        has_get_fdata = hasattr(ref_niimg, "get_fdata")
        has_affine = hasattr(ref_niimg, "affine")
    if not ((has_get_data or has_get_fdata) and has_affine):
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
    if copy_header:
        header = copy.deepcopy(ref_niimg.header)
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
    cluster_threshold : :obj:`float`
        Cluster-size threshold, in voxels, to apply to ``arr``.
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


def threshold_img(
    img,
    threshold,
    cluster_threshold=0,
    two_sided=True,
    mask_img=None,
    copy=True,
    copy_header=False,
):
    """Threshold the given input image, mostly statistical or atlas images.

    Thresholding can be done based on direct image intensities or selection
    threshold with given percentile.

    .. versionchanged:: 0.9.0
        New ``cluster_threshold`` and ``two_sided`` parameters added.

    .. versionadded:: 0.2

    Parameters
    ----------
    img : a 3D/4D Niimg-like object
        Image containing statistical or atlas maps which should be thresholded.

    threshold : :obj:`float` or :obj:`str`
        Voxels with intensities less than the requested threshold
        will be set to zero.
        Those with intensities greater or equal than the requested threshold
        will keep their original value.
        If float, we threshold the image based on image intensities.
        The given value should be within the range of minimum and maximum
        intensity of the input image.
        If string, it should finish with percent sign e.g. "80%"
        and we threshold based on the score obtained
        using this percentile on the image data.
        The given string should be within the range of "0%" to "100%".
        The percentile rank is computed using
        :func:`scipy.stats.scoreatpercentile`.

    cluster_threshold : :obj:`float`, default=0
        Cluster size threshold, in voxels. In the returned thresholded map,
        sets of connected voxels (``clusters``) with size smaller
        than this number will be removed.

        .. versionadded:: 0.9.0

    two_sided : :obj:`bool`, default=True
        Whether the thresholding should yield both positive and negative
        part of the maps.

        .. versionadded:: 0.9.0

    mask_img : Niimg-like object, default=None
        Mask image applied to mask the input data.
        If None, no masking will be applied.

    copy : :obj:`bool`, default=True
        If True, input array is not modified. True by default: the filtering
        is not performed in-place.

    copy_header : :obj:`bool`, default=False
        Whether to copy the header of the input image to the output.

        .. versionadded:: 0.11.0

        This parameter will be set to True by default in 0.13.0.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Thresholded image of the given input image.

    See Also
    --------
    nilearn.glm.threshold_stats_img :
        Threshold a statistical image using the alpha value, optionally with
        false positive control.

    """
    from .. import masking
    from . import resampling

    # TODO: remove this warning in 0.13.0
    check_copy_header(copy_header)

    img = check_niimg(img)
    img_data = safe_get_data(img, ensure_finite=True, copy_data=copy)
    affine = img.affine

    if mask_img is not None:
        mask_img = check_niimg_3d(mask_img)
        if not check_same_fov(img, mask_img):
            # TODO switch to force_resample=True
            # when bumping to version > 0.13
            mask_img = resampling.resample_img(
                mask_img,
                target_affine=affine,
                target_shape=img.shape[:3],
                interpolation="nearest",
                copy_header=True,
                force_resample=False,
            )

        mask_data, _ = masking.load_mask_img(mask_img)
        # Set as 0 for the values which are outside of the mask
        img_data[mask_data == 0.0] = 0.0

    cutoff_threshold = check_threshold(
        threshold,
        img_data,
        percentile_func=scoreatpercentile,
        name="threshold",
    )

    # Apply threshold
    if two_sided:
        img_data[np.abs(img_data) < cutoff_threshold] = 0.0
    else:
        img_data[img_data < cutoff_threshold] = 0.0

    # Expand to 4D to support both 3D and 4D
    expand_to_4d = img_data.ndim == 3
    if expand_to_4d:
        img_data = img_data[:, :, :, None]

    # Perform cluster thresholding, if requested
    if cluster_threshold > 0:
        for i_vol in range(img_data.shape[3]):
            img_data[..., i_vol] = _apply_cluster_size_threshold(
                img_data[..., i_vol],
                cluster_threshold,
            )

    if expand_to_4d:
        # Reduce back to 3D
        img_data = img_data[:, :, :, 0]

    # Reconstitute img object
    thresholded_img = new_img_like(
        img, img_data, affine, copy_header=copy_header
    )

    return thresholded_img


def math_img(formula, copy_header_from=None, **imgs):
    """Interpret a numpy based string formula using niimg in named parameters.

    .. versionadded:: 0.2.3

    Parameters
    ----------
    formula : :obj:`str`
        The mathematical formula to apply to image internal data. It can use
        numpy imported as 'np'.

    copy_header_from : :obj:`str`, default=None
        Takes the variable name of one of the images in the formula.
        The header of this image will be copied to the result of the formula.
        Note that the result image and the image to copy the header from,
        should have the same number of dimensions. If None, the default
        :class:`~nibabel.nifti1.Nifti1Header` is used.

        .. versionadded:: 0.10.4

    imgs : images (:class:`~nibabel.nifti1.Nifti1Image` or file names)
        Keyword arguments corresponding to the variables in the formula as
        Nifti images. All input images should have the same geometry (shape,
        affine).

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Result of the formula as a Nifti image. Note that the dimension of the
        result image can be smaller than the input image. The affine is the
        same as the input image.

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

    .. versionadded:: 0.10.4

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
        return new_img_like(niimg, result, niimg.affine)
    niimg = check_niimg(imgs[copy_header_from])
    # only copy the header if the result and the input image to copy the
    # header from have the same shape
    if result.ndim != niimg.ndim:
        raise ValueError(
            "Cannot copy the header. "
            "The result of the formula has a different number of "
            "dimensions than the image to copy the header from."
        )
    return new_img_like(niimg, result, niimg.affine, copy_header=True)


def binarize_img(
    img, threshold=0, mask_img=None, two_sided=True, copy_header=False
):
    """Binarize an image such that its values are either 0 or 1.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    img : a 3D/4D Niimg-like object
        Image which should be binarized.

    threshold : :obj:`float` or :obj:`str`
        If float, we threshold the image based on image intensities meaning
        voxels which have intensities greater than this value will be kept.
        The given value should be within the range of minimum and
        maximum intensity of the input image.
        If string, it should finish with percent sign e.g. "80%" and we
        threshold based on the score obtained using this percentile on
        the image data. The voxels which have intensities greater than
        this score will be kept. The given string should be
        within the range of "0%" to "100%".

    mask_img : Niimg-like object, default=None
        Mask image applied to mask the input data.
        If None, no masking will be applied.

    two_sided : :obj:`bool`
        If `True`, threshold is applied to the absolute value of the image.
        If `False`, threshold is applied to the original value of the image.

        .. versionadded:: 0.10.3

    copy_header : :obj:`bool`, default=False
        Whether to copy the header of the input image to the output.

        .. versionadded:: 0.11.0

        This parameter will be set to True by default in 0.13.0.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Binarized version of the given input image. Output dtype is int.

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
     >>> img = binarize_img(anatomical_image, copy_header=True)

    """
    warnings.warn(
        'The current default behavior for the "two_sided" argument '
        'is  "True". This behavior will be changed to "False" in '
        "version 0.13.",
        DeprecationWarning,
        stacklevel=3,
    )

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

    .. versionadded:: 0.2.5

    Parameters
    ----------
    imgs : Niimg-like object
        4D image. The signals in the last dimension are filtered (see
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

    standardize : :obj:`bool`, default=True
        If True, returned signals are set to unit variance.

    confounds : :class:`numpy.ndarray`, :obj:`str` or :obj:`list` of
        Confounds timeseries. optional
        Shape must be (instant number, confound number),
        or just (instant number,)
        The number of time instants in signals and confounds must be
        identical (i.e. signals.shape[0] == confounds.shape[0]).
        If a string is provided, it is assumed to be the name of a csv file
        containing signals as columns, with an optional one-line header.
        If a list is provided, all confounds are removed from the input
        signal, as if all were in the same array.

    low_pass : :obj:`float`, optional
        Low cutoff frequencies, in Hertz.

    high_pass : :obj:`float`, optional
        High cutoff frequencies, in Hertz.

    t_r : :obj:`float`, optional
        Repetition time, in second (sampling period). Set to None if not
        specified. Mandatory if used together with `low_pass` or `high_pass`.

    ensure_finite : :obj:`bool`, default=False
        If True, the non-finite values (NaNs and infs) found in the images
        will be replaced by zeros.

    mask_img : Niimg-like object, optional
        If provided, signal is only cleaned from voxels inside the mask. If
        mask is provided, it should have same shape and affine as imgs.
        If not provided, all voxels are used.
        See :ref:`extracting_data`.

    kwargs : dict
        Keyword arguments to be passed to functions called
        within this function.
        Kwargs prefixed with ``'clean__'`` will be passed to
        :func:`~nilearn.signal.clean`.
        Within :func:`~nilearn.signal.clean`, kwargs prefixed with
        ``'butterworth__'`` will be passed to the Butterworth filter
        (i.e., ``clean__butterworth__``).

    Returns
    -------
    Niimg-like object
        Input images, cleaned. Same shape as `imgs`.

    Notes
    -----
    Confounds removal is based on a projection on the orthogonal
    of the signal space [:footcite:t:`Friston1994`].

    Orthogonalization between temporal filters and confound removal is based on
    suggestions in [:footcite:t:`Lindquist2018`].

    References
    ----------
    .. footbibliography::

    See Also
    --------
        nilearn.signal.clean

    """
    # Avoid circular import
    from .. import masking

    imgs_ = check_niimg_4d(imgs)

    # Check if t_r is set, otherwise propose t_r from imgs header
    if (low_pass is not None or high_pass is not None) and t_r is None:
        # We raise an error, instead of using the header's t_r as this
        # value is considered to be non-reliable
        raise ValueError(
            "Repetition time (t_r) must be specified for filtering. "
            "You specified None. "
            f"imgs header suggest it to be {imgs.header.get_zooms()[3]}"
        )

    # Prepare signal for cleaning
    if mask_img is not None:
        signals = masking.apply_mask(imgs_, mask_img)
    else:
        signals = get_data(imgs_).reshape(-1, imgs_.shape[-1]).T

    # Clean signal
    clean_kwargs = {
        k[7:]: v for k, v in kwargs.items() if k.startswith("clean__")
    }
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
        imgs_ = new_img_like(
            imgs_, data.T.reshape(sample_shape), copy_header=True
        )
    else:
        imgs_ = new_img_like(
            imgs_, data.T.reshape(imgs_.shape), copy_header=True
        )

    return imgs_


def load_img(img, wildcards=True, dtype=None):
    """Load a Niimg-like object from filenames or list of filenames.

    .. versionadded:: 0.2.5

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

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns
    -------
    3D/4D Niimg-like object
        Result can be :class:`~nibabel.nifti1.Nifti1Image` or the input, as-is.
        It is guaranteed that
        the returned object has an affine attributes and that
        nilearn.image.get_data returns its data.

    """
    return check_niimg(img, wildcards=wildcards, dtype=dtype)


def concat_imgs(
    niimgs,
    dtype=np.float32,
    ensure_ndim=None,
    memory=None,
    memory_level=0,
    auto_resample=False,
    verbose=0,
):
    """Concatenate a list of 3D/4D niimgs of varying lengths.

    The niimgs list can contain niftis/paths to images of varying dimensions
    (i.e., 3D or 4D) as well as different 3D shapes and affines, as they
    will be matched to the first image in the list if auto_resample=True.

    Parameters
    ----------
    niimgs : iterable of Niimg-like objects or glob pattern
        See :ref:`extracting_data`.
        Niimgs to concatenate.

    dtype : numpy dtype, default=np.float32
        The dtype of the returned image.

    ensure_ndim : integer, optional
        Indicate the dimensionality of the expected niimg. An
        error is raised if the niimg is of another dimensionality.

    auto_resample : boolean, default=False
        Converts all images to the space of the first one.

    verbose : int, default=0
        Controls the amount of verbosity (0 means no messages).

    memory : instance of joblib.Memory or string, default=None
        Used to cache the resampling process.
        By default, no caching is done.
        If a string is given, it is the path to the caching directory.
        If ``None`` is passed will default to ``Memory(location=None)``.

    memory_level : integer, default=0
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    Returns
    -------
    concatenated : nibabel.Nifti1Image
        A single image.

    See Also
    --------
    nilearn.image.index_img

    """
    from ..image import new_img_like  # avoid circular imports

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
        )
    ):
        nii_str = (
            f"image {niimg}" if isinstance(niimg, str) else f"image #{index}"
        )
        logger.log(f"Concatenating {index + 1}: {nii_str}", verbose)

        data[..., cur_4d_index : cur_4d_index + size] = _get_data(niimg)
        cur_4d_index += size

    return new_img_like(
        first_niimg, data, first_niimg.affine, copy_header=True
    )


def largest_connected_component_img(imgs):
    """Return the largest connected component of an image or list of images.

    .. versionadded:: 0.3.1

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
        ret.append(
            new_img_like(img, largest_component, affine, copy_header=True)
        )

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
    if not isinstance(img, spatialimages.SpatialImage):
        raise ValueError("Input value is not an image")
    return new_img_like(
        img,
        safe_get_data(img, copy_data=True),
        img.affine.copy(),
        copy_header=True,
    )
