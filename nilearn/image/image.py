"""
Preprocessing functions for images.

See also nilearn.signal.
"""
# Authors: Philippe Gervais, Alexandre Abraham
# License: simplified BSD

import collections
from distutils.version import LooseVersion

import numpy as np
from scipy import ndimage
from scipy.stats import scoreatpercentile
import copy
import nibabel
from sklearn.externals.joblib import Parallel, delayed

from .. import signal
from .._utils import (check_niimg_4d, check_niimg_3d, check_niimg, as_ndarray,
                      _repr_niimgs)
from .._utils.niimg_conversions import _index_img, _check_same_fov
from .._utils.niimg import _safe_get_data
from .._utils.compat import _basestring
from .._utils.param_validation import check_threshold


def high_variance_confounds(imgs, n_confounds=5, percentile=2.,
                            detrend=True, mask_img=None):
    """ Return confounds signals extracted from input signals with highest
        variance.

        Parameters
        ==========
        imgs: Niimg-like object
            See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
            4D image.

        mask_img: Niimg-like object
            See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
            If provided, confounds are extracted from voxels inside the mask.
            If not provided, all voxels are used.

        n_confounds: int
            Number of confounds to return

        percentile: float
            Highest-variance signals percentile to keep before computing the
            singular value decomposition, 0. <= `percentile` <= 100.
            mask_img.sum() * percentile / 100. must be greater than n_confounds.

        detrend: bool
            If True, detrend signals before processing.

        Returns
        =======
        v: numpy.ndarray
            highest variance confounds. Shape: (number of scans, n_confounds)

        Notes
        ======
        This method is related to what has been published in the literature
        as 'CompCor' (Behzadi NeuroImage 2007).

        The implemented algorithm does the following:

        - compute sum of squares for each signals (no mean removal)
        - keep a given percentile of signals with highest variance (percentile)
        - compute an svd of the extracted signals
        - return a given number (n_confounds) of signals from the svd with
          highest singular values.

        See also
        ========
        nilearn.signal.high_variance_confounds
    """
    from .. import masking

    if mask_img is not None:
        sigs = masking.apply_mask(imgs, mask_img)
    else:
        # Load the data only if it doesn't need to be masked
        imgs = check_niimg_4d(imgs)
        sigs = as_ndarray(imgs.get_data())
        # Not using apply_mask here saves memory in most cases.
        del imgs  # help reduce memory consumption
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T

    return signal.high_variance_confounds(sigs, n_confounds=n_confounds,
                                           percentile=percentile,
                                           detrend=detrend)


def _fast_smooth_array(arr):
    """Simple smoothing which is less computationally expensive than
    applying a gaussian filter.

    Only the first three dimensions of the array will be smoothed. The
    filter uses [0.2, 1, 0.2] weights in each direction and use a
    normalisation to preserve the local average value.

    Parameters
    ==========
    arr: numpy.ndarray
        4D array, with image number as last dimension. 3D arrays are
        also accepted.

    Returns
    =======
    smoothed_arr: numpy.ndarray
        Smoothed array.

    Note
    ====
    Rather than calling this function directly, users are encouraged
    to call the high-level function :func:`smooth_img` with
    fwhm='fast'.

    """
    neighbor_weight = 0.2
    # 6 neighbors in 3D if not on an edge
    nb_neighbors = 6
    # This scale ensures that a uniform array stays uniform
    # except on the array edges
    scale = 1 + nb_neighbors * neighbor_weight

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
        If fwhm='fast', the affine is not used and can be None

    fwhm: scalar, numpy.ndarray, 'fast' or None
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a scalar is given, width is identical on all three directions.
        A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
        If fwhm == 'fast', a fast smoothing will be performed with
        a filter [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the local average value.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed).


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

    if arr.dtype.kind == 'i':
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            # We don't need crazy precision
            arr = arr.astype(np.float32)
    if copy:
        arr = arr.copy()

    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0

    if fwhm == 'fast':
        arr = _fast_smooth_array(arr)
    elif fwhm is not None:
        # Keep only the scale part.
        affine = affine[:3, :3]

        # Convert from a FWHM to a sigma:
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        sigma = fwhm / (fwhm_over_sigma_ratio * vox_size)
        for n, s in enumerate(sigma):
            ndimage.gaussian_filter1d(arr, s, output=arr, axis=n)

    return arr


def smooth_img(imgs, fwhm):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of arr.
    In all cases, non-finite values in input image are replaced by zeros.

    Parameters
    ==========
    imgs: Niimg-like object or iterable of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        Image(s) to smooth.

    fwhm: scalar, numpy.ndarray, 'fast' or None
        Smoothing strength, as a Full-Width at Half Maximum, in millimeters.
        If a scalar is given, width is identical on all three directions.
        A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
        If fwhm == 'fast', a fast smoothing will be performed with
        a filter [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the scale.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed)

    Returns
    =======
    filtered_img: nibabel.Nifti1Image or list of.
        Input image, filtered. If imgs is an iterable, then filtered_img is a
        list.
    """

    # Use hasattr() instead of isinstance to workaround a Python 2.6/2.7 bug
    # See http://bugs.python.org/issue7624
    if hasattr(imgs, "__iter__") \
       and not isinstance(imgs, _basestring):
        single_img = False
    else:
        single_img = True
        imgs = [imgs]

    ret = []
    for img in imgs:
        img = check_niimg(img)
        affine = img.get_affine()
        filtered = _smooth_array(img.get_data(), affine, fwhm=fwhm,
                                 ensure_finite=True, copy=True)
        ret.append(new_img_like(img, filtered, affine, copy_header=True))

    if single_img:
        return ret[0]
    else:
        return ret


def _crop_img_to(img, slices, copy=True):
    """Crops image to a smaller size

    Crop img to size indicated by slices and adjust affine
    accordingly

    Parameters
    ==========
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        Img to be cropped. If slices has less entries than img
        has dimensions, the slices will be applied to the first len(slices)
        dimensions

    slices: list of slices
        Defines the range of the crop.
        E.g. [slice(20, 200), slice(40, 150), slice(0, 100)]
        defines a 3D cube

    copy: boolean
        Specifies whether cropped data is to be copied or not.
        Default: True

    Returns
    =======
    cropped_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        Cropped version of the input image
    """

    img = check_niimg(img)

    data = img.get_data()
    affine = img.get_affine()

    cropped_data = data[slices]
    if copy:
        cropped_data = cropped_data.copy()

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin

    return new_img_like(img, cropped_data, new_affine)


def crop_img(img, rtol=1e-8, copy=True):
    """Crops img as much as possible

    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.

    Parameters
    ==========
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        img to be cropped.

    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.

    copy: boolean
        Specifies whether cropped data is copied or not.

    Returns
    =======
    cropped_img: image
        Cropped version of the input image
    """

    img = check_niimg(img)
    data = img.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    return _crop_img_to(img, slices, copy=copy)


def _compute_mean(imgs, target_affine=None,
                  target_shape=None, smooth=False):
    from . import resampling
    input_repr = _repr_niimgs(imgs)

    imgs = check_niimg(imgs)
    mean_data = _safe_get_data(imgs)
    affine = imgs.get_affine()
    # Free memory ASAP
    imgs = None
    if not mean_data.ndim in (3, 4):
        raise ValueError('Computation expects 3D or 4D '
                         'images, but %i dimensions were given (%s)'
                         % (mean_data.ndim, input_repr))
    if mean_data.ndim == 4:
        mean_data = mean_data.mean(axis=-1)
    else:
        mean_data = mean_data.copy()
    mean_data = resampling.resample_img(
        nibabel.Nifti1Image(mean_data, affine),
        target_affine=target_affine, target_shape=target_shape,
        copy=False)
    affine = mean_data.get_affine()
    mean_data = mean_data.get_data()

    if smooth:
        nan_mask = np.isnan(mean_data)
        mean_data = _smooth_array(mean_data, affine=np.eye(4), fwhm=smooth,
                                  ensure_finite=True, copy=False)
        mean_data[nan_mask] = np.nan

    return mean_data, affine


def mean_img(imgs, target_affine=None, target_shape=None,
             verbose=0, n_jobs=1):
    """ Compute the mean of the images (in the time dimension of 4th dimension)

    Note that if list of 4D images are given, the mean of each 4D image is
    computed separately, and the resulting mean is computed after.

    Parameters
    ==========

    imgs: Niimg-like object or iterable of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        Images to mean.

    target_affine: numpy.ndarray, optional
        If specified, the image is resampled corresponding to this new affine.
        target_affine can be a 3x3 or a 4x4 matrix

    target_shape: tuple or list, optional
        If specified, the image will be resized to match this new shape.
        len(target_shape) must be equal to 3.
        A target_affine has to be specified jointly with target_shape.

    verbose: int, optional
        Controls the amount of verbosity: higher numbers give
        more messages (0 means no messages).

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    Returns
    =======
    mean: nibabel.Nifti1Image
        mean image

    See Also
    ========
    nilearn.image.math_img : For more general operations on images

    """
    if (isinstance(imgs, _basestring) or
            not isinstance(imgs, collections.Iterable)):
        imgs = [imgs, ]

    imgs_iter = iter(imgs)
    first_img = check_niimg(next(imgs_iter))

    # Compute the first mean to retrieve the reference
    # target_affine and target_shape if_needed
    n_imgs = 1
    running_mean, first_affine = _compute_mean(first_img,
                target_affine=target_affine,
                target_shape=target_shape)

    if target_affine is None or target_shape is None:
        target_affine = first_affine
        target_shape = running_mean.shape[:3]

    for this_mean in Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_compute_mean)(n, target_affine=target_affine,
                                   target_shape=target_shape)
            for n in imgs_iter):
        n_imgs += 1
        # _compute_mean returns (mean_img, affine)
        this_mean = this_mean[0]
        running_mean += this_mean

    running_mean = running_mean / float(n_imgs)
    return new_img_like(first_img, running_mean, target_affine)


def swap_img_hemispheres(img):
    """Performs swapping of hemispheres in the indicated nifti.

       Use case: synchronizing ROIs across hemispheres

    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        Images to swap.

    Returns
    -------
    output: nibabel.Nifti1Image
        hemispherically swapped image

    Notes
    -----
    Supposes a nifti of a brain that is sagitally aligned

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
    out_img = new_img_like(img, img.get_data()[::-1], img.get_affine(),
                           copy_header=True)

    return out_img


def index_img(imgs, index):
    """Indexes into a 4D Niimg-like object in the fourth dimension.

    Common use cases include extracting a 3D image out of `img` or
    creating a 4D image whose data is a subset of `img` data.

    Parameters
    ----------
    imgs: 4D Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.

    index: Any type compatible with numpy array indexing
        Used for indexing the 4D data array in the fourth dimension.

    Returns
    -------
    output: nibabel.Nifti1Image

    See Also
    --------
    nilearn.image.concat_imgs
    nilearn.image.iter_img

    Examples
    --------
    First we concatenate two mni152 images to create a 4D-image::

     >>> from nilearn import datasets
     >>> from nilearn.image import concat_imgs, index_img
     >>> joint_mni_image = concat_imgs([datasets.load_mni152_template(),
     ...                                datasets.load_mni152_template()])
     >>> print(joint_mni_image.shape)
     (91, 109, 91, 2)

    We can now select one slice from the last dimension of this 4D-image::

     >>> single_mni_image = index_img(joint_mni_image, 1)
     >>> print(single_mni_image.shape)
     (91, 109, 91)
    """
    imgs = check_niimg_4d(imgs)
    return _index_img(imgs, index)


def iter_img(imgs):
    """Iterates over a 4D Niimg-like object in the fourth dimension.

    Parameters
    ----------
    imgs: 4D Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.

    Returns
    -------
    output: iterator of 3D nibabel.Nifti1Image

    See Also
    --------
    nilearn.image.index_img

    """
    return check_niimg_4d(imgs, return_iterator=True)


def new_img_like(ref_niimg, data, affine=None, copy_header=False):
    """Create a new image of the same class as the reference image

    Parameters
    ----------
    ref_niimg: image
        Reference image. The new image will be of the same type.

    data: numpy array
        Data to be stored in the image

    affine: 4x4 numpy array, optional
        Transformation matrix

    copy_header: boolean, optional
        Indicated if the header of the reference image should be used to
        create the new image

    Returns
    -------
    new_img: image
        A loaded image with the same type (and header) as the reference image.
    """
    # Hand-written loading code to avoid too much memory consumption
    orig_ref_niimg = ref_niimg
    if (not isinstance(ref_niimg, _basestring)
            and not hasattr(ref_niimg, 'get_data')
            and hasattr(ref_niimg, '__iter__')):
        ref_niimg = ref_niimg[0]
    if not (hasattr(ref_niimg, 'get_data')
              and hasattr(ref_niimg,'get_affine')):
        if isinstance(ref_niimg, _basestring):
            ref_niimg = nibabel.load(ref_niimg)
        else:
            raise TypeError(('The reference image should be a niimg, %r '
                            'was passed') % orig_ref_niimg )

    if affine is None:
        affine = ref_niimg.get_affine()
    if data.dtype == bool:
        default_dtype = np.int8
        if (LooseVersion(nibabel.__version__) >= LooseVersion('1.2.0') and
                isinstance(ref_niimg, nibabel.freesurfer.mghformat.MGHImage)):
            default_dtype = np.uint8
        data = as_ndarray(data, dtype=default_dtype)
    header = None
    if copy_header:
        header = copy.deepcopy(ref_niimg.get_header())
        header['scl_slope'] = 0.
        header['scl_inter'] = 0.
        header['glmax'] = 0.
        header['cal_max'] = np.max(data) if data.size > 0 else 0.
        header['cal_max'] = np.min(data) if data.size > 0 else 0.
    return ref_niimg.__class__(data, affine, header=header)


def threshold_img(img, threshold, mask_img=None):
    """ Threshold the given input image, mostly statistical or atlas images.

    Thresholding can be done based on direct image intensities or selection
    threshold with given percentile.

    .. versionadded:: 0.2

    Parameters
    ----------
    img: a 3D/4D Niimg-like object
        Image contains of statistical or atlas maps which should be thresholded.

    threshold: float or str
        If float, we threshold the image based on image intensities meaning
        voxels which have intensities greater than this value will be kept.
        The given value should be within the range of minimum and
        maximum intensity of the input image.
        If string, it should finish with percent sign e.g. "80%" and we threshold
        based on the score obtained using this percentile on the image data. The
        voxels which have intensities greater than this score will be kept.
        The given string should be within the range of "0%" to "100%".

    mask_img: Niimg-like object, default None, optional
        Mask image applied to mask the input data.
        If None, no masking will be applied.

    Returns
    -------
    threshold_img: Nifti1Image
        thresholded image of the given input image.
    """
    from . import resampling
    from .. import masking

    img = check_niimg(img)
    img_data = img.get_data()
    affine = img.get_affine()

    img_data = np.nan_to_num(img_data)

    if mask_img is not None:
        if not _check_same_fov(img, mask_img):
            mask_img = resampling.resample_img(mask_img, target_affine=affine,
                                               target_shape=img.shape[:3],
                                               interpolation="nearest")

        mask_data, _ = masking._load_mask_img(mask_img)
        # Set as 0 for the values which are outside of the mask
        img_data[mask_data == 0.] = 0.

    if threshold is None:
        raise ValueError("The input parameter 'threshold' is empty. "
                         "Please give either a float value or a string as e.g. '90%'.")
    else:
        cutoff_threshold = check_threshold(threshold, img_data,
                                           percentile_func=scoreatpercentile,
                                           name='threshold')

    img_data[np.abs(img_data) < cutoff_threshold] = 0.
    threshold_img = new_img_like(img, img_data, affine)

    return threshold_img


def math_img(formula, **imgs):
    """Interpret a numpy based string formula using niimg in named parameters.

    .. versionadded:: 0.2.3

    Parameters
    ----------
    formula: str
        The mathematical formula to apply to image internal data. It can use
        numpy imported as 'np'.
    imgs: images (Nifti1Image or file names)
        Keyword arguments corresponding to the variables in the formula as
        Nifti images. All input images should have the same geometry (shape,
        affine).

    Returns
    -------
    return_img: Nifti1Image
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

    Notes
    -----

    This function is the Python equivalent of ImCal in SPM or fslmaths
    in FSL.

    """
    try:
        # Check that input images are valid niimg and have a compatible shape
        # and affine.
        niimgs = []
        for image in imgs.values():
            niimgs.append(check_niimg(image))
        _check_same_fov(*niimgs, raise_error=True)
    except Exception as exc:
        exc.args = (("Input images cannot be compared, you provided '{0}',"
                     .format(imgs.values()),) + exc.args)
        raise

    # Computing input data as a dictionary of numpy arrays. Keep a reference
    # niimg for building the result as a new niimg.
    niimg = None
    data_dict = {}
    for key, img in imgs.items():
        niimg = check_niimg(img)
        data_dict[key] = _safe_get_data(niimg)

    # Add a reference to numpy in the kwargs of eval so that numpy functions
    # can be called from there.
    data_dict['np'] = np
    try:
        result = eval(formula, data_dict)
    except Exception as exc:
        exc.args = (("Input formula couldn't be processed, you provided '{0}',"
                     .format(formula),) + exc.args)
        raise

    return new_img_like(niimg, result, niimg.get_affine())
