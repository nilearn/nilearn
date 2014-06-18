"""
Preprocessing functions for images.

See also nilearn.signal.
"""
# Authors: Philippe Gervais, Alexandre Abraham
# License: simplified BSD

import collections

import numpy as np
from scipy import ndimage
import nibabel
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from .. import signal
from .._utils import check_niimgs, check_niimg, as_ndarray, _repr_niimgs
from .._utils.niimg_conversions import (_safe_get_data, check_niimgs,
                                        _index_niimgs)
from .. import masking
from nilearn.image import reorder_img
from .resampling import get_bounds, resample_img


def high_variance_confounds(imgs, n_confounds=5, percentile=2.,
                            detrend=True, mask_img=None):
    """ Return confounds signals extracted from input signals with highest
        variance.

        Parameters
        ==========
        imgs: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            4D image.

        mask_img: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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

    if mask_img is not None:
        sigs = masking.apply_mask(imgs, mask_img)
    else:
        # Load the data only if it doesn't need to be masked
        imgs = check_niimgs(imgs)
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
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        One or several niimage(s), either 3D or 4D.

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
       and not isinstance(imgs, basestring):
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
        ret.append(nibabel.Nifti1Image(filtered, affine))

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
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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

    new_img = nibabel.Nifti1Image(cropped_data, new_affine)

    return new_img


def crop_img(niimg, rtol=1e-8, padding=1, copy=True):
    """Crops niimg as much as possible

    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.

    Parameters
    ==========
    img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        img to be cropped.

    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.

    padding: int or tuple, default 1
        indicates the number of voxels to use for padding. Padding
        will never go further than image border.

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

    # pad with default one voxel to avoid resampling problems
    padding = np.atleast_1d(padding)
    start = np.maximum(start - padding, 0)
    end = np.minimum(end + padding, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    return _crop_img_to(img, slices, copy=copy)


class NiftiCropper(BaseEstimator, TransformerMixin):
    """Crops an image along its sampling axes.

    Parameters
    ==========

    bounding box: ((xmin, xmax), (ymin, ymax), (zmin, zmax)), default None
        bounding box specifying the crop.
        If `None` is passed, then the bounding box will be specified such
        that all non-zero (in the sense of the tolerance) voxels will be
        kept within the new bounding box.
    affine: ndarray, shape=4, 4
        affine transformation specifying the orientation of the bounding
        box and the origin of the coordinates.
    rtol: float, default 1e-8
        If crop is computed, then all voxel values below this relative
        tolerance value will be considered zero.
    """

    def __init__(
        self,
        bounding_box=None,
        affine=None,
        resample=False,
        cropping_threshold=None,
        copy=True):

        self.bounding_box = bounding_box
        self.affine = affine
        self.resample = resample
        self.cropping_threshold = cropping_threshold
        self.copy = copy

    def fit(self, niimg=None):

        if niimg is None:
            # No image passed. Check if enough information is provided
            # for cropping. Affine can be None, but bounding_box must
            # be provided

            self.affine_ = self.affine
            if self.bounding_box is None:
                raise ValueError("If no image is provided at fit, then "
                                 "a bounding box is required.")
            self.bounding_box_ = self.bounding_box

        else:
            # Niimg provided. We complete all *missing* information using
            # its specifications.
            niimg = check_niimg(niimg)
            niimg_affine = niimg.get_affine()

            if self.affine is None:
                # Use the affine provided by the image if none is specified
                self.affine_ = niimg_affine
            else:
                # Otherwise use the specified one
                self.affine_ = self.affine

            if self.bounding_box is not None:
                # Use the provided bounding box
                self.bounding_box_ = self.bounding_box
            else:
                # In this case we need to infer the bounding box from the
                # image itself.
                if self.cropping_threshold is None:
                    # No cropping threshold means that we can use the
                    # bounding box of the image, transform it to the
                    # potentially different new affine space, and deduce
                    # the new bounding box

                    if (self.affine_ == niimg_affine).all():
                        transform_affine = np.eye(4)
                    else:
                        transform_affine = \
                            np.linalg.inv(self.affine_).dot(niimg_affine)

                    # Use get_bounds from resample_img to find the new
                    # bounding box
                    self.bounding_box_ = get_bounds(niimg.shape,
                                                    transform_affine)
                else:
                    # A cropping threshold is given, so we find the smallest
                    # bounding box around the non-zero part of the data wrt
                    # the target affine.
                    # We crop along the coordinate axes, without copying, to
                    # reduce data size. Then, if the affine is oblique, we
                    # make it diagonal and repeat the coordinate cropping.

                    coord_crop = crop_img(niimg, self.cropping_threshold,
                                          copy=False)

                    # If the target affine coincides with that of the image,
                    # then we are already done.
                    if (self.affine_ == niimg_affine).all():
                        # All we need to do now is infer the bounding box
                        # from the new affine offset of coord_crop
                        affine_offset_mm = (coord_crop.get_affine()[:3, 3] -
                                            self.affine_[:3, 3])
                        affine_offset = np.linalg.inv(
                            self.affine_[:3, :3]).dot(affine_offset_mm)
                        bbox_min = affine_offset
                        bbox_max = affine_offset + np.array(coord_crop.shape)
                        self.bounding_box_ = zip(bbox_min, bbox_max)
                    
                    # If the target affine does not coincide with that of the
                    # image, then we probably have to resample to obtain the
                    # bounding box. At a later stage, we should check for
                    # coaxial dilations/contractions and axis permutations
                    else:
                        # Resample the cropped image, apply crop_img again
                        # to obtain the minimal bounding box, and then infer
                        # the minimal bounding box with respect to the target
                        # affine.

                        # Calling resample_img with target_affine and without
                        # target_shape infers the new shape such that it
                        # contains all the data, and the affine origin, if
                        # specified.
                        resampled = resample_img(
                            coord_crop,
                            target_affine=self.affine_,
                            target_shape=None,
                            copy=True)
                        cropped_resampled = crop_img(
                            resampled, self.cropping_threshold, copy=False)

                        # If the affine that was given initially was 3x3,
                        # then use the inferred 4D affine for the cropper.
                        # This will result in all minimal bounding box
                        # values being zero and the maximal ones
                        # corresponding to the cropped shape

                        cropped_resampled_affine = \
                            cropped_resampled.get_affine()
                        if self.affine_.shape == (3, 3):
                            # Bounding box corresponds to cropped shape
                            self.affine_ = cropped_resampled_affine
                            bbox_min = np.array([0, 0, 0])
                            bbox_max = np.array(cropped_resampled.shape)
                            self.bounding_box_ = zip(bbox_min, bbox_max)
                        else:
                            # Here we need to calculate the beginning
                            # of the bounding box from the cropped image
                            affine_offset_mm = (
                                cropped_resampled_affine[:3, :] -
                                self.affine_[:3, :])
                            affine_offset = np.linalg.inv(
                                self.affine_[:3, :3]).dot(affine_offset_mm)
                            bbox_min = affine_offset
                            bbox_max = (affine_offset +
                                        np.array(cropped_resampled.shape))
                            self.bounding_box_ = zip(bbox_min, bbox_max)

        return self

    def transform(self, niimg):
        "Extracts the content within the specified bounding box from niimg"

        if not hasattr(self, "bounding_box_"):
            raise ValueError("Object NiftiCropper needs to be fit first")

        niimg = check_niimg(niimg)

        if self.affine_ is None:
            # in this case just crop using the bounding box information
            # along the axes of the niimg
            slices = [slice(start, end)
                      for start, end in self.bounding_box_]
            return _crop_img_to(niimg, slices, self.copy)
        else:
            # Here we use our self.affine_ and our bounding box to locate
            # the region for cropping.
            # If niimg and this object share the same affine, and the
            # bounding box has integer values, then we do not need to
            # resample. There are other situations where resampling could
            # be avoided: If the niimg affine axes are permuted with respect
            # to the self.affine_ axes, if the axis directions are shared,
            # but not necessarily the voxel sizes, and lastly, if the affine
            # offset difference is integer (or not) and if the bounding box
            # shape is not integer. All these options can be added later.
            niimg_affine = niimg.get_affine()
            if (niimg_affine == self.affine_).all():
                # Bounding box is assumed to be integer
                slices = [slice(start, end)
                          for start, end in self.bounding_box_]
                return _crop_img_to(niimg, slices, self.copy)

            # Before the last resort, resampling, the other options can be
            # played through. Here comes the last resort.

            # If bbox is not int, then we introduce a left bias here
            bbox_int = np.array(self.bounding_box_).astype(int)
            bbox_offset = bbox_int[:, 0]
            bbox_shape = bbox_int[:, 1] - bbox_offset
            new_affine = self.affine_.copy()
            new_affine[:3, :] += new_affine[:3, :3].dot(bbox_offset)
            return resample_img(niimg,
                                target_affine=new_affine,
                                target_shape=bbox_shape)

                                            

def _compute_mean(imgs, target_affine=None,
                  target_shape=None, smooth=False):
    from . import resampling
    input_repr = _repr_niimgs(imgs)

    imgs = check_niimgs(imgs, accept_3d=True)
    mean_img = _safe_get_data(imgs)
    if not mean_img.ndim in (3, 4):
        raise ValueError('Computation expects 3D or 4D '
                         'images, but %i dimensions were given (%s)'
                         % (mean_img.ndim, input_repr))
    if mean_img.ndim == 4:
        mean_img = mean_img.mean(axis=-1)
    mean_img = resampling.resample_img(
        nibabel.Nifti1Image(mean_img, imgs.get_affine()),
        target_affine=target_affine, target_shape=target_shape)
    affine = mean_img.get_affine()
    mean_img = mean_img.get_data()

    if smooth:
        nan_mask = np.isnan(mean_img)
        mean_img = _smooth_array(mean_img, affine=np.eye(4), fwhm=smooth,
                                 ensure_finite=True, copy=False)
        mean_img[nan_mask] = np.nan

    return mean_img, affine


def mean_img(imgs, target_affine=None, target_shape=None,
             verbose=0, n_jobs=1):
    """ Compute the mean of the images (in the time dimension of 4th dimension)

    Note that if list of 4D images are given, the mean of each 4D image is
    computed separately, and the resulting mean is computed after.

    Parameters
    ==========

    imgs: Niimg-like object or iterable of Niimg-like objects
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        One or several niimage(s), either 3D or 4D (note that these
        can be file names).

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

    """
    if (isinstance(imgs, basestring) or
        not isinstance(imgs, collections.Iterable)):
        imgs = [imgs, ]
        total_n_imgs = 1
    else:
        try:
            total_n_imgs = len(imgs)
        except:
            total_n_imgs = None

    imgs_iter = iter(imgs)

    if target_affine is None or target_shape is None:
        # Compute the first mean to retrieve the reference
        # target_affine and target_shape
        n_imgs = 1
        running_mean, target_affine = _compute_mean(next(imgs_iter),
                    target_affine=target_affine,
                    target_shape=target_shape)
        target_shape = running_mean.shape[:3]
    else:
        running_mean = None
        n_imgs = 0

    if not (total_n_imgs == 1 and n_imgs == 1):
        for this_mean in Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_compute_mean)(n, target_affine=target_affine,
                                       target_shape=target_shape)
                for n in imgs_iter):
            n_imgs += 1
            # _compute_mean returns (mean_img, affine)
            this_mean = this_mean[0]
            if running_mean is None:
                running_mean = this_mean
            else:
                running_mean += this_mean

    running_mean = running_mean / float(n_imgs)
    return nibabel.Nifti1Image(running_mean, target_affine)


def swap_img_hemispheres(img):
    """Performs swapping of hemispheres in the indicated nifti.

       Use case: synchronizing ROIs across hemispheres

    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        One or several niimgs.

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

    # Check input is really a path to a nifti file or a nifti object
    img = check_niimg(img)

    # get nifti in x-y-z order
    img = reorder_img(img)

    # create swapped nifti object
    out_img = nibabel.Nifti1Image(img.get_data()[::-1], img.get_affine(),
                                  header=img.get_header())

    return out_img


def index_img(imgs, index):
    """Indexes into a 4D Niimg-like object in the fourth dimension.

    Common use cases include extracting a 3D image out of `img` or
    creating a 4D image whose data is a subset of `img` data.

    Parameters
    ----------
    imgs: 4D Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.

    index: Any type compatible with numpy array indexing
        Used for indexing the 4D data array in the fourth dimension.

    Returns
    -------
    output: nibabel.Nifti1Image

    """
    imgs = check_niimgs(imgs)
    return _index_niimgs(imgs, index)


def iter_img(imgs):
    """Iterates over a 4D Niimg-like object in the fourth dimension.

    Parameters
    ----------
    imgs: 4D Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.

    Returns
    -------
    output: iterator of 3D nibabel.Nifti1Image
    """
    return check_niimgs(imgs, return_iterator=True)
