"""
Preprocessing functions for images.

See also nilearn.signal.
"""
# Authors: Philippe Gervais, Alexandre Abraham
# License: simplified BSD

import numpy as np
import nibabel

from .. import signal
from .._utils import check_niimgs, check_niimg, as_ndarray
from .. import masking


def high_variance_confounds(niimgs, n_confounds=10, percentile=1.,
                            detrend=True, mask_img=None):
    """ Return confounds signals extracted from input signals with highest
        variance.

        Parameters
        ==========
        niimgs: niimg
            4D image.

        mask_img: niimg
            If provided, confounds are extracted from voxels inside the mask.
            If not provided, all voxels are used.

        n_confounds: int
            Number of confounds to return

        percentile: float
            Highest-variance signals percentile to keep before computing the
            singular value decomposition.
            mask_img.sum() * percentile must be greater than n_confounds.

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

    niimgs = check_niimgs(niimgs)
    if mask_img is not None:
        mask_img = check_niimg(mask_img)
        sigs = masking.apply_mask(niimgs, mask_img)
    else:
        sigs = as_ndarray(niimgs.get_data())
        # Not using apply_mask here saves memory in most cases.
        del niimgs  # help reduce memory consumption
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T

    return signal.high_variance_confounds(sigs, n_confounds=n_confounds,
                                           percentile=percentile,
                                           detrend=detrend)


def smooth(niimgs, fwhm):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of arr.
    In all cases, non-finite values in input image are replaced by zeros.

    Parameters
    ==========
    niimgs: niimgs or iterable of niimgs
        One or several niimage(s), either 3D or 4D.

    fwhm: scalar or numpy.ndarray
        Smoothing strength, as a Full-Width at Half Maximum, in millimeters.
        If a scalar is given, width is identical on all three directions.
        A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed)

    Returns
    =======
    filtered_img: nibabel.Nifti1Image or list of.
        Input image, filtered. If niimgs is an iterable, then filtered_img is a
        list.
    """

    # Use hasattr() instead of isinstance to workaround a Python 2.6/2.7 bug
    # See http://bugs.python.org/issue7624
    if hasattr(niimgs, "__iter__") \
       and not isinstance(niimgs, basestring):
        single_img = False
    else:
        single_img = True
        niimgs = [niimgs]

    ret = []
    for img in niimgs:
        img = check_niimg(img)
        affine = img.get_affine()
        filtered = masking._smooth_array(img.get_data(), affine,
                                         fwhm=fwhm, ensure_finite=True,
                                         copy=True)
        ret.append(nibabel.Nifti1Image(filtered, affine))

    if single_img:
        return ret[0]
    else:
        return ret


def _crop_img_to(niimg, slices, copy=True):
    """Crops niimg to size indicated by slices"""

    niimg = check_niimg(niimg)

    data = niimg.get_data()
    affine = niimg.get_affine()

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

    new_niimg = nibabel.Nifti1Image(cropped_data, new_affine)

    return new_niimg


def crop_img(niimg, copy=True):
    """Crops niimg as much as possible in all three axes,
    making sure to only remove zero-valued voxels"""

    niimg = check_niimg(niimg)
    data = niimg.get_data()

    coords = np.array(np.where(data != 0))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape)

    slices = [slice(s, e) for s, e in zip(start, end)]

    return _crop_img_to(niimg, slices, copy=copy)

