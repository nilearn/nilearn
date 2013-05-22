"""
Transformer for computing ROI signals.
"""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

import nibabel

from .. import utils
from ..utils import CacheMixin
from .. import signal
from .. import region
from .. import masking
from .. import resampling
from .. import image


def _compose_err_msg(msg, **kwargs):
    """Append key-value pairs to msg, for display.

    Parameters
    ==========
    msg: string
        arbitrary message
    kwargs: dict
        arbitrary dictionary

    Returns
    =======
    updated_msg: string
        msg, with "key: value" appended. Only string values are appended.
    """
    updated_msg = msg
    for k, v in kwargs.iteritems():
        if isinstance(v, basestring):
            updated_msg += "\n" + k + ": " + v

    return updated_msg


class NiftiLabelsMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Extract labeled-defined region signals from images.

    Parameters
    ==========
    labels_img: niimg
        Region definitions, as one image of labels.

    background_label: number, optional
        Label used in labels_img to represent background.

    mask_img: niimg, optional
        Mask to apply to regions before extracting signals.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    memory: joblib.Memory or str, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: int, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    See also
    ========
    nisl.io.NiftiMasker
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, labels_img, background_label=0, mask_img=None,
                 smoothing_fwhm=None, standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None, verbose=0), memory_level=0,
                 verbose=0):
        self.labels_img = labels_img
        self.background_label = background_label
        self.mask_img = mask_img

        # Parameters for _smooth_array
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.
        """
        labels_img = utils.check_niimg(self.labels_img)
        # Since a copy is required, order can be forced as well.
        self.labels_data_ = utils.as_ndarray(labels_img.get_data(),
                                        copy=True, order="C")
        self.labels_affine_ = utils.as_ndarray(labels_img.get_affine())
        del labels_img

        if self.mask_img is not None:
            mask_data, mask_affine = masking._load_mask_img(self.mask_img)
            if mask_data.shape != self.labels_data_.shape[:3]:
                raise ValueError(
                    _compose_err_msg(
                        "Regions and mask do not have the same shape",
                        mask_img=self.mask_img, labels_img=self.labels_img))
            if not np.allclose(mask_affine, self.labels_affine_):
                raise ValueError(_compose_err_msg(
                    "Regions and mask do not have the same affine.",
                    mask_img=self.mask_img, labels_img=self.labels_img))
            self.labels_data_[
                np.logical_not(mask_data)] = self.background_label

        return self

    def fit_transform(self, niimgs, confounds=None):
        return self.fit().transform(niimgs, confounds=confounds)

    def transform(self, niimgs, confounds=None):
        """Extract signals from images.

        Parameters
        ==========
        niimgs: niimg
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.
        confounds: array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        =======
        signals: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        """
        niimgs = utils.check_niimgs(niimgs)
        data = utils.as_ndarray(niimgs.get_data())
        affine = niimgs.get_affine()
        if self.smoothing_fwhm is not None:
            # FIXME: useless copy if input parameter niimg is a string.
            data = self._cache(masking._smooth_array, memory_level=1)(
                data, affine, fwhm=self.smoothing_fwhm, copy=True)

        region_signals, self.labels_ = self._cache(
            region.img_to_signals_labels, memory_level=1)(
                nibabel.Nifti1Image(data, affine),
                nibabel.Nifti1Image(self.labels_data_, self.labels_affine_),
                background_label=self.background_label)

        region_signals = self._cache(signal.clean, memory_level=1
                                     )(region_signals,
                                       detrend=self.detrend,
                                       standardize=self.standardize,
                                       t_r=self.t_r,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       confounds=confounds)
        return region_signals

    def inverse_transform(self, signals):
        """Compute voxel signals from region signals

        Parameters
        ==========
        signals (2D numpy.ndarray)
            Signal for each region.
            shape: (number of scans, number of regions)

        Returns
        =======
        voxel_signals (Nifti1Image)
            Signal for each voxel
            shape: (number of scans, number of voxels)
        """
        return region.signals_to_img_labels(
            signals,
            nibabel.Nifti1Image(self.labels_data_, self.labels_affine_),
            background_label=self.background_label)


class NiftiMapsMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Extract maps-defined region signals from images.

    Parameters
    ==========
    maps_img: niimg
        Region definitions, as one image of labels.

    mask_img: niimg, optional
        Mask to apply to regions before extracting signals.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    resampling_target: string, optional.
        Gives which image gives the final shape/size. Accepted values
        are "mask", "maps" and None. For example, if `resampling_target`
        is "mask" then maps_img and images provided to fit() are resampled to
        the shape and affine of mask_img. "None" means no resampling. If shapes
        and affines do not match, a ValueError is raised.

    memory: joblib.Memory or str, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: int, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    See also
    ========
    nisl.io.NiftiMasker
    nisl.io.NiftiLabelsMasker
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, maps_img, mask_img=None,
                 smoothing_fwhm=None, standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 resampling_target=None,
                 memory=Memory(cachedir=None, verbose=0), memory_level=0,
                 verbose=0):
        self.maps_img = maps_img
        self.mask_img = mask_img

        # Parameters for image.smooth
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        if resampling_target not in ("mask", "maps", None):
            raise ValueError("invalid value for 'resampling_target' parameter")

        if self.mask_img is None and resampling_target == "mask":
            raise ValueError("resampling_target has been set to 'mask' but no mask "
                             "has been provided.\nSet resampling_target to something else"
                             " or provide a mask.")

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.
        """
        # Load images
        self.maps_img_ = utils.check_niimg(self.maps_img)

        if self.mask_img is not None:
            self.mask_img_ = utils.check_niimg(self.mask_img)
        else:
            self.mask_img_ = None

        # Check shapes and affines or resample.
        if self.resampling_target is None and self.mask_img_ is not None:
            if utils._get_shape(self.mask_img_) \
                    != utils._get_shape(self.maps_img_)[:3]:
                raise ValueError(
                    _compose_err_msg(
                        "Regions and mask do not have the same shape",
                        mask_img=self.mask_img, maps_img=self.maps_img))
            if not np.allclose(self.mask_img_.get_affine(),
                               self.maps_img_.get_affine()):
                raise ValueError(
                    _compose_err_msg(
                        "Regions and mask do not have the same affine.",
                        mask_img=self.mask_img, maps_img=self.maps_img))

            # Since a copy is required, order can be forced as well.
            maps_data = utils.as_ndarray(self.maps_img_.get_data(),
                                          copy=True, order="C")
            maps_affine = utils.as_ndarray(self.maps_img_.get_affine())
            self.maps_img_ = nibabel.Nifti1Image(maps_data, maps_affine)

        elif self.resampling_target == "mask" and self.mask_img_ is not None:
            self.maps_img_ = resampling.resample_img(
                self.maps_img_,
                target_affine=self.mask_img_.get_affine(),
                target_shape=utils._get_shape(self.mask_img_),
                interpolation="nearest",
                copy=True)

        elif self.resampling_target == "maps" and self.mask_img_ is not None:
            self.mask_img_ = resampling.resample_img(
                self.mask_img_,
                target_affine=self.maps_img_.get_affine(),
                target_shape=utils._get_shape(self.maps_img_)[:3],
                interpolation="continuous",
                copy=True)

        return self

    def fit_transform(self, niimgs, confounds=None):
        return self.fit().transform(niimgs, confounds=confounds)

    def transform(self, niimgs, confounds=None):
        """Extract signals from images.

        Parameters
        ==========
        niimgs: niimg
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds: array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        =======
        region_signals: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)
        """
        niimgs = utils.check_niimgs(niimgs)

        # FIXME: add _cache to the calls to resampling.resample.
        if self.resampling_target == "mask":
            niimgs = resampling.resample_img(
                niimgs, interpolation="continuous",
                target_shape=utils._get_shape(self.mask_img_),
                target_affine=self.mask_img_.get_affine())

        if self.resampling_target == "maps":
            niimgs = resampling.resample_img(
                niimgs, interpolation="continuous",
                target_shape=utils._get_shape(self.maps_img_)[:3],
                target_affine=self.maps_img_.get_affine())

        if self.smoothing_fwhm is not None:
            niimgs = self._cache(image.smooth, memory_level=1)(
                niimgs, fwhm=self.smoothing_fwhm)

        region_signals, self.labels_ = self._cache(
            region.img_to_signals_maps, memory_level=1)(
                niimgs,
                self.maps_img_,
                mask_img=self.mask_img_)

        region_signals = self._cache(signal.clean, memory_level=1
                                     )(region_signals,
                                       detrend=self.detrend,
                                       standardize=self.standardize,
                                       t_r=self.t_r,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       confounds=confounds)
        return region_signals

    def inverse_transform(self, region_signals):
        """Compute voxel signals from region signals

        Parameters
        ==========
        region_signals: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        Returns
        =======
        voxel_signals: nibabel.Nifti1Image
            Signal for each voxel. shape: that of maps.
        """
        return region.signals_to_img_maps(region_signals, self.maps_img_,
                                          mask_img=self.mask_img_)
