"""
Transformer for computing ROI signals.
"""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

from .. import _utils
from .._utils import logger
from .._utils import CacheMixin
from .._utils import _compose_err_msg
from .._utils.niimg_conversions import _check_same_fov
from .. import signal
from .. import region
from .. import masking
from .. import image


class NiftiLabelsMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Class for masking of Niimg-like objects.

    NiftiLabelsMasker is useful when data from non-overlapping volumes should
    be extracted (contrarily to NiftiMapsMasker). Use case: Summarize brain
    signals from clusters that were obtained by prior K-means or Ward
    clustering.

    Parameters
    ==========
    labels_img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Region definitions, as one image of labels.

    background_label: number, optional
        Label used in labels_img to represent background.

    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Mask to apply to regions before extracting signals.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

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

    resampling_target: {"data", "labels", None}, optional.
        Gives which image gives the final shape/size. For example, if
        `resampling_target` is "data", the atlas is resampled to the
        shape of the data if needed. If it is "labels" then mask_img
        and images provided to fit() are resampled to the shape and
        affine of maps_img. "None" means no resampling: if shapes and
        affines do not match, a ValueError is raised. Defaults to "data".

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
    nilearn.input_data.NiftiMasker
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, labels_img, background_label=0, mask_img=None,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 resampling_target="data",
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
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

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        if resampling_target not in ("labels", "data", None):
            raise ValueError("invalid value for 'resampling_target' "
                             "parameter: " + str(resampling_target))

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.
        """
        logger.log("loading data from %s" %
                   _utils._repr_niimgs(self.labels_img)[:200],
                   verbose=self.verbose)
        self.labels_img_ = _utils.check_niimg_3d(self.labels_img)
        if self.mask_img is not None:
            logger.log("loading data from %s" %
                       _utils._repr_niimgs(self.mask_img)[:200],
                       verbose=self.verbose)
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)
        else:
            self.mask_img_ = None

        # Check shapes and affines or resample.
        if self.mask_img_ is not None:
            if self.resampling_target == "data":
                # resampling will be done at transform time
                pass
            elif self.resampling_target is None:
                if self.mask_img_.shape != self.labels_img_.shape[:3]:
                    raise ValueError(
                        _compose_err_msg(
                            "Regions and mask do not have the same shape",
                            mask_img=self.mask_img,
                            labels_img=self.labels_img))
                if not np.allclose(self.mask_img_.get_affine(),
                                   self.labels_img_.get_affine()):
                    raise ValueError(_compose_err_msg(
                        "Regions and mask do not have the same affine.",
                        mask_img=self.mask_img, labels_img=self.labels_img))

            elif self.resampling_target == "labels":
                logger.log("resampling the mask", verbose=self.verbose)
                self.mask_img_ = image.resample_img(
                    self.mask_img_,
                    target_affine=self.labels_img_.get_affine(),
                    target_shape=self.labels_img_.shape[:3],
                    interpolation="nearest",
                    copy=True)
            else:
                raise ValueError("Invalid value for resampling_target: "
                                 + str(self.resampling_target))

            mask_data, mask_affine = masking._load_mask_img(self.mask_img_)

        return self

    def fit_transform(self, imgs, confounds=None):
        return self.fit().transform(imgs, confounds=confounds)

    def _check_fitted(self):
        if not hasattr(self, "labels_img_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform(self, imgs, confounds=None):
        """Extract signals from images.

        Parameters
        ==========
        imgs: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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
        self._check_fitted()

        logger.log("loading images: %s" %
                   _utils._repr_niimgs(imgs)[:200], verbose=self.verbose)
        imgs = _utils.check_niimg_4d(imgs)

        if self.resampling_target == "data":
            if not hasattr(self, '_resampled_labels_img_'):
                self._resampled_labels_img_ = self.labels_img_
            if not _check_same_fov(imgs, self._resampled_labels_img_):
                logger.log("resampling labels", verbose=self.verbose)
                self._resampled_labels_img_ = self._cache(image.resample_img,
                    func_memory_level=1)(
                        self.labels_img_, interpolation="nearest",
                        target_shape=imgs.shape[:3],
                        target_affine=imgs.get_affine(),
                    )
        elif self.resampling_target == "labels":
            self._resampled_labels_img_ = self.labels_img_
            logger.log("resampling images", verbose=self.verbose)
            imgs = self._cache(image.resample_img, func_memory_level=1)(
                imgs, interpolation="continuous",
                target_shape=self.labels_img_.shape,
                target_affine=self.labels_img_.get_affine())
        else:
            self._resampled_labels_img_ = self.labels_img_

        if self.smoothing_fwhm is not None:
            logger.log("smoothing images", verbose=self.verbose)
            imgs = self._cache(image.smooth_img, func_memory_level=1)(
                imgs, fwhm=self.smoothing_fwhm)

        logger.log("extracting region signals", verbose=self.verbose)
        region_signals, self.labels_ = self._cache(
            region.img_to_signals_labels, func_memory_level=1)(
                imgs, self._resampled_labels_img_,
                background_label=self.background_label)

        logger.log("cleaning extracted signals", verbose=self.verbose)
        region_signals = self._cache(signal.clean, func_memory_level=1
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

        Any mask given at initialization is taken into account.

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
        self._check_fitted()

        logger.log("computing image from signals", verbose=self.verbose)
        return region.signals_to_img_labels(
            signals, self.labels_img_, self.mask_img_,
            background_label=self.background_label)
