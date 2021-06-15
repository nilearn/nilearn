"""
Transformer for computing ROI signals.
"""

import numpy as np
from joblib import Memory

from .. import _utils
from .._utils import logger, CacheMixin
from .._utils.class_inspect import get_params
from .._utils.niimg_conversions import _check_same_fov
from .. import image
from .base_masker import filter_and_extract, BaseMasker
from nilearn.image import get_data


class _ExtractionFunctor(object):

    func_name = 'nifti_maps_masker_extractor'

    def __init__(self, _resampled_maps_img_, _resampled_mask_img_):
        self._resampled_maps_img_ = _resampled_maps_img_
        self._resampled_mask_img_ = _resampled_mask_img_

    def __call__(self, imgs):
        from ..regions import signal_extraction

        return signal_extraction.img_to_signals_maps(
            imgs, self._resampled_maps_img_,
            mask_img=self._resampled_mask_img_)


class NiftiMapsMasker(BaseMasker, CacheMixin):
    """Class for masking of Niimg-like objects.

    NiftiMapsMasker is useful when data from overlapping volumes should be
    extracted (contrarily to NiftiLabelsMasker). Use case: Summarize brain
    signals from large-scale networks obtained by prior PCA or ICA.

    Note that, Inf or NaN present in the given input images are automatically
    put to zero rather than considered as missing data.

    Parameters
    ----------
    maps_img : 4D niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Set of continuous maps. One representative time course per map is
        extracted using least square regression.

    mask_img : 3D niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to regions before extracting signals.

    allow_overlap : boolean, optional
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel). Default=True.

    smoothing_fwhm : float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize : {False, True, 'zscore', 'psc'}, optional
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        True : the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        False : Do not standardize the data.
        Default=False.

    standardize_confounds : boolean, optional
        If standardize_confounds is True, the confounds are z-scored:
        their mean is put to 0 and their variance to 1 in the time dimension.
        Default=True.

    high_variance_confounds : boolean, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.

    detrend : boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details. Default=False.

    low_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    resampling_target : {"data", "mask", "maps", None}, optional.
        Gives which image gives the final shape/size. For example, if
        `resampling_target` is "mask" then maps_img and images provided to
        fit() are resampled to the shape and affine of mask_img. "None" means
        no resampling: if shapes and affines do not match, a ValueError is
        raised. Default="data".

    memory : joblib.Memory or str, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : int, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.
        Default=0.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        Default=0.

    Notes
    -----
    If resampling_target is set to "maps", every 3D image processed by
    transform() will be resampled to the shape of maps_img. It may lead to a
    very large memory consumption if the voxel number in maps_img is large.

    See also
    --------
    nilearn.input_data.NiftiMasker
    nilearn.input_data.NiftiLabelsMasker

    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, maps_img, mask_img=None,
                 allow_overlap=True, smoothing_fwhm=None, standardize=False,
                 standardize_confounds=True, high_variance_confounds=False,
                 detrend=False, low_pass=None, high_pass=None, t_r=None,
                 dtype=None, resampling_target="data",
                 memory=Memory(location=None, verbose=0), memory_level=0,
                 verbose=0):
        self.maps_img = maps_img
        self.mask_img = mask_img

        # Maps Masker parameter
        self.allow_overlap = allow_overlap

        # Parameters for image.smooth
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.dtype = dtype

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        if resampling_target not in ("mask", "maps", "data", None):
            raise ValueError("invalid value for 'resampling_target'"
                             " parameter: " + str(resampling_target))

        if self.mask_img is None and resampling_target == "mask":
            raise ValueError(
                "resampling_target has been set to 'mask' but no mask "
                "has been provided.\nSet resampling_target to something else"
                " or provide a mask.")

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.

        """
        # Load images
        logger.log("loading regions from %s" %
                   _utils._repr_niimgs(self.maps_img,
                                       shorten=(not self.verbose)),
                   verbose=self.verbose)
        self.maps_img_ = _utils.check_niimg(self.maps_img, dtype=self.dtype, atleast_4d=True)
        self.maps_img_ = image.clean_img(self.maps_img_, detrend=False,
                                         standardize=False,
                                         ensure_finite=True)
        if self.mask_img is not None:
            logger.log("loading mask from %s" %
                       _utils._repr_niimgs(self.mask_img,
                                           shorten=(not self.verbose)),
                       verbose=self.verbose)
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)
        else:
            self.mask_img_ = None

        # Check shapes and affines or resample.
        if self.resampling_target is None and self.mask_img_ is not None:
            _check_same_fov(mask=self.mask_img_, maps=self.maps_img_,
                            raise_error=True)

        elif self.resampling_target == "mask" and self.mask_img_ is not None:
            if self.verbose > 0:
                print("Resampling maps")
            self.maps_img_ = image.resample_img(
                self.maps_img_,
                target_affine=self.mask_img_.affine,
                target_shape=self.mask_img_.shape,
                interpolation="continuous",
                copy=True)

        elif self.resampling_target == "maps" and self.mask_img_ is not None:
            if self.verbose > 0:
                print("Resampling mask")
            self.mask_img_ = image.resample_img(
                self.mask_img_,
                target_affine=self.maps_img_.affine,
                target_shape=self.maps_img_.shape[:3],
                interpolation="nearest",
                copy=True)

        return self

    def _check_fitted(self):
        if not hasattr(self, "maps_img_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def fit_transform(self, imgs, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction.

        """
        return self.fit().transform(imgs, confounds=confounds,
                                    sample_mask=sample_mask)

    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds : CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each map.
            shape: (number of scans, number of maps)

        """
        # We handle the resampling of maps and mask separately because the
        # affine of the maps and mask images should not impact the extraction
        # of the signal.

        if not hasattr(self, '_resampled_maps_img_'):
            self._resampled_maps_img_ = self.maps_img_
        if not hasattr(self, '_resampled_mask_img_'):
            self._resampled_mask_img_ = self.mask_img_

        if self.resampling_target is None:
            imgs_ = _utils.check_niimg(imgs, atleast_4d=True)
            images = dict(maps=self.maps_img_, data=imgs_)
            if self.mask_img_ is not None:
                images['mask'] = self.mask_img_
            _check_same_fov(raise_error=True, **images)
        else:
            if self.resampling_target == "data":
                imgs_ = _utils.check_niimg(imgs, atleast_4d=True)
                ref_img = imgs_
            elif self.resampling_target == "mask":
                self._resampled_mask_img_ = self.mask_img_
                ref_img = self.mask_img_
            elif self.resampling_target == "maps":
                self._resampled_maps_img_ = self.maps_img_
                ref_img = self.maps_img_

            if not _check_same_fov(ref_img, self._resampled_maps_img_):
                if self.verbose > 0:
                    print("Resampling maps")
                self._resampled_maps_img_ = self._cache(image.resample_img)(
                    self.maps_img_, interpolation="continuous",
                    target_shape=ref_img.shape[:3],
                    target_affine=ref_img.affine)

            if (self.mask_img_ is not None
                    and not _check_same_fov(ref_img, self.mask_img_)):
                if self.verbose > 0:
                    print("Resampling mask")
                self._resampled_mask_img_ = self._cache(image.resample_img)(
                    self.mask_img_, interpolation="nearest",
                    target_shape=ref_img.shape[:3],
                    target_affine=ref_img.affine)

        if not self.allow_overlap:
            # Check if there is an overlap.

            # If float, we set low values to 0
            data = get_data(self._resampled_maps_img_)
            dtype = data.dtype
            if dtype.kind == 'f':
                data[data < np.finfo(dtype).eps] = 0.

            # Check the overlaps
            if np.any(np.sum(data > 0., axis=3) > 1):
                raise ValueError(
                    'Overlap detected in the maps. The overlap may be '
                    'due to the atlas itself or possibly introduced by '
                    'resampling'
                )

        target_shape = None
        target_affine = None
        if self.resampling_target != 'data':
            target_shape = self._resampled_maps_img_.shape[:3]
            target_affine = self._resampled_maps_img_.affine

        params = get_params(NiftiMapsMasker, self,
                            ignore=['resampling_target'])
        params['target_shape'] = target_shape
        params['target_affine'] = target_affine

        region_signals, labels_ = self._cache(
            filter_and_extract, ignore=['verbose', 'memory', 'memory_level'])(
                # Images
                imgs, _ExtractionFunctor(self._resampled_maps_img_,
                                         self._resampled_mask_img_),
                # Pre-treatments
                params,
                confounds=confounds,
                sample_mask=sample_mask,
                dtype=self.dtype,
                # Caching
                memory=self.memory,
                memory_level=self.memory_level,
                # kwargs
                verbose=self.verbose)
        self.labels_ = labels_
        return region_signals

    def inverse_transform(self, region_signals):
        """Compute voxel signals from region signals

        Any mask given at initialization is taken into account.

        Parameters
        ----------
        region_signals : 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        Returns
        -------
        voxel_signals : nibabel.Nifti1Image
            Signal for each voxel. shape: that of maps.

        """
        from ..regions import signal_extraction

        self._check_fitted()

        logger.log("computing image from signals", verbose=self.verbose)
        return signal_extraction.signals_to_img_maps(
            region_signals, self.maps_img_, mask_img=self.mask_img_)
