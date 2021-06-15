"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import warnings
from copy import copy as copy_object

from joblib import Memory

from .base_masker import BaseMasker, filter_and_extract
from .. import _utils
from .. import image
from .. import masking
from .._utils import CacheMixin
from .._utils.class_inspect import get_params
from .._utils.helpers import remove_parameters, rename_parameters
from .._utils.niimg import img_data_dtype
from .._utils.niimg_conversions import _check_same_fov
from nilearn.image import get_data


class _ExtractionFunctor(object):
    func_name = 'nifti_masker_extractor'

    def __init__(self, mask_img_):
        self.mask_img_ = mask_img_

    def __call__(self, imgs):
        return(masking.apply_mask(imgs, self.mask_img_,
                                  dtype=img_data_dtype(imgs)), imgs.affine)


def filter_and_mask(imgs, mask_img_, parameters,
                    memory_level=0, memory=Memory(location=None),
                    verbose=0,
                    confounds=None,
                    sample_mask=None,
                    copy=True,
                    dtype=None):
    """Extract representative time series using given mask.

    Parameters
    ----------
    imgs : 3D/4D Niimg-like object
        Images to be masked. Can be 3-dimensional or 4-dimensional.

    For all other parameters refer to NiftiMasker documentation.

    Returns
    -------
    signals : 2D numpy array
        Signals extracted using the provided mask. It is a scikit-learn
        friendly 2D array with shape n_sample x n_features.

    """
    imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4)

    # Check whether resampling is truly necessary. If so, crop mask
    # as small as possible in order to speed up the process

    if not _check_same_fov(imgs, mask_img_):
        parameters = copy_object(parameters)
        # now we can crop
        mask_img_ = image.crop_img(mask_img_, copy=False)
        parameters['target_shape'] = mask_img_.shape
        parameters['target_affine'] = mask_img_.affine

    data, affine = filter_and_extract(imgs, _ExtractionFunctor(mask_img_),
                                      parameters,
                                      memory_level=memory_level,
                                      memory=memory,
                                      verbose=verbose,
                                      confounds=confounds,
                                      sample_mask=sample_mask,
                                      copy=copy,
                                      dtype=dtype)

    # For _later_: missing value removal or imputing of missing data
    # (i.e. we want to get rid of NaNs, if smoothing must be done
    # earlier)
    # Optionally: 'doctor_nan', remove voxels with NaNs, other option
    # for later: some form of imputation
    return data


class NiftiMasker(BaseMasker, CacheMixin):
    """Applying a mask to extract time-series from Niimg-like objects.

    NiftiMasker is useful when preprocessing (detrending, standardization,
    resampling, etc.) of in-mask voxels is necessary. Use case: working with
    time series of resting-state or task maps.

    Parameters
    ----------
    mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask for the data. If not given, a mask is computed in the fit step.
        Optional parameters (mask_args and mask_strategy) can be set to
        fine tune the mask extraction. If the mask and the images have different
        resolutions, the images are resampled to the mask resolution. If target_shape
        and/or target_affine are provided, the mask is resampled first.
        After this, the images are resampled to the resampled mask.

    runs : numpy array, optional
        Add a run level to the preprocessing. Each run will be
        detrended independently. Must be a 1D array of n_samples elements.
        'runs' replaces 'sessions' after release 0.9.0.
        Using 'session' will result in an error after release 0.9.0.

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
        documentation for details: :func:`nilearn.signal.clean`.
        Default=False.

    low_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details: :func:`nilearn.signal.clean`.

    high_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details: :func:`nilearn.signal.clean`.

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details: :func:`nilearn.signal.clean`.

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy : {'background', 'epi' or 'template'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, 'epi' if they
        are raw EPI images, or you could use 'template' which will
        extract the gray matter part of your data by resampling the MNI152
        brain mask for your data's field of view.
        Depending on this value, the mask will be computed from
        masking.compute_background_mask, masking.compute_epi_mask or
        masking.compute_brain_mask. Default='background'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    sample_mask : Any type compatible with numpy-array indexing, optional
        Masks the niimgs along time/fourth dimension. This complements
        3D masking by the mask_img argument. This masking step is applied
        before data preprocessing at the beginning of NiftiMasker.transform.
        This is useful to perform data subselection as part of a scikit-learn
        pipeline.

            .. deprecated:: 0.8.0
                `sample_mask` is deprecated in 0.8.0 and will be removed in
                0.9.0.

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    memory : instance of joblib.Memory or string, optional
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching. Default=1.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        Default=0.

    reports : boolean, optional
        If set to True, data is saved in order to produce a report.
        Default=True.

    Attributes
    ----------
    `mask_img_` : nibabel.Nifti1Image
        The mask of the data, or the computed one.

    `affine_` : 4x4 numpy array
        Affine of the transformed image.

    See also
    --------
    nilearn.masking.compute_background_mask
    nilearn.masking.compute_epi_mask
    nilearn.image.resample_img
    nilearn.image.high_variance_confounds
    nilearn.masking.apply_mask
    nilearn.signal.clean

    """
    @remove_parameters(['sample_mask'],
                       ('Deprecated in 0.8.0. Supply `sample_masker` through '
                        '`transform` or `fit_transform` methods instead. '),
                       '0.9.0')
    @rename_parameters({'sessions': 'runs'}, '0.9.0')
    def __init__(self, mask_img=None, runs=None, smoothing_fwhm=None,
                 standardize=False, standardize_confounds=True, detrend=False,
                 high_variance_confounds=False, low_pass=None, high_pass=None,
                 t_r=None, target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None, sample_mask=None,
                 dtype=None, memory_level=1, memory=Memory(location=None),
                 verbose=0, reports=True,
                 ):
        # Mask is provided or computed
        self.mask_img = mask_img
        self.runs = runs
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self._sample_mask = sample_mask
        self.dtype = dtype

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.reports = reports
        self._report_content = dict()
        self._report_content['description'] = (
            'This report shows the input Nifti image overlaid '
            'with the outlines of the mask (in green). We '
            'recommend to inspect the report for the overlap '
            'between the mask and its input image. ')
        self._report_content['warning_message'] = None
        self._overlay_text = ('\n To see the input Nifti image before '
                              'resampling, hover over the displayed image.')
        self._shelving = False

    @property
    def sessions(self):
        warnings.warn(DeprecationWarning("`sessions` attribute is deprecated "
                                         "and  will be removed in 0.9.0, use "
                                         "`runs` instead."))
        return self.runs

    @property
    def sample_mask(self):
        warnings.warn(DeprecationWarning(
            "Deprecated. `sample_mask` will be removed  in 0.9.0 in favor of "
            "supplying `sample_mask` through method `transform` or "
            "`fit_transform`."))
        return self._sample_mask

    def generate_report(self):
        from nilearn.reporting.html_report import generate_report
        return generate_report(self)

    def _reporting(self):
        """
        Returns
        -------
        displays : list
            A list of all displays to be rendered.

        """
        try:
            from nilearn import plotting
        except ImportError:
            with warnings.catch_warnings():
                mpl_unavail_msg = ('Matplotlib is not imported! '
                                'No reports will be generated.')
                warnings.filterwarnings('always', message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning,
                            message=mpl_unavail_msg)
                return [None]

        # Handle the edge case where this function is
        # called with a masker having report capabilities disabled
        if self._reporting_data is None:
            return [None]

        img = self._reporting_data['images']
        mask = self._reporting_data['mask']

        if img is not None:
            dim = image.load_img(img).shape
            if len(dim) == 4:
                # compute middle image from 4D series for plotting
                img = image.index_img(img, dim[-1] // 2)
        else:  # images were not provided to fit
            msg = ("No image provided to fit in NiftiMasker. "
                   "Setting image to mask for reporting.")
            warnings.warn(msg)
            self._report_content['warning_message'] = msg
            img = mask

        # create display of retained input mask, image
        # for visual comparison
        init_display = plotting.plot_img(img,
                                         black_bg=False,
                                         cmap='CMRmap_r')
        if mask is not None:
            init_display.add_contours(mask, levels=[.5], colors='g',
                                      linewidths=2.5)

        if 'transform' not in self._reporting_data:
            return [init_display]

        else:  # if resampling was performed
            self._report_content['description'] += self._overlay_text

            # create display of resampled NiftiImage and mask
            # assuming that resampl_img has same dim as img
            resampl_img, resampl_mask = self._reporting_data['transform']
            if resampl_img is not None:
                if len(dim) == 4:
                    # compute middle image from 4D series for plotting
                    resampl_img = image.index_img(resampl_img, dim[-1] // 2)
            else:  # images were not provided to fit
                resampl_img = resampl_mask

            final_display = plotting.plot_img(resampl_img,
                                              black_bg=False,
                                              cmap='CMRmap_r')
            final_display.add_contours(resampl_mask, levels=[.5],
                                       colors='g', linewidths=2.5)

        return [init_display, final_display]

    def _check_fitted(self):
        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def fit(self, imgs=None, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        imgs : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        """
        # y=None is for scikit-learn compatibility (unused here).

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print("[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                _utils._repr_niimgs(imgs, shorten=False)))

        # Compute the mask if not given by the user
        if self.mask_img is None:
            mask_args = (self.mask_args if self.mask_args is not None
                         else {})
            if self.mask_strategy == 'background':
                compute_mask = masking.compute_background_mask
            elif self.mask_strategy == 'epi':
                compute_mask = masking.compute_epi_mask
            elif self.mask_strategy == 'template':
                compute_mask = masking.compute_brain_mask
            else:
                raise ValueError("Unknown value of mask_strategy '%s'. "
                                 "Acceptable values are 'background', "
                                 "'epi' and 'template'." % self.mask_strategy)
            if self.verbose > 0:
                print("[%s.fit] Computing the mask" % self.__class__.__name__)
            self.mask_img_ = self._cache(compute_mask, ignore=['verbose'])(
                imgs, verbose=max(0, self.verbose - 1), **mask_args)
        else:
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)

        if self.reports:  # save inputs for reporting
            self._reporting_data = {'images': imgs, 'mask': self.mask_img_}
        else:
            self._reporting_data = None

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print("[%s.fit] Resampling mask" % self.__class__.__name__)
        self.mask_img_ = self._cache(image.resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=False, interpolation='nearest')
        if self.target_affine is not None:  # resample image to target affine
            self.affine_ = self.target_affine
        else:  # resample image to mask affine
            self.affine_ = self.mask_img_.affine
        # Load data in memory
        get_data(self.mask_img_)
        if self.verbose > 10:
            print("[%s.fit] Finished fit" % self.__class__.__name__)

        if (self.target_shape is not None) or (self.target_affine is not None):
            if self.reports:
                if imgs is not None:
                    resampl_imgs = self._cache(image.resample_img)(
                        imgs, target_affine=self.affine_,
                        copy=False, interpolation='nearest')
                else:  # imgs not provided to fit
                    resampl_imgs = None
                self._reporting_data['transform'] = [resampl_imgs, self.mask_img_]

        return self

    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None,
                              copy=True):
        """Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds : CSV file or array-like or pandas DataFrame, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details: :func:`nilearn.signal.clean`.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

        copy : Boolean, optional
            Indicates whether a copy is returned or not. Default=True.

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each voxel inside the mask.
            shape: (number of scans, number of voxels)

        """

        # Ignore the mask-computing params: they are not useful and will
        # just invalid the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        params = get_params(self.__class__, self,
                            ignore=['mask_img', 'mask_args', 'mask_strategy',
                                    '_sample_mask', 'sample_mask'])

        if hasattr(self, '_sample_mask') and self._sample_mask is not None:
            if sample_mask is not None:
                warnings.warn(
                    UserWarning("Overwriting deprecated attribute "
                                "`NiftiMasker.sample_mask` with parameter "
                                "`sample_mask` in method `transform`.")
                )
            else:
                sample_mask = self._sample_mask

        data = self._cache(filter_and_mask,
                           ignore=['verbose', 'memory', 'memory_level',
                                   'copy'],
                           shelve=self._shelving)(
            imgs, self.mask_img_, params,
            memory_level=self.memory_level,
            memory=self.memory,
            verbose=self.verbose,
            confounds=confounds,
            sample_mask=sample_mask,
            copy=copy,
            dtype=self.dtype
        )

        return data
