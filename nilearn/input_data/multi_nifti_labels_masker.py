"""
Transformer for computing ROI signals of multiple 4D images
"""

import collections
import itertools
import warnings

from nilearn._utils.compat import Memory, Parallel, delayed

from .. import _utils
from .. import image
from .. import masking
from .._utils import CacheMixin
from .._utils.class_inspect import get_params
from .._utils.compat import _basestring, izip
from .._utils.niimg_conversions import _iter_check_niimg
from .nifti_labels_masker import NiftiLabelsMasker
from nilearn.image import get_data


class MultiNiftiLabelsMasker(NiftiLabelsMasker, CacheMixin):
    """Class for masking of Niimg-like objects.

    MultiNiftiLabelsMasker is useful when data from non-overlapping volumes and from
    different subjects should be extracted (contrary to NiftiLabelsMasker). Use case:

    Parameters
    ----------
    labels_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Region definitions, as one image of labels.
        background_label: number, optional
        Label used in labels_img to represent background.

    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to regions before extracting signals.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: {'zscore', 'psc', True, False}, default is 'zscore'
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        True : the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        False : Do not standardize the data.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

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

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    strategy: {'sum', 'mean', 'median', 'mininum', 'maximum', 'variance',
        'standard_deviation'}, default is 'mean'
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, mininum, maximum, variance,
        standard_deviation

    See also
    --------
    nilearn.input_data.NiftiMasker
    nilearn.input_data.NiftiLabelsMasker
    """

    def __init__(self, labels_img, background_label=0, mask_img=None,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None, dtype=None,
                 resampling_target="data",
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
                 n_jobs=1, verbose=0, strategy="mean"):
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
        self.dtype = dtype

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

        available_reduction_strategies = {'mean', 'median', 'sum',
                                          'minimum', 'maximum',
                                          'standard_deviation', 'variance'}

        if strategy not in available_reduction_strategies:
            raise ValueError("Invalid strategy '{}'. Valid strategies are {}.".format
                             (strategy, available_reduction_strategies))

        self.strategy = strategy

        if resampling_target not in ("labels", "data", None):
            raise ValueError("invalid value for 'resampling_target' "
                             "parameter: {}".format(resampling_target))

    def transform_imgs(self, imgs_list, confounds=None, n_jobs=1):
        """Extract signals from a list of 4D niimgs.

        Parameters
        ----------
        imgs_list: list of 4D Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Images to process. Each element of the list is a 4D image.

        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: list of (number of scans, number of confounds)

        Returns
        -------
        region_signals: list of 2D numpy.ndarray
            List of signals for each label per subject.
            shape: list of (number of scans, number of labels)
        """
        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that {} has not been fitted. '
                             'You must call fit() before calling transform().'.format
                             (self.__class__.__name__))

        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))

        func = self._cache(self.transform_single_imgs,
                           ignore=['verbose', 'memory', 'memory_level',
                                   'copy'])
        data = Parallel(n_jobs=n_jobs)(delayed(func)
                                       (imgs=imgs, confounds=cfs)
                                       for imgs, cfs in izip(imgs_list, confounds))
        return data

    def transform(self, imgs, confounds=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to signal.clean. Please see the
            corresponding documentation for details.

        Returns
        -------
        data: {list of numpy arrays}
            preprocessed images
        """

        self._check_fitted()
        if (not hasattr(imgs, '__iter__')
                or isinstance(imgs, _basestring)):
            return self.transform_single_imgs(imgs)
        return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs)
