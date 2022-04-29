"""
Transformer for computing ROI signals of multiple 4D images
"""

import itertools

from joblib import Memory, Parallel, delayed

from .._utils import CacheMixin
from .._utils.niimg_conversions import _iter_check_niimg
from .nifti_labels_masker import NiftiLabelsMasker


class MultiNiftiLabelsMasker(NiftiLabelsMasker, CacheMixin):
    """Class for masking of Niimg-like objects.
    MultiNiftiLabelsMasker is useful when data from non-overlapping volumes
    and from different subjects should be extracted (contrary to
    NiftiLabelsMasker).
    Parameters
    ----------
    labels_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Region definitions, as one image of labels.

    labels : list of str, optional
        Full labels corresponding to the labels image. This is used
        to improve reporting quality if provided.
        Warning: The labels must be consistent with the label
        values provided through `labels_img`.

    background_label: number, optional
        Label used in labels_img to represent background.

    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to regions before extracting signals.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width-at-half-maximum
        in millimeters of the spatial smoothing to apply to the signal.

    standardize: {'zscore', 'psc', True, False}, default is 'zscore'
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        True : the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        False : Do not standardize the data.

    standardize_confounds: boolean, optional,  default is True
        If standardize_confounds is True, the confounds are z-scored:
        their mean is put to 0 and their variance to 1 in the time dimension

    high_variance_confounds : boolean, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.

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

    strategy: {'sum', 'mean', 'median', 'minimum', 'maximum', 'variance',
        'standard_deviation'}, default is 'mean'
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, minimum, maximum, variance,
        standard_deviation

    reports : boolean, optional
         If set to True, data is saved in order to produce a report.
         Default=True.

    See also
    --------
    nilearn.input_data.NiftiMasker
    nilearn.input_data.NiftiLabelsMasker

    """

    def __init__(self, labels_img, labels=None, background_label=0,
                 mask_img=None, smoothing_fwhm=None, standardize=False,
                 standardize_confounds=True, high_variance_confounds=False,
                 detrend=False, low_pass=None, high_pass=None, t_r=None,
                 dtype=None, resampling_target="data",
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
                 n_jobs=1, verbose=0, strategy="mean", reports=True):
        self.labels_img = labels_img
        self.labels = labels
        self.background_label = background_label
        self.mask_img = mask_img

        # Parameters for _smooth_array
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
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.reports = reports
        self._report_content = dict()
        self._report_content['description'] = (
            'This reports shows the regions '
            'defined by the labels of the mask.')
        self._report_content['warning_message'] = None

        available_reduction_strategies = {'mean', 'median', 'sum',
                                          'minimum', 'maximum',
                                          'standard_deviation', 'variance'}

        if strategy not in available_reduction_strategies:
            raise ValueError(
                "Invalid strategy '{}'. Valid strategies are {}.".format
                (strategy, available_reduction_strategies))

        self.strategy = strategy

        if resampling_target not in ("labels", "data", None):
            raise ValueError("invalid value for 'resampling_target' "
                             "parameter: {}".format(resampling_target))

    def transform_imgs(self, imgs_list, confounds=None, n_jobs=1,
                       sample_mask=None):
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

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals: list of 2D numpy.ndarray
            List of signals for each label per subject.
            shape: list of (number of scans, number of labels)

        """
        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

        if not hasattr(self, 'mask_img_'):
            raise ValueError(
                'It seems that {} has not been fitted. '
                'You must call fit() before calling transform().'.format
                (self.__class__.__name__))

        niimg_iter = _iter_check_niimg(imgs_list, ensure_ndim=None,
                                       atleast_4d=False,
                                       memory=self.memory,
                                       memory_level=self.memory_level,
                                       verbose=self.verbose)

        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))

        func = self._cache(self.transform_single_imgs)

        region_signals = Parallel(n_jobs=n_jobs)(
            delayed(func)(imgs=imgs, confounds=cfs, sample_mask=sample_mask)
            for imgs, cfs in zip(niimg_iter, confounds))
        return region_signals

    def transform(self, imgs, confounds=None, sample_mask=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to signal.clean. Please see the
            corresponding documentation for details.

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        data: {list of numpy arrays}
            preprocessed images

        """

        self._check_fitted()
        if (not hasattr(imgs, '__iter__')
                or isinstance(imgs, str)):
            return self.transform_single_imgs(imgs)
        return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs,
                                   sample_mask=sample_mask)
