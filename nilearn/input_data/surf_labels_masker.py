"""
Transformer for computing ROI signals from surface data.
"""

import numpy as np

from joblib import Memory

import collections.abc
from ..surface import nisurf
from scipy import ndimage
from .._utils import logger, CacheMixin, _compose_err_msg
from .._utils.class_inspect import get_params
from .base_surf_masker import BaseSurfMasker, filter_and_extract


def _all_files_check(surfs):
    '''In the case where surfs are a list of not yet loaded files,
    we should load and proc files 1 by 1

    Parameters
    ----------
    surfs : 2D Nisurfs-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        input images.


    Returns
    -------
    all_files : bool
        If surfs is a list of 1D Nisurf-data-like file_paths returns
        True, otherwise false.
    '''

    all_files =\
        (isinstance(surfs, collections.abc.Iterable)) and (
            all([isinstance(surf, str) for surf in surfs]))

    return all_files


def _get_single_surf(surfs, n, all_files):
    '''Helper function to return the right single
    surface, based on an all_files passed case where
    files have yet to be loaded, or a case where the input is
    already loaded

    Parameters
    ----------
    surfs : 2D Nisurfs-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        input images.

    n : int
        The index of the surf in which to return

    all_files : bool
        Indicator as to if the passed surfs are a list-like
        of file paths

    Returns
    -------
    surf : 1D Nisurf-data like
        The n'th surf in passed surfs
    '''

    if all_files:
        return nisurf.check_nisurf_1d(surfs[n])
    else:
        return surfs[:, n]


def surf_to_signals_labels(surfs, labels_surf, background_label=0,
                           order='F', strategy='mean'):
    """Extract region signals from surfs.

    This function is applicable to regions defined by labels.

    Parameters
    ----------
    surfs: 2D Nisurfs-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        input images. If a list of 1D Nisurf-data-like file_paths, will
        load and proc 1 by 1.

    labels_surf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        regions definition as labels. By default, the label zero is used to
        denote an absence of region. Use background_label to change it.

    background_label: number
        number representing background in labels_surf.

    order: str
        ordering of output array ("C" or "F"). Defaults to "F".

    strategy: str
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, mininum, maximum, variance,
        standard_deviation

    Returns
    -------
    signals: numpy.ndarray
        Signals extracted from each region.
        Shape is: (number of scans, number of labels/regions)

    labels: list or tuple
        corresponding labels for each signal. signal[:, n] was extracted from
        the region with label labels[n].

    See also
    --------
    nilearn.regions.img_to_signals_labels
    """

    available_reduction_strategies = {'mean', 'median', 'sum',
                                      'minimum', 'maximum',
                                      'standard_deviation', 'variance'}
    if strategy not in available_reduction_strategies:
        raise ValueError(str.format(
            "Invalid strategy '{}'. Valid strategies are {}.",
            strategy,
            available_reduction_strategies
        ))
    reduction_function = getattr(ndimage.measurements, strategy)

    # Load labels
    labels_surf = nisurf.check_nisurf_1d(labels_surf)
    labels = list(np.unique(labels_surf))
    if background_label in labels:
        labels.remove(background_label)

    # Check for a list of files case
    all_files = _all_files_check(surfs)
    if not all_files:
        surfs = nisurf.check_nisurf_2d(surfs, atleast_2d=True)
        surf_dtype, surf_sz = surfs.dtype, surfs.shape[-1]

    # If a list of files, grab dtype + size from first one
    else:
        first_surf = nisurf.check_nisurf_1d(surfs[0])
        surf_dtype, surf_sz = first_surf.dtype, first_surf.shape[-1]

    target_datatype = np.float32 if surf_dtype == np.float32 else np.float64
    signals = np.ndarray((surf_sz, len(labels)), dtype=target_datatype,
                         order=order)

    # Process by data point / # of scans
    for n in range(surf_sz):

        # Grab the next surface, depending on if is still a file
        # path or not
        surf = _get_single_surf(surfs, n, all_files)
        if surf.shape != labels_surf.shape:
            raise ValueError("labels_surf and each passed surf must have the "
                             "the same shape.")

        # Pass version of surf-data w/o NaN's or infs
        signals[n] =\
            np.asarray(reduction_function(
                nisurf._safe_get_data(surf, ensure_finite=True),
                labels=labels_surf,
                index=labels))

    # Set to zero signals for missing labels. Workaround for Scipy behaviour
    missing_labels = set(labels) - set(np.unique(labels_surf))
    labels_index = dict([(l, n) for n, l in enumerate(labels)])
    for l in missing_labels:
        signals[:, labels_index[l]] = 0
    return signals, labels


def signals_to_surf_labels(signals, labels_surf,
                           background_label=0, order="C"):
    """Create a surface from region signal defined labels.

    The same region signal is used for each verex of the corresponding 1D
    surface.

    Parameters
    ----------
    signals: numpy.ndarray
        2D array with shape: (scan number, number of regions in labels_surf)

    labels_surf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        regions definition as labels. By default, the label zero is used to
        denote an absence of region. Use background_label to change it.

    background_label: number
        label to use for "no region".

    order: str
        ordering of output array ("C" or "F"). Defaults to "C".

    Returns
    -------
    surf: 2D Nisurfs-data-like
        Reconstructed surface as 2D Numpy array.
        dtype is that of "signals". Shape is that of labels_surf.

    See also
    --------
    nilearn.regions.signals_to_surf_labels
    """

    labels_surf = nisurf.check_nisurf_1d(labels_surf)
    signals = np.asarray(signals)

    labels = list(np.unique(labels_surf))
    if background_label in labels:
        labels.remove(background_label)

    # Init data as array of zeros, shape = (# or vertex, # of datapoints)
    data = np.zeros((labels_surf.shape[0], signals.shape[0]),
                    dtype=signals.dtype, order=order)

    # This seems fast enough for surface data
    # for C ordering it only takes 700ms
    # for fake data w/ 5k datapoints, 100k vertices, and 1000 unique regions
    for i, label in enumerate(labels):
        data[np.where(labels_surf == label)] = signals[:, i]

    return data


class _ExtractionFunctor(object):

    func_name = 'surf_labels_masker_extractor'

    def __init__(self, _labels_surf_, background_label, strategy):
        self._labels_surf_ = _labels_surf_
        self.background_label = background_label
        self.strategy = strategy

    def __call__(self, surfs):
        return surf_to_signals_labels(surfs, self._labels_surf_,
                                      background_label=self.background_label,
                                      strategy=self.strategy)


class SurfLabelsMasker(BaseSurfMasker, CacheMixin):
    """Class for masking of Nisurf-like objects.

    SurfLabelsMasker is useful when data from non-overlapping surfaces should
    be extracted. Use case: Extract ROIs from surface data or project surface
    ROIs onto a surface.

    Parameters
    ----------
    labels_surf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Region definitions, as one surface of labels.

    background_label: number, optional
        Label used in labels_surf to represent background.

    mask_labels_surf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to the passed labels_surf, setting where
        mask_labels_surf == 0 to the background labels.

    standardize: {'zscore', 'psc', True, False}, default is False
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

    memory: joblib.Memory or str, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: int, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    strategy: str
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, mininum, maximum, variance,
        standard_deviation
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, labels_surf, background_label=0, mask_labels_surf=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None, dtype=None,
                 memory=Memory(location=None, verbose=0), memory_level=1,
                 verbose=0, strategy="mean"):
        self.labels_surf = labels_surf
        self.background_label = background_label
        self.mask_labels_surf = mask_labels_surf

        # Parameters for clean()
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.dtype = dtype

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        available_reduction_strategies = {'mean', 'median', 'sum',
                                          'minimum', 'maximum',
                                          'standard_deviation', 'variance'}

        if strategy not in available_reduction_strategies:
            raise ValueError(str.format(
                "Invalid strategy '{}'. Valid strategies are {}.",
                strategy,
                available_reduction_strategies
            ))

        # Check background label
        if not isinstance(background_label, int):
            raise ValueError(str.format(
                "Invalid background_label '{}'. Must be int.",
                strategy
            ))

        self.strategy = strategy

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.
        """

        # Load labels_surf
        logger.log("loading data from %s" %
                   nisurf._repr_nisurfs_data(self.labels_surf)[:200],
                   verbose=self.verbose)
        self.labels_surf_ = nisurf.check_nisurf_1d(self.labels_surf)

        # Load mask_labels_surf, if any
        if self.mask_labels_surf is not None:
            logger.log("loading data from %s" %
                       nisurf._repr_nisurfs_data(self.mask_labels_surf)[:200],
                       verbose=self.verbose)
            self.mask_labels_surf_ =\
                nisurf._load_surf_mask(self.mask_labels_surf)
        else:
            self.mask_labels_surf_ = None

        # Check shapes
        if self.mask_labels_surf_ is not None:
            if self.mask_labels_surf_.shape != self.labels_surf_.shape:
                raise ValueError(
                    _compose_err_msg(
                        "Regions and mask do not have the same shape",
                        mask_labels_surf=self.mask_labels_surf,
                        labels_surf=self.labels_surf))

            # Mask labels, as set area outside of mask to background
            self.labels_surf_[~self.mask_labels_surf_] = self.background_label

        return self

    def fit_transform(self, surfs, confounds=None):
        """ Prepare and perform signal extraction from regions.
        """
        return self.fit().transform(surfs, confounds=confounds)

    def _check_fitted(self):
        if not hasattr(self, "labels_surf_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform_single_surfs(self, surfs, confounds=None):
        """Extract signals from a single 2D nisurf.

        Parameters
        ----------
        surfs: 1D/2D Nisurf-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Surfaces to process. It must boil down to a 2D Numpy array
            with scans number as last dimension.

        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        -------
        region_signals: 2D numpy.ndarray
            Signal for each label.
            shape: (number of scans, number of labels)
        """

        params = get_params(SurfLabelsMasker, self)
        region_signals, labels_ =\
            self._cache(filter_and_extract,
                        ignore=['verbose', 'memory', 'memory_level'])(
                # Images
                surfs, _ExtractionFunctor(self.labels_surf_,
                                          self.background_label,
                                          self.strategy),
                # Pre-processing
                params,
                confounds=confounds,
                dtype=self.dtype,
                # Caching
                memory=self.memory,
                memory_level=self.memory_level,
                verbose=self.verbose)

        self.labels_ = labels_

        return region_signals

    def inverse_transform(self, signals):
        """Compute surface signals from region signals

        Any mask given at initialization is taken into account.

        Parameters
        ----------
        signals: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        Returns
        -------
        surface_signals : 2D Numpy array
            Signal for each vertex
            shape: (number of scans, number of voxels)
        """

        self._check_fitted()

        logger.log("computing surface from signals", verbose=self.verbose)
        return signals_to_surf_labels(signals, self.labels_surf_,
                                      background_label=self.background_label)


            


