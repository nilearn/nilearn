"""
Transformer used to apply basic transformations on surface data.
"""
# Author: Sage Hahn
# License: simplified BSD

import abc

from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Memory

from ..surface import nisurf
from .. import signal
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.class_inspect import enclosing_scope_name


def filter_and_extract(surfs, extraction_function,
                       parameters,
                       memory_level=0, memory=Memory(location=None),
                       verbose=0,
                       confounds=None,
                       copy=True,
                       dtype=None):
    """Extract representative time series using given function.

    Parameters
    ----------
    surfs: 1D/2D Nisurf-data-like object
        Surfaces to be masked as Numpy arrays.
        Can be 1-dimensional or 2-dimensional.

    extraction_function: function
        Function used to extract the time series from 2D data. This function
        should take surfaces as argument and returns a tuple containing a 2D
        array with masked signals along with a auxiliary value used if
        returning a second value is needed.
        If any other parameter is needed, a functor or a partial
        function must be provided.

    Returns
    -------
    signals: 2D numpy array
        Signals extracted using the extraction function. It is a scikit-learn
        friendly 2D array with shape n_samples x n_features.
    """
    # Since the calling class can be any *Surf*Masker, we look for exact type
    if verbose > 0:
        class_name = enclosing_scope_name(stack_level=10)

    if verbose > 0:
        print("[%s] Loading data from %s" % (
            class_name,
            nisurf._repr_nisurfs_data(surfs)[:200]))

    # Load surface ensuring 2D
    surfs = nisurf.check_nisurf_2d(surfs, atleast_2d=True, dtype=dtype)

    if verbose > 0:
        print("[%s] Extracting region signals" % class_name)
    region_signals, aux = cache(extraction_function, memory,
                                func_memory_level=2,
                                memory_level=memory_level)(surfs)

    # Temporal
    # --------
    # Detrending (optional)
    # Filtering
    # Confounds removing (from csv file or numpy array)
    # Normalizing
    if verbose > 0:
        print("[%s] Cleaning extracted signals" % class_name)
    sessions = parameters.get('sessions')
    region_signals = cache(
        signal.clean, memory=memory, func_memory_level=2,
        memory_level=memory_level)(
            region_signals,
            detrend=parameters['detrend'],
            standardize=parameters['standardize'],
            t_r=parameters['t_r'],
            low_pass=parameters['low_pass'],
            high_pass=parameters['high_pass'],
            confounds=confounds,
            sessions=sessions)

    return region_signals, aux


class BaseSurfMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Base class for SurfMaskers
    """

    @abc.abstractmethod
    def transform_single_surfs(self, surfs, confounds=None, copy=True):
        """Extract signals from a single 1 or 2D surface.

        Parameters
        ----------
        surfs: Nisurf-data-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Surfaces to process. If 2D, must have shape
            (number of surfs, number of elements).

        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        -------
        region_signals: 2D numpy.ndarray
            Signal for each element.
            shape: (number of scans, number of elements)
        """
        raise NotImplementedError()

    def transform(self, surfs, confounds=None):
        """Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        surfs: Nisurf-data-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Surfaces to process. If 2D, must have shape
            (number of surfs, number of elements).

        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        -------
        region_signals: 2D numpy.ndarray
            Signal for each element.
            shape: (number of scans, number of elements)
        """

        self._check_fitted()

        return self.transform_single_surfs(surfs, confounds)

    @abc.abstractmethod
    def fit_transform(self, X, y=None, confounds=None, **fit_params):
        """Fit to data, then transform it

        Parameters
        ----------
        X : Nisurf-data-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Surfaces to process. If 2D, must have shape
            (number of surfs, number of elements).

        y : numpy array of shape [n_samples]
            Target values.

        confounds: list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length as surfs_list.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, X):
        """ Transform the 2D data matrix back to surface brain space.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _check_fitted(self):
        raise NotImplementedError()
