"""
Base class for decomposition estimators, utilities for masking and dimension
reduction of group data
"""
from __future__ import division
from math import ceil
import itertools
from distutils.version import LooseVersion

import numpy as np

from scipy import linalg
import sklearn
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, svd_flip
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.niimg import _safe_get_data
from .._utils.compat import _basestring
from ..input_data import NiftiMapsMasker
from ..input_data.masker_validation import check_embedded_nifti_masker


def fast_svd(X, n_components, random_state=None):
    """ Automatically switch between randomized and lapack SVD (heuristic
        of scikit-learn).

    Parameters
    ==========
    X: array, shape (n_samples, n_features)
        The data to decompose

    n_components: integer
        The order of the dimensionality of the truncated SVD

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    ========

    U: array, shape (n_samples, n_components)
        The first matrix of the truncated svd

    S: array, shape (n_components)
        The second matric of the truncated svd

    V: array, shape (n_components, n_features)
        The last matric of the truncated svd

    """
    random_state = check_random_state(random_state)
    # Small problem, just call full PCA
    if max(X.shape) <= 500:
        svd_solver = 'full'
    elif n_components >= 1 and n_components < .8 * min(X.shape):
        svd_solver = 'randomized'
    # This is also the case of n_components in (0,1)
    else:
        svd_solver = 'full'

    # Call different fits for either full or truncated SVD
    if svd_solver == 'full':
        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)
        # The "copy" are there to free the reference on the non reduced
        # data, and hence clear memory early
        U = U[:, :n_components].copy()
        S = S[:n_components]
        V = V[:n_components].copy()
    else:
        if LooseVersion(sklearn.__version__) >= LooseVersion('0.17'):
            n_iter = 'auto'
        else:
            n_iter = 3
        U, S, V = randomized_svd(X, n_components=n_components,
                                 n_iter=n_iter,
                                 flip_sign=True,
                                 random_state=random_state)
    return U, S, V



def mask_and_reduce(masker, imgs,
                    confounds=None,
                    reduction_ratio='auto',
                    n_components=None, random_state=None,
                    memory_level=0,
                    memory=Memory(cachedir=None),
                    n_jobs=1):
    """Mask and reduce provided 4D images with given masker.

    Uses a PCA (randomized for small reduction ratio) or a range finding matrix
    on time series to reduce data size in time direction. For multiple images,
    the concatenation of data is returned, either as an ndarray or a memorymap
    (useful for big datasets that do not fit in memory).

    Parameters
    ----------
    masker: NiftiMasker or MultiNiftiMasker
        Instance used to mask provided data.

    imgs: list of 4D Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html
        List of subject data to mask, reduce and stack.

    confounds: CSV file path or 2D matrix, optional
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    reduction_ratio: 'auto' or float between 0. and 1.
        - Between 0. or 1. : controls data reduction in the temporal domain
        , 1. means no reduction, < 1. calls for an SVD based reduction.
        - if set to 'auto', estimator will set the number of components per
          reduced session to be n_components.

    n_components: integer, optional
        Number of components per subject to be extracted by dimension reduction

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    memory_level: integer, optional
        Integer indicating the level of memorization. The higher, the more
        function calls are cached.

    memory: joblib.Memory
        Used to cache the function calls.

    Returns
    ------
    data: ndarray or memorymap
        Concatenation of reduced data.
    """
    if not hasattr(imgs, '__iter__'):
        imgs = [imgs]

    if reduction_ratio == 'auto':
        if n_components is None:
            # Reduction ratio is 1 if
            # neither n_components nor ratio is provided
            reduction_ratio = 1
    else:
        if reduction_ratio is None:
            reduction_ratio = 1
        else:
            reduction_ratio = float(reduction_ratio)
        if not 0 <= reduction_ratio <= 1:
            raise ValueError('Reduction ratio should be between 0. and 1.,'
                             'got %.2f' % reduction_ratio)

    if confounds is None:
        confounds = itertools.repeat(confounds)

    if reduction_ratio == 'auto':
        n_samples = n_components
        reduction_ratio = None
    else:
        # We'll let _mask_and_reduce_single decide on the number of
        # samples based on the reduction_ratio
        n_samples = None

    data_list = Parallel(n_jobs=n_jobs)(
        delayed(_mask_and_reduce_single)(
            masker,
            img, confound,
            reduction_ratio=reduction_ratio,
            n_samples=n_samples,
            memory=memory,
            memory_level=memory_level,
            random_state=random_state
        ) for img, confound in zip(imgs, confounds))

    subject_n_samples = [subject_data.shape[0]
                         for subject_data in data_list]

    n_samples = np.sum(subject_n_samples)
    n_voxels = int(np.sum(_safe_get_data(masker.mask_img_)))
    data = np.empty((n_samples, n_voxels), order='F',
                    dtype='float64')

    current_position = 0
    for i, next_position in enumerate(np.cumsum(subject_n_samples)):
        data[current_position:next_position] = data_list[i]
        current_position = next_position
        # Clear memory as fast as possible: remove the reference on
        # the corresponding block of data
        data_list[i] = None
    return data


def _mask_and_reduce_single(masker,
                            img, confound,
                            reduction_ratio=None,
                            n_samples=None,
                            memory=None,
                            memory_level=0,
                            random_state=None):
    """Utility function for multiprocessing from MaskReducer"""
    this_data = masker.transform(img, confound)
    # Now get rid of the img as fast as possible, to free a
    # reference count on it, and possibly free the corresponding
    # data
    del img
    random_state = check_random_state(random_state)

    data_n_samples = this_data.shape[0]
    if reduction_ratio is None:
        assert n_samples is not None
        n_samples = min(n_samples, data_n_samples)
    else:
        n_samples = int(ceil(data_n_samples * reduction_ratio))

    U, S, V = cache(fast_svd, memory,
                        memory_level=memory_level,
                        func_memory_level=3)(this_data.T,
                                             n_samples,
                                             random_state=random_state)
    U = U.T.copy()
    U = U * S[:, np.newaxis]
    return U


class BaseDecomposition(BaseEstimator, CacheMixin):
    """Base class for matrix factorization based decomposition estimators.

    Handles mask logic, provides transform and inverse_transform methods

     .. versionadded:: 0.2

    Parameters
    ----------
    n_components: int
        Number of components to extract, for each 4D-Niimage

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

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

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background', 'epi'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or
        masking.compute_epi_mask. Default is 'background'.

    mask_args: dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory: instance of joblib.Memory or str
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Attributes
    ----------
    `mask_img_` : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    """

    def __init__(self, n_components=20,
                 random_state=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1,
                 verbose=0):
        self.n_components = n_components
        self.random_state = random_state
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, imgs, y=None, confounds=None):
        """Base fit for decomposition estimators : compute the embedded masker

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which the mask is calculated. If this is a list,
            the affine is considered the same for all.

        """
        if isinstance(imgs, _basestring) or not hasattr(imgs, '__iter__'):
            # these classes are meant for list of 4D images
            # (multi-subject), we want it to work also on a single
            # subject, so we hack it.
            imgs = [imgs, ]
        if len(imgs) == 0:
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError('Need one or more Niimg-like objects as input, '
                             'an empty list was given.')
        self.masker_ = check_embedded_nifti_masker(self)

        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        return self

    def _check_components_(self):
        if not hasattr(self, 'components_'):
            if self.__class__.__name__ == 'BaseDecomposition':
                raise ValueError("Object has no components_ attribute. "
                                 "This may be because "
                                 "BaseDecomposition is directly "
                                 "being used.")
            else:
                raise ValueError("Object has no components_ attribute. "
                                 "This is probably because fit has not "
                                 "been called.")

    def transform(self, imgs, confounds=None):
        """Project the data into a reduced representation

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data to be projected

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        Returns
        ----------
        loadings: list of 2D ndarray,
            For each subject, each sample, loadings for each decomposition
            components
            shape: number of subjects * (number of scans, number of regions)

        """

        self._check_components_()
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 4D/ list of 4D data?
        if confounds is None:
            confounds = [None] * len(imgs)
        return [nifti_maps_masker.transform(img, confounds=confound)
                for img, confound in zip(imgs, confounds)]

    def inverse_transform(self, loadings):
        """Use provided loadings to compute corresponding linear component
        combination in whole-brain voxel space

        Parameters
        ----------
        loadings: list of numpy array (n_samples x n_components)
            Component signals to tranform back into voxel signals

        Returns
        -------
        reconstructed_imgs: list of nibabel.Nifti1Image
            For each loading, reconstructed Nifti1Image

        """
        if not hasattr(self, 'components_'):
            ValueError('Object has no components_ attribute. This is either '
                       'because fit has not been called or because'
                       '_DecompositionEstimator has directly been used')
        self._check_components_()
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 2D/ list of 2D data?
        return [nifti_maps_masker.inverse_transform(loading)
                for loading in loadings]

    def _sort_by_score(self, data):
        """Sort components on the explained variance over data of estimator
        components_"""
        components_score = self._raw_score(data, per_component=True)
        order = np.argsort(components_score)[::-1]
        self.components_ = self.components_[order]

    def _raw_score(self, data, per_component=True):
        """Return explained variance over data of estimator components_"""
        return self._cache(explained_variance)(data, self.components_,
                                               per_component=per_component)

    def score(self, imgs, confounds=None):
        """Score function based on explained variance on imgs.

        Should only be used by DecompositionEstimator derived classes

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data to be scored

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        Returns
        -------
        score: float,
            Holds the score for each subjects. Score is two dimensional
            if per_component is True. First dimension
            is squeezed if the number of subjects is one
        """
        data = mask_and_reduce(self.masker_, imgs, confounds,
                               reduction_ratio=1.)
        return self._raw_score(data, per_component=False)


def explained_variance(X, components, per_component=True):
    """Score function based on explained variance

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components_

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """
    full_var = np.var(X)
    n_components = components.shape[0]
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components = components / S[:, np.newaxis]
    projected_data = components.dot(X.T)
    if per_component:
        res_var = np.zeros(n_components)
        for i in range(n_components):
            res = X - np.outer(projected_data[i],
                               components[i])
            res_var[i] = np.var(res)
        return np.maximum(0., 1. - res_var / full_var)
    else:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(components.T, X.T)
        res_var = X - lr.coef_.dot(components)
        res_var **= 2
        res_var = np.sum(res_var)
        return np.maximum(0., 1. - res_var / full_var)
