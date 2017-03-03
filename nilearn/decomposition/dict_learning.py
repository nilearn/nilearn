"""
Dictionary learning estimator: Perform a map learning algorithm by learning
a temporal dense dictionary along with sparse spatial loadings, that
constitutes output maps
"""

# Author: Arthur Mensch
# License: BSD 3 clause

from __future__ import division

import warnings
from distutils.version import LooseVersion

import numpy as np
import sklearn
from sklearn.base import TransformerMixin
from sklearn.decomposition import dict_learning_online
from sklearn.externals.joblib import Memory
from sklearn.linear_model import Ridge

from .base import BaseDecomposition, mask_and_reduce
from .canica import CanICA


if LooseVersion(sklearn.__version__) >= LooseVersion('0.17'):
    # check_input=False is an optimization available only in sklearn >=0.17
    sparse_encode_args = {'check_input': False}


def _compute_loadings(components, data):
    ridge = Ridge(fit_intercept=None, alpha=1e-8)
    ridge.fit(components.T, np.asarray(data.T))
    loadings = ridge.coef_.T

    S = np.sqrt(np.sum(loadings ** 2, axis=0))
    S[S == 0] = 1
    loadings /= S[np.newaxis, :]
    return loadings


class DictLearning(BaseDecomposition, TransformerMixin):
    """Perform a map learning algorithm based on spatial component sparsity,
    over a CanICA initialization.  This yields more stable maps than CanICA.

     .. versionadded:: 0.2

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components: int
        Number of components to extract.

    batch_size : int, optional, default=20
        The number of samples to take in each batch.

    n_epochs: float
        Number of epochs the algorithm should run on the data.

    alpha: float, optional, default=1
        Sparsity controlling parameter.

    dict_init: Niimg-like object, optional
        Initial estimation of dictionary maps. Would be computed from CanICA if
        not provided.

    reduction_ratio: 'auto' or float between 0. and 1.
        - Between 0. or 1. : controls data reduction in the temporal domain.
          1. means no reduction, < 1. calls for an SVD based reduction.
        - if set to 'auto', estimator will set the number of components per
          reduced session to be n_components.

    method : {'lars', 'cd'}
        Coding method used by sklearn backend. Below are the possible values.
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    References
    ----------
    * Arthur Mensch, Gael Varoquaux, Bertrand Thirion,
      Compressed online dictionary learning for fast resting-state fMRI
      decomposition.
      IEEE 13th International Symposium on Biomedical Imaging (ISBI), 2016.
      pp. 1282-1285
    """

    def __init__(self, n_components=20,
                 n_epochs=1, alpha=10, reduction_ratio='auto', dict_init=None,
                 random_state=None, batch_size=20, method="cd", mask=None,
                 smoothing_fwhm=4, standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None, target_affine=None,
                 target_shape=None, mask_strategy='epi', mask_args=None,
                 n_jobs=1, verbose=0, memory=Memory(cachedir=None),
                 memory_level=0):
        BaseDecomposition.__init__(self, n_components=n_components,
                                   random_state=random_state, mask=mask,
                                   smoothing_fwhm=smoothing_fwhm,
                                   standardize=standardize, detrend=detrend,
                                   low_pass=low_pass, high_pass=high_pass,
                                   t_r=t_r, target_affine=target_affine,
                                   target_shape=target_shape,
                                   mask_strategy=mask_strategy,
                                   mask_args=mask_args, memory=memory,
                                   memory_level=memory_level, n_jobs=n_jobs,
                                   verbose=verbose)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.method = method
        self.alpha = alpha
        self.reduction_ratio = reduction_ratio
        self.dict_init = dict_init

    def _init_dict(self, data):
        if self.dict_init is not None:
            components = self.masker_.transform(self.dict_init)
        else:
            canica = CanICA(n_components=self.n_components,
                            # CanICA specific parameters
                            do_cca=True, threshold=float(self.n_components),
                            n_init=1,
                            # mask parameter is not useful as we bypass masking
                            mask=self.masker_, random_state=self.random_state,
                            memory=self.memory, memory_level=self.memory_level,
                            n_jobs=self.n_jobs, verbose=self.verbose)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # We use protected function _raw_fit as data
                # has already been unmasked
                canica._raw_fit(data)
            components = canica.components_
        S = (components ** 2).sum(axis=1)
        S[S == 0] = 1
        components /= S[:, np.newaxis]
        self.components_init_ = components

    def _init_loadings(self, data):
        self.loadings_init_ = self._cache(_compute_loadings)(
            self.components_init_, data)

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and component maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        # Base logic for decomposition estimators
        BaseDecomposition.fit(self, imgs)

        if self.verbose:
            print('[DictLearning] Loading data')
        data = mask_and_reduce(self.masker_, imgs, confounds,
                               reduction_ratio=self.reduction_ratio,
                               n_components=self.n_components,
                               random_state=self.random_state,
                               memory_level=max(0, self.memory_level - 1),
                               n_jobs=self.n_jobs, memory=self.memory)
        if self.verbose:
            print('[DictLearning] Learning initial components')
        self._init_dict(data)
        self._raw_fit(data)
        return self

    def _raw_fit(self, data):
        """Helper function that direcly process unmasked data

        Parameters
        ----------
        data: ndarray,
            Shape (n_samples, n_features)
        """
        _, n_features = data.shape

        if self.verbose:
            print('[DictLearning] Computing initial loadings')
        self._init_loadings(data)

        dict_init = self.loadings_init_

        n_iter = ((n_features - 1) // self.batch_size + 1) * self.n_epochs

        if self.verbose:
            print('[DictLearning] Learning dictionary')
        self.components_, _ = self._cache(dict_learning_online)(
            data.T, self.n_components, alpha=self.alpha, n_iter=n_iter,
            batch_size=self.batch_size, method=self.method,
            dict_init=dict_init, verbose=max(0, self.verbose - 1),
            random_state=self.random_state, return_code=True, shuffle=True,
            n_jobs=1)
        self.components_ = self.components_.T
        # Unit-variance scaling
        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]

        # Flip signs in each composant so that positive part is l1 larger
        # than negative part. Empirically this yield more positive looking maps
        # than with setting the max to be positive.
        for component in self.components_:
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1

        return self
