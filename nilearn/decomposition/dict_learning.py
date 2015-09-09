"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause

from __future__ import division
import os
from tempfile import mkdtemp
import warnings

# WindowsError only exist on Windows
from math import ceil
from os.path import join
import time

try:
    WindowsError
except NameError:
    WindowsError = None

import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.linear_model import Ridge, LinearRegression

from sklearn.decomposition import dict_learning_online, sparse_encode

from sklearn.base import TransformerMixin

from .canica import CanICA
from .._utils.cache_mixin import CacheMixin
from .base import DecompositionEstimator, mask_and_reduce

class DictLearning(DecompositionEstimator, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components: int
        Number of components to extract

    n_epochs: float
        Number of epochs the algorithm should run on the data

    alpha: float, optional, default=1
        Sparsity controlling parameter

    dict_init: Niimg-like object, optional
        Initial estimation of dictionary maps. Would be computed from CanICA if
        not provided

    reduction_ratio: 'auto' or float, optional
        - Between 0. or 1. : controls compression of data, 1. means no
        compression
        - if set to 'auto', estimator will guess a good compression trade-off
        between speed and accuracy

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
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

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

    max_nbytes: int,
        Size (in bytes) above which the intermediary unmasked data will be
        stored as a tempory memory map

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    References
    ----------
    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous
    activity
    Information Processing in Medical Imaging, 2011, pp. 562-573,
    Lecture Notes in Computer Science

    """

    def __init__(self, n_components=20,
                 n_epochs=1, alpha=1, dict_init=None,
                 reduction_ratio='auto',
                 random_state=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, max_nbytes=1e9, temp_dir=None, verbose=0,
                 debug_folder=None,
                 batch_size=10,
                 ):
        DecompositionEstimator.__init__(self, n_components=n_components,
                                        random_state=random_state,
                                        mask=mask,
                                        smoothing_fwhm=smoothing_fwhm,
                                        standardize=standardize,
                                        detrend=detrend,
                                        low_pass=low_pass, high_pass=high_pass,
                                        t_r=t_r,
                                        target_affine=target_affine,
                                        target_shape=target_shape,
                                        mask_strategy=mask_strategy,
                                        mask_args=mask_args,
                                        memory=memory,
                                        memory_level=memory_level,
                                        n_jobs=n_jobs,
                                        max_nbytes=max_nbytes,
                                        verbose=verbose)
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.dict_init = dict_init
        self.reduction_ratio = reduction_ratio
        self.debug_folder = debug_folder
        self.batch_size = batch_size
        self.temp_dir = temp_dir

    def _dump_debug(self):
        if hasattr(self, 'debug_info_'):
            (residual, sparsity, values) = self.debug_info_
            n_iter = residual.shape[0]
            components_img = self.masker_.inverse_transform(self.components_)
            components_img.to_filename(join(self.debug_folder,
                                            'components_%i.nii.gz' % n_iter))
            # Debug info
            np.save(join(self.debug_folder, 'residual'), residual)
            np.save(join(self.debug_folder, 'values'), values)
            np.save(join(self.debug_folder, 'time'), self.time_)


    def _init_dict(self, data):

        if self.dict_init is not None:
            components = self.masker_.transform(self.dict_init)
        else:
            canica = CanICA(n_components=self.n_components,
                            # CanICA specific parameters
                            do_cca=True, threshold=float(self.n_components),
                            n_init=1,
                            # mask parameter is not useful as we bypass masking
                            mask=None,
                            random_state=self.random_state,
                            memory=self.memory,
                            memory_level=self.memory_level,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # We use protected function _raw_fit as data
                # has already been unmasked
                canica._raw_fit(data)
            components = canica.components_
        ridge = Ridge(fit_intercept=None, alpha=0.)
        ridge.fit(components.T, data.T)
        self._dict_init = ridge.coef_.T
        S = np.sqrt(np.sum(self._dict_init ** 2, axis=0))
        S[S == 0] = 1
        self._dict_init /= S[np.newaxis, :]

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrixf
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        # Base logic for decomposition estimators
        DecompositionEstimator.fit(self, imgs)

        debug = self.debug_folder is not None

        if self.verbose:
            print('[DictLearning] Loading data')

        self.time_ = np.zeros(2)
        t0 = time.time()
        with mask_and_reduce(self.masker_, imgs, confounds,
                             reduction_ratio=self.reduction_ratio,
                             n_components=self.n_components,
                             random_state=self.random_state,
                             memory_level=max(0, self.memory_level - 1),
                             temp_dir=self.temp_dir,
                             n_jobs=self.n_jobs,
                             memory=self.memory,
                             max_nbytes=self.max_nbytes) as data:
            self.time_[1] += time.time() - t0
            if self.verbose:
                print('[DictLearning] Initializating dictionary')
            self._init_dict(data)

            if self.n_epochs < 0:
                self.n_epochs = 1
            # Performing more than 10 epochs would probably useless
            if self.n_epochs > 10:
                self.n_epochs = 10
            n_iter = int(ceil((data.shape[1] / self.batch_size * self.n_epochs)))

            if self.verbose:
                print('[DictLearning] Learning dictionary')
            t0 = time.time()
            res = self._cache(dict_learning_online,
                              func_memory_level=2)(
                data.T,
                self.n_components,
                alpha=self.alpha,
                n_iter=n_iter,
                batch_size=self.batch_size,
                method='cd',
                return_code=False,
                dict_init=self._dict_init,
                return_debug_info=debug,
                verbose=max(0, self.verbose - 1),
                random_state=self.random_state,
                shuffle=True,
                n_jobs=1)
            self.time_[0] += time.time() - t0
            if debug:
                dictionary, self.debug_info_ = res
            else:
                dictionary = res
            t0 = time.time()
            self.components_ = self._cache(sparse_encode,
                                           func_memory_level=2,
                                           ignore=['n_jobs'])\
                (data.T, dictionary, algorithm='lasso_cd', alpha=self.alpha,
                 n_jobs=self.n_jobs, check_input=False).T
            self.time_[0] += time.time() - t0

        # flip signs in each composant positive part is l1 larger
        #  than negative part
        for component in self.components_:
            if np.sum(component[component > 0]) <\
                    - np.sum(component[component < 0]):
                component *= -1

        # Normalize components
        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]

        self._dump_debug()


        return self
