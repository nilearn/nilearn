"""
DictLearning
"""

# Author: Arthur Mensch
# License: BSD 3 clause

from distutils.version import LooseVersion

import sklearn

import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.linear_model import Ridge

from sklearn.decomposition import dict_learning_online

from sklearn.base import TransformerMixin

from .._utils import as_ndarray
from .canica import CanICA
from .._utils.cache_mixin import CacheMixin
from ._subject_pca import _SubjectPCA
from .._utils.class_inspect import get_params



class DictLearning(_SubjectPCA, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    data: array-like, shape = [[n_samples, n_features], ...]
        Training vector, where n_samples is the number of samples,
        n_features is the number of features. There is one vector per
        subject.

    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    alpha: float, optional, default=1
        Sparsity controlling parameter

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

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

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    References
    ----------

    """

    def __init__(self, mask=None, n_components=20,
                 smoothing_fwhm=6,
                 standardize=True,
                 random_state=0,
                 target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 alpha=1,
                 n_iter=1000,
                 # Common options
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 dict_init=None
                 ):
        _SubjectPCA.__init__(self,
                           mask=mask, memory=memory, memory_level=memory_level,
                           n_jobs=n_jobs, verbose=verbose,
                           n_components=n_components, smoothing_fwhm=smoothing_fwhm,
                           target_affine=target_affine, target_shape=target_shape,
                           random_state=random_state, high_pass=high_pass, low_pass=low_pass,
                           t_r=t_r,
                           standardize=standardize,
                           whiten=False)
        self.n_iter = n_iter
        self.alpha = alpha
        self.dict_init = dict_init

    def _init_dict(self, imgs, y=None, confounds=None):
        if self.dict_init is not None:
            self.dict_init_ = self.dict_init
        else:
            # Perform a CanICA initialization using SinglePCA parameters
            parameters = get_params(_SubjectPCA, self).copy()
            # Efficient CanICA requires CCA
            parameters.pop('whiten')
            canica = CanICA(do_cca=True, threshold=float(self.n_components), n_init=1,
                            **parameters)
            canica.fit(imgs, y=y, confounds=confounds)

            ridge = Ridge(alpha=1e-6, fit_intercept=None)
            ridge.fit(canica.components_.T, self.components_.T)
            self.dict_init_ = ridge.coef_.T
            S = np.sqrt(np.sum(self.dict_init_ ** 2, axis=0))
            self.dict_init_ /= S[np.newaxis, :]

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        if self.n_iter == 'auto':
            # ceil(self.data_fat.shape[0] / self.batch_size)
            self.n_iter = (self.data_flat_.shape[0] - 1) / self.batch_size + 1

        if self.verbose:
            print('[DictLearning] Loading data')
        _SubjectPCA.fit(self, imgs, y=y, confounds=confounds)

        if self.verbose:
            print('[DictLearning] Initializating dictionary')
        self._init_dict(imgs, y, confounds)

        if self.verbose:
            print('[DictLearning] Learning dictionary')
        parameters = {}
        if LooseVersion(sklearn.__version__).version >= [0, 13]:
            parameters['batch_size'] = 10
        else:
            parameters['chunk_size'] = 10
        self.components_, _ = self._cache(dict_learning_online, func_memory_level=2)(
            self.components_.T,
            self.n_components,
            alpha=self.alpha,
            n_iter=self.n_iter,
            method='lars',
            return_code=True,
            verbose=max(0, self.verbose - 1),
            random_state=self.random_state,
            shuffle=True,
            n_jobs=1, **parameters)
        self.components_ = self.components_.T
        if self.verbose:
            print('')
            print('[DictLearning] Learning code')
        self.components_ = as_ndarray(self.components_)
        # flip signs in each composant positive part is l1 larger than negative part
        for component in self.components_:
            if np.sum(component[component > 0]) < - np.sum(component[component <= 0]):
                component *= -1

        return self