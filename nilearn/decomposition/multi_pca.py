"""
PCA dimension reduction on multiple subjects.
This is a good initialization method for ICA.
"""
import os
import shutil
import warnings
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.utils.extmath import randomized_svd
from sklearn.base import TransformerMixin

from .._utils.cache_mixin import CacheMixin
from .base import DecompositionEstimator, mask_and_reduce

class MultiPCA(DecompositionEstimator, TransformerMixin, CacheMixin):
    """Perform Multi Subject Principal Component Analysis.
    This is a good initialization method for ICA.

    Perform a PCA on each subject, stack the results, and reduce them
    at group level. An optional Canonical Correlation Analysis can be
     performed at group level.

    Parameters
    ----------
    n_components: int
        Number of components to extract

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    mask: Niimg-like object, instance of NiftiMasker or MultiNiftiMasker,
     optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
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

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    max_nbytes: int,
        Size (in bytes) above which the intermediary unmasked data will be
        stored as a tempory memory map

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Attributes
    ----------
    `_pca_masker_`: instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_`: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `components_`: 2D numpy array (n_components x n-voxels)
        Array of masked extracted components. They can be unmasked thanks to
        the `_pca_masker_` attribute.
    """

    def __init__(self, n_components=20, do_cca=True,
                 random_state=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, max_nbytes=1e9,
                 verbose=0
                 ):
        self.n_components = n_components
        self.do_cca = do_cca

        DecompositionEstimator.__init__(self, n_components=n_components,
                                        random_state=random_state,
                                        mask=mask,
                                        smoothing_fwhm=smoothing_fwhm,
                                        standardize=standardize,
                                        detrend=detrend,
                                        low_pass=low_pass,
                                        high_pass=high_pass, t_r=t_r,
                                        target_affine=target_affine,
                                        target_shape=target_shape,
                                        mask_strategy=mask_strategy,
                                        mask_args=mask_args,
                                        memory=memory,
                                        memory_level=memory_level,
                                        n_jobs=n_jobs, max_nbytes=max_nbytes,
                                        verbose=verbose)
        
    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        DecompositionEstimator.fit(self, imgs)

        with mask_and_reduce(self.masker_, imgs, confounds,
                             n_components=self.n_components,
                             random_state=self.random_state,
                             memory=self.memory,
                             memory_level=max(0, self.memory_level -1),
                             max_nbytes=self.max_nbytes) as data:
            self._raw_fit(data)
        return self

    def _raw_fit(self, data):
        """Helper function that direcly process unmasked data"""
        if self.do_cca:
            S = np.sqrt(np.sum(data ** 2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]
        self.components_, self.variance_, _ = self._cache(
            randomized_svd, func_memory_level=2)(
                data.T, n_components=self.n_components,
            random_state=self.random_state)
        if self.do_cca:
            data *= S[:, np.newaxis]
        self.components_ = self.components_.T

