"""
PCA dimension reduction on multiple subjects.
This is a good initialization method for ICA.
"""
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.utils.extmath import randomized_svd
from sklearn.base import TransformerMixin

from .base import BaseDecomposition, mask_and_reduce


class MultiPCA(BaseDecomposition, TransformerMixin):
    """Perform Multi Subject Principal Component Analysis.

    Perform a PCA on each subject, stack the results, and reduce them
    at group level. An optional Canonical Correlation Analysis can be
    performed at group level. This is a good initialization method for ICA.

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

    mask: Niimg-like object, instance of NiftiMasker or MultiNiftiMasker, optional
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

    incremental_group_pca: bool, optional
        Whether to use an IncrementalPCA for the group PCA. This will
        reduce memory usage possibly at the expense of precision. This
        feature is only supported for scikit-learn versions 0.16 onwards.

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

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Attributes
    ----------
    `masker_` : instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_` : Niimg-like object
        See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `components_` : 2D numpy array (n_components x n-voxels)
        Array of masked extracted components. They can be unmasked thanks to
        the `masker_` attribute.

    """

    def __init__(self, n_components=20,
                 mask=None,
                 smoothing_fwhm=None,
                 do_cca=True,
                 random_state=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 incremental_group_pca=False,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1,
                 verbose=0
                 ):
        self.n_components = n_components
        self.do_cca = do_cca
        self.incremental_group_pca = incremental_group_pca

        if self.incremental_group_pca:
            try:
                from sklearn.decomposition.incremental_pca import IncrementalPCA
            except ImportError as exc:
                import sklearn
                message = ('IncrementalPCA is only supported in '
                           'scikit-learn version 0.16 onwards, '
                           'your scikit-learn version is {0}. '
                           "Please set 'incremental_group_pca' "
                           'to False').format(sklearn.__version__)
                exc.args += (message, )
                raise

        BaseDecomposition.__init__(self, n_components=n_components,
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
                                   n_jobs=n_jobs,
                                   verbose=verbose)

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/manipulating_images.html#niimg.
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        """
        if hasattr(imgs, '__iter__'):
            self._nb_subjects_ = len(imgs)
        else:
            self._nb_subjects_ = 1

        BaseDecomposition.fit(self, imgs)

        data = mask_and_reduce(self.masker_, imgs,
                               confounds=confounds,
                               n_components=self.n_components,
                               random_state=self.random_state,
                               memory=self.memory,
                               memory_level=max(0, self.memory_level - 1),
                               n_jobs=self.n_jobs)
        self._raw_fit(data)

        return self

    def _raw_fit(self, data):
        """Helper function that directly process unmasked data"""
        if self.do_cca:
            S = np.sqrt(np.sum(data ** 2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]

        self.components_, self.variance_ = self._group_pca(data)

        if self.do_cca:
            data *= S[:, np.newaxis]

    def _group_pca(self, data):
        if self.incremental_group_pca:
            components, variance = self._cache(
                _incremental_group_pca, func_memory_level=2)(
                    data, self.n_components, self._nb_subjects_)
        else:
            components, variance, _ = self._cache(
                randomized_svd, func_memory_level=2)(
                    data.T, n_components=self.n_components,
                    transpose=True,
                    random_state=self.random_state, n_iter=3)
            components = components.T

        return components, variance


def _incremental_group_pca(subject_pcas, n_components, nb_subjects):
    # delayed import because IncrementalPCA is only supported for
    # sklearn versions > 0.16
    from sklearn.decomposition.incremental_pca import IncrementalPCA

    pca = IncrementalPCA(n_components=n_components)
    subject_length = len(subject_pcas) // nb_subjects
    subject_indices = (slice(i, i + subject_length)
                       for i in range(0, len(subject_pcas), subject_length))

    for subject_slice in subject_indices:
        pca.partial_fit(subject_pcas[subject_slice])

    return pca.components_, pca.singular_values_
