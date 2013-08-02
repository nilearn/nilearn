"""
CanICA
"""

# Author: Alexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause
import distutils

import numpy as np
from scipy.stats import scoreatpercentile

import sklearn
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state

from .multi_pca import MultiPCA
from .._utils.cache_mixin import CacheMixin


class CanICA(MultiPCA, CacheMixin):
    """Perform Canonical Independent Component Analysis.

    Parameters
    ----------
    mask: filename, NiImage or MultiNiftiMasker instance, optional
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

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    threshold: None, 'auto' or float
        If None, no thresholding is applied. If 'auto',
        then we apply a thresholding that will keep the n_voxels,
        more intense voxels across all the maps, n_voxels being the number
        of voxels in a brain volume. A float value indicates the
        ratio of voxels to keep (2. means keeping 2 x n_voxels voxels).

    n_init: int, optional
        The number of times the fastICA algorithm is restarted

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

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    References
    ----------
    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177
    """

    def __init__(self, mask=None, n_components=20,
                 smoothing_fwhm=6, do_cca=True,
                 threshold='auto', n_init=10,
                 random_state=0,
                 target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 # Common options
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):
        super(CanICA, self).__init__(
            mask=mask, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose, do_cca=do_cca,
            n_components=n_components, smoothing_fwhm=smoothing_fwhm,
            target_affine=target_affine, target_shape=target_shape)
        self.threshold = threshold
        self.random_state = random_state
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.n_init = n_init

    def fit(self, niimgs, y=None, confounds=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        MultiPCA.fit(self, niimgs, y=y, confounds=confounds)
        random_state = check_random_state(self.random_state)

        sparsity = np.infty
        for rs in range(self.n_init):
            if (distutils.version.LooseVersion(sklearn.__version__).version
                    > [0, 12]):
                # random_state in fastica was added in 0.13
                ica_maps_ = self._cache(fastica, memory_level=6)(
                    self.components_.T,
                    whiten=False,
                    fun='cube',
                    random_state=random_state)[2]
            else:
                ica_maps_ = self._cache(fastica, memory_level=6)(
                    self.components_.T, whiten=False,
                    fun='cube')[2]
            ica_maps_ = ica_maps_.T

            sparsity_ = np.sum(np.abs(ica_maps_), axis=1).max()
            if sparsity_ < sparsity:
                sparsity = sparsity_
                ica_maps = ica_maps_

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        elif self.threshold is not None:
            raise ValueError("Threshold must be None, "
                             "'auto' or float. You provided %s." %
                             str(self.threshold))
        if ratio is not None:
            abs_ica_maps = np.abs(ica_maps)
            threshold = scoreatpercentile(
                    abs_ica_maps,
                    100. - (100. / len(ica_maps)) * ratio)
            ica_maps[abs_ica_maps < threshold] = 0.
        self.components_ = ica_maps

        return self
