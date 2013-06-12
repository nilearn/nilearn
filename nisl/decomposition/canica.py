"""
CanICA
"""

# Author: Alexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause
import distutils

import numpy as np
from scipy import stats

import sklearn
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

from .multi_pca import MultiPCA
from .._utils.cache_mixin import cache
from ..io import NiftiMultiMasker


class CanICA(MultiPCA):
    """Perform Canonical Independent Component Analysis.

    Parameters
    ----------
    data: array-like, shape = [[n_samples, n_features], ...]
        Training vector, where n_samples is the number of samples,
        n_features is the number of features. There is one vector per
        subject.

    n_components: int
        Number of components to extract

    kurtosis_thr: boolean or float
        If kurtosis_thr is None, the algorithm is run regardless of the
        kurtosis. If it is False, then the algorithm will iter on the
        number of components to find a kurtosis greater than their number.
        If float, the kurtosis will additionally be thresholded by the
        given value.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    References
    ----------
    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177
    """

    def __init__(self, mask=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 # MultiPCA options
                 do_cca=True, n_components=20,
                 # CanICA options
                 kurtosis_thr=None, threshold='auto', random_state=0,
                 # Common options
             ):
        super(CanICA, self).__init__(
            mask, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose, do_cca=do_cca,
            n_components=n_components)
        self.kurtosis_thr = kurtosis_thr
        self.threshold = threshold
        self.random_state = random_state

    def _find_high_kurtosis(self, pcas, ref_memory_level=0,
                            memory=Memory(cachedir=None)):
        random_state = check_random_state(self.random_state)

        if not self.kurtosis_thr:
            kurtosis_thr = -np.inf
        else:
            kurtosis_thr = self.kurtosis_thr
        n_components = self.n_components

        while n_components < 3 * self.n_components:
            group_maps = cache(
                randomized_svd, memory, ref_memory_level, memory_level=2
            )(pcas, n_components)[0]
            group_maps = group_maps[:, :n_components]

            if (distutils.version.LooseVersion(sklearn.__version__).version
                    > [0, 12]):
                # random_state in fastica was added in 0.13
                ica_maps = cache(fastica, memory, ref_memory_level,
                                 memory_level=1)(
                    group_maps,
                    whiten=False,
                    fun='cube',
                    random_state=random_state)[2]
            else:
                ica_maps = cache(fastica, memory, ref_memory_level,
                                 memory_level=1)(group_maps, whiten=False,
                                             fun='cube')[2]
            ica_maps = ica_maps.T
            kurtosis = stats.kurtosis(ica_maps, axis=1)
            kurtosis_mask = kurtosis > kurtosis_thr
            if np.sum(kurtosis_mask) >= n_components:
                order = np.argsort(kurtosis)[::-1]
                ica_maps = ica_maps[order[:n_components]]
                break
            n_components += 1

            del group_maps
        else:
            raise ValueError('Could not find components with high-enough'
                             ' kurtosis')
        self.n_components_ = n_components
        return ica_maps

    def fit(self, data, y=None):

        MultiPCA.fit(self, data)
        ica_maps = self._find_high_kurtosis(self.components_.T,
                                            ref_memory_level=self.memory,
                                            memory=self.memory)

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        if self.threshold is not None:
            raveled = np.abs(ica_maps).ravel()
            argsort = np.argsort(raveled)
            n_voxels = ica_maps[0].size
            threshold = raveled[argsort[- ratio * n_voxels]]
            ica_maps[np.abs(ica_maps) < threshold] = 0.

        self.components_ = ica_maps
        # For the moment, store also the components_img
        self.components_img_ = \
            self.mask.inverse_transform(ica_maps)

        return self