"""
CanICA
"""

# Author: ALexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause
import copy

import numpy as np
from scipy import linalg, stats

from sklearn.base import TransformerMixin
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

from .decomposition_model import DecompositionModel


def subject_pca(subject_data, n_components, mem):
    subject_data -= subject_data.mean(axis=0)
    std = subject_data.std(axis=0)
    std[std == 0] = 1
    subject_data /= std
    subject_data = subject_data.T
    subject_data = mem.cache(linalg.svd)(subject_data,
                                         full_matrices=False)[0]
    # We copy here to avoid keeping a reference on the big array
    subject_data = subject_data[:, :2 * n_components].copy()
    return subject_data


class CanICA(DecompositionModel, TransformerMixin):
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

    Returns
    -------


    Notes
    -----


    """

    def __init__(self, n_components,
                 memory=Memory(cachedir=None),
                 kurtosis_thr=None,
                 maps_only=False,
                 random_state=None,
                 n_jobs=1, verbose=0):
        self.n_components = n_components
        self.memory = memory
        self.kurtosis_thr = kurtosis_thr
        self.maps_only = maps_only
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _find_high_kurtosis(self, pcas, memory):
        random_state = check_random_state(self.random_state)

        if self.kurtosis_thr is False:
            kurtosis_thr = -np.inf
        else:
            kurtosis_thr = self.kurtosis_thr
        n_components = self.n_components

        while n_components < 3 * self.n_components:
            group_maps = memory.cache(
                randomized_svd)(pcas, n_components)[0]
            group_maps = group_maps[:, :n_components]

            ica_maps = memory.cache(fastica)(group_maps, whiten=False,
                                             fun='cube',
                                             random_state=random_state)[2]
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

    def fit(self, data, Y=None):
        if hasattr(data, 'copy'):
            # It's an array
            data = data.copy()
        else:
            # Probably a list
            data = copy.deepcopy(data)

        memory = self.memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        pcas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(subject_pca)(subject_data,
                                 n_components=self.n_components, mem=memory)
            for subject_data in data)
        pcas = np.concatenate(pcas, axis=1)

        if self.kurtosis_thr is None:
            group_maps = memory.cache(randomized_svd)(
                pcas, self.n_components)[0]
            group_maps = group_maps[:, :self.n_components]
            ica_maps = memory.cache(fastica)(group_maps, whiten=False,
                                             fun='cube',
                                             random_state=self.random_state)[2]
            ica_maps = ica_maps.T
        else:
            ica_maps = self._find_high_kurtosis(pcas, memory)

        del pcas
        self.maps_ = ica_maps
        if not self.maps_only:
            # Relearn the time series
            self.learn_from_maps(data)

        return self

    def transform(self, X, y=None):
        """Apply un-mixing matrix "W" to X to recover the sources

            S = X * W.T
        """
        return np.dot(X, self.maps_.T)
