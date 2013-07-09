"""
Sparse recovery of group maps using ICA.
"""
import copy

import numpy as np
from scipy import linalg, stats

from sklearn.decomposition.fastica_ import fastica
from sklearn.utils.extmath import randomized_svd
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.base import BaseEstimator


def subject_pca(subject_data, n_components, mem, randomized_svd=False):
    subject_data = np.asarray(subject_data).copy()
    subject_data -= subject_data.mean(axis=0)
    # PCA
    std = subject_data.std(axis=0)
    std[std == 0] = 1
    subject_data /= std
    subject_data = subject_data.T
    if randomized_svd:
        subject_data = mem.cache(randomized_svd)(subject_data, n_components)[0]
    else:
        subject_data = mem.cache(linalg.svd)(subject_data,
                                                full_matrices=False)[0]
    # We copy here to avoid keeping a reference on the big array
    subject_data = subject_data[:, :2 * n_components].copy()
    return subject_data


###############################################################################
class ICAModel(BaseEstimator):
    """ Learn our model using ICA and thresholding.
    """

    def __init__(self, n_components, threshold=1,
                       mem=Memory(cachedir=None), store_pca=False,
                       kurtosis_thr=False, maps_only=False,
                       randomized_svd=False,
                       n_jobs=1, verbose=0):
        self.n_components = n_components
        self.mem = mem
        self.threshold = threshold
        self.store_pca = store_pca
        self.kurtosis_thr = kurtosis_thr
        self.n_jobs = n_jobs
        self.randomized_svd = randomized_svd
        self.verbose = verbose
        self.maps_only = maps_only

    def fit(self, data, y=None):
        if hasattr(data, 'copy'):
            # It's an array
            data = data.copy()
        else:
            # Probably a list
            data = copy.deepcopy(data)
        pcas = list()
        # Do PCAs and CCAs
        pcas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(subject_pca)(subject_data,
                        n_components=self.n_components, mem=self.mem,
                        randomized_svd=randomized_svd)
                for subject_data in data)

        if self.kurtosis_thr is False:
            kurtosis_thr = -np.inf
        else:
            kurtosis_thr = self.kurtosis_thr
        n_components = self.n_components
        pcas = np.concatenate(pcas, axis=1)
        while n_components < 3 * self.n_components:
            group_maps = self.mem.cache(randomized_svd)(pcas, n_components)[0]
            if self.store_pca:
                self.pca_ = group_maps
            group_maps = group_maps[:, :n_components]

            ica_maps = self.mem.cache(fastica)(group_maps, whiten=False,
                                               fun='cube')[2]
            ica_maps = ica_maps.T
            kurtosis = stats.kurtosis(ica_maps, axis=1)
            kurtosis_mask = kurtosis > kurtosis_thr
            if np.sum(kurtosis_mask) >= self.n_components:
                order = np.argsort(kurtosis)[::-1]
                ica_maps = ica_maps[order[:self.n_components]]
                break
            n_components += 1
            del group_maps
        else:
            raise ValueError('Could not find components with high-enough'
            ' kurtosis')
        del pcas
        if not ica_maps.flags.writeable:
            ica_maps = np.asarray(ica_maps).copy()
        # Threshold
        ica_maps[np.abs(ica_maps) <
                 self.threshold / np.sqrt(ica_maps.shape[1])] = 0
        self.maps_ = ica_maps

        return self
