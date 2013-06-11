"""
pca dimension reduction on multiple subjects
"""
from scipy import linalg
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.extmath import randomized_svd
from sklearn.externals.joblib import Memory

from ..io import NiftiMultiMasker, NiftiMapsMasker
from ..io.base_masker import filter_and_mask


def session_pca(niimgs, mask_img, parameters,
                n_components=20,
                ref_memory_level=0,
                memory=Memory(cachedir=None),
                verbose=0,
                confounds=None,
                copy=True):
    # XXX: we should warn the user that we enable these options if they are
    # not set
    parameters['detrend'] = True
    parameters['standardize'] = True
    data, affine = filter_and_mask(
                    niimgs, mask_img, parameters,
                    ref_memory_level=ref_memory_level,
                    memory=memory,
                    verbose=verbose,
                    confounds=confounds,
                    copy=copy)
    U, S, _ = linalg.svd(data.T, full_matrices=False)
    U = U.T[:n_components].copy()
    S = S[:n_components]
    return U, S


class MultiPCA(NiftiMultiMasker, TransformerMixin):

    do_cca = True
    n_components = 20

    def fit(self, niimgs=None, y=None):
        """Compute the mask and the components """
        # First learn the mask
        NiftiMultiMasker.fit(self, niimgs)

        # Now do the subject-level signal extraction (i.e. data-loading +
        # PCA)
        # XXX change the name of this variable
        subject_pca = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                            delayed(session_pca)(niimg, self.mask_img_,
                                    self.parameters,
                                    n_components=self.n_components,
                                    # XXX: need to give all the filtering
                                    # other options
                            )
                            for niimg in niimgs)

        subject_pcas, subject_svd_vals = zip(*subject_pca)

        if len(niimgs) > 1:
            if not self.do_cca:
                for subject_pca, subject_svd_val in zip(
                                    subject_pcas, subject_svd_vals):
                    subject_pca *= subject_svd_val
            data = np.empty(
                    (len(niimgs) * self.n_components,
                    subject_pcas[0].shape[1]), dtype=subject_pcas[0].dtype)
            for index, subject_pca in enumerate(subject_pcas):
                data[index * self.n_components:
                            (index + 1) * self.n_components] = subject_pca
            data, variance, _ = randomized_svd(data.T,
                                    n_components=self.n_components)
            data = data.T
        else:
            data = subject_pcas[0]
        self.components_ = data
        return self

    def transform(self, niimgs):
        """ Project the data into a reduced representation
        """
        valid_params = NiftiMapsMasker._get_param_names()
        params = dict((name, param) for name, param in self.get_params()
                      if name in valid_params)
        params['mask_img'] = self.mask_img_
        params['maps_img'] = NiftiMapsMasker.inverse_transform(self,
                                    self.components_)
        nifti_maps_masker = NiftiMapsMasker(**params)
        # XXX: dealing properly with 4D/ list of 4D data?
        return [nifti_maps_masker.transform(niimg)
                for niimg in niimgs]
