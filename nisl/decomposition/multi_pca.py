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
from .._utils.class_inspect import get_params


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
    if n_components <= data.shape[0] / 4:
        U, S, _ = randomized_svd(data.T, n_components)
    else:
        U, S, _ = linalg.svd(data.T, full_matrices=False)
    U = U.T[:n_components].copy()
    S = S[:n_components]
    return U, S


class MultiPCA(NiftiMultiMasker, TransformerMixin):

    def __init__(self, mask=None, smoothing_fwhm=None,
             standardize=True, detrend=True,
             low_pass=None, high_pass=None, t_r=None,
             target_affine=None, target_shape=None,
             mask_connected=True, mask_opening=False,
             mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
             memory=Memory(cachedir=None), memory_level=0,
             n_jobs=1, verbose=0,
             # MultiPCA options
             do_cca=True, n_components=20
             ):
        super(MultiPCA, self).__init__(
            mask, smoothing_fwhm, standardize, detrend, low_pass, high_pass,
            t_r, target_affine, target_shape, mask_connected, mask_opening,
            mask_lower_cutoff, mask_upper_cutoff, memory, memory_level,
            n_jobs, verbose)
        self.do_cca = do_cca
        self.n_components = n_components
        self.parameters = get_params(NiftiMultiMasker, self)

    def fit(self, niimgs=None, y=None, confounds=None):
        """Compute the mask and the components """
        # First learn the mask
        NiftiMultiMasker.fit(self, niimgs)

        # XXX: we should warn the user that we enable these options if they are
        # not set

        self.standardize = True
        self.detrend = True
        self.parameters['detrend'] = True
        self.parameters['standardize'] = True

        # Now do the subject-level signal extraction (i.e. data-loading +
        # PCA)
        subject_pcas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                            delayed(session_pca)(niimg, self.mask_img_,
                                    self.parameters,
                                    n_components=self.n_components,
                                    memory=self.memory,
                                    ref_memory_level=self.memory_level,
                                    confounds=confounds,
                                    verbose=self.verbose
                            )
                            for niimg in niimgs)

        subject_pcas, subject_svd_vals = zip(*subject_pcas)

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
