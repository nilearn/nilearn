"""
PCA dimension reduction on multiple subjects
"""
import copy

from scipy import linalg
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.extmath import randomized_svd
from sklearn.externals.joblib import Memory

from ..io import NiftiMultiMasker, NiftiMapsMasker
from ..io.base_masker import filter_and_mask
from .._utils.class_inspect import get_params
from .._utils.cache_mixin import cache


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
    data, affine = cache(
        filter_and_mask, memory=memory, ref_memory_level=ref_memory_level,
        memory_level=2)(
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


class MultiPCA(TransformerMixin):
    """Perform Multi Subject Principal Component Analysis.

    Perform a PCA on each subject and stack the results. An optional Canical
    Correlation Analysis can also be performed.

    Parameters
    ----------
    mask: filename, NiImage or NiftiMultiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a NiftiMultiMasker with default
        parameters.

    n_components: int
        Number of components to extract

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.
    """

    def __init__(self, mask=None,
             memory=Memory(cachedir=None), memory_level=0,
             n_jobs=1, verbose=0,
             # MultiPCA options
             do_cca=True, n_components=20,
             smoothing_fwhm=None, target_affine=None
             ):
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.do_cca = do_cca
        self.n_components = n_components
        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine

    def fit(self, niimgs=None, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        # First, learn the mask
        if not isinstance(self.mask, NiftiMultiMasker):
            self.mask_ = NiftiMultiMasker(mask=self.mask,
                                         smoothing_fwhm=self.smoothing_fwhm,
                                         target_affine=self.target_affine)
        else:
            self.mask_ = copy.copy(self.mask)
            # XXX Change parameters of the masker for smoothing and
            # target_affine
        if self.mask_.mask is None:
            self.mask_.fit(niimgs)
        else:
            self.mask_.fit()

        # XXX: we should warn the user that we enable these options if they are
        # not set

        self.standardize = True
        self.detrend = True

        parameters = get_params(NiftiMultiMasker, self.mask_)
        parameters['detrend'] = True
        parameters['standardize'] = True

        # Now do the subject-level signal extraction (i.e. data-loading +
        # PCA)
        subject_pcas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                            delayed(session_pca)(niimg, self.mask_.mask_img_,
                                    parameters,
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
        # For the moment, store also the components_img
        self.components_img_ = self.mask_.inverse_transform(data)
        return self

    def transform(self, niimgs, confounds=None):
        """ Project the data into a reduced representation

        Parameters
        ----------
        niimgs: nifti like images
            Data to be projected

        confounds: CSV file path or 2D matrix
            This parameter is passed to nisl.signal.clean. Please see the
            related documentation for details
        """

        nifti_maps_masker = NiftiMapsMasker(
            self.components_img_, self.mask_.mask_img_,
            reasample_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 4D/ list of 4D data?
        if confounds is None:
            confounds = [None] * len(niimgs)
        return [nifti_maps_masker.transform(niimg, confounds=confound)
                for niimg, confound in zip(niimgs, confounds)]

    def inverse_transform(self, component_signals):
        """ Transform regions signals into voxel signals

        Parameters
        ----------
        component_signals: list of numpy array (n_samples x n_components)
            Component signals to tranform back into voxel signals
        """
        nifti_maps_masker = NiftiMapsMasker(
            self.components_img_, self.mask_.mask_img_,
            reasample_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 2D/ list of 2D data?
        return [nifti_maps_masker.inverse_transform(signal)
                for signal in component_signals]
