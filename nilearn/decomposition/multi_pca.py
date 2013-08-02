"""
PCA dimension reduction on multiple subjects
"""
import warnings
import itertools

from scipy import linalg
import numpy as np

import nibabel

from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.utils.extmath import randomized_svd

from ..input_data import MultiNiftiMasker, NiftiMapsMasker
from ..input_data.base_masker import filter_and_mask
from .._utils.class_inspect import get_params
from .._utils.cache_mixin import cache


def session_pca(niimgs, mask_img, parameters,
                n_components=20,
                confounds=None,
                ref_memory_level=0,
                memory=Memory(cachedir=None),
                verbose=0,
                copy=True):
    """Filter, mask and compute PCA on niimgs

    This is an helper function whose first call `base_masker.filter_and_mask`
    and then apply a PCA to reduce the number of time series.

    Parameters
    ----------
    niimgs: list of Niimg
        List of subject data

    mask_img: Niimage
        Mask to apply on the data

    parameters: dictionary
        Dictionary of parameters passed to `filter_and_mask`. Please see the
        documentation of the `NiftiMasker` for more informations.

    confounds: CSV file path or 2D matrix
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    n_components: integer, optional
        Number of components to be extracted by the PCA

    ref_memory_level: integer, optional
        Integer indicating the level of memorization. The higher, the more
        function calls are cached.

    memory: joblib.Memory
        Used to cache the function calls.

    verbose: integer, optional
        Indicate the level of verbosity

    copy: boolean, optional
        Whether or not data should be copied
    """

    data, affine = cache(
        filter_and_mask, memory=memory, ref_memory_level=ref_memory_level,
        memory_level=2,
        ignore=['verbose', 'memory', 'ref_memory_level', 'copy'])(
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


class MultiPCA(BaseEstimator, TransformerMixin):
    """Perform Multi Subject Principal Component Analysis.

    Perform a PCA on each subject and stack the results. An optional Canonical
    Correlation Analysis can also be performed.

    Parameters
    ----------
    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    mask: filename, NiImage or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

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

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ----------
    `masker_`: instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_`: Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `components_`: 2D numpy array (n_components x n-voxels)
        Array of masked extracted components. They can be unmasked thanks to
        the `masker_` attribute.
    """

    def __init__(self, n_components=20, smoothing_fwhm=None, mask=None,
                 do_cca=True, target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.do_cca = do_cca
        self.n_components = n_components
        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine
        self.target_shape = target_shape

    def fit(self, niimgs=None, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        # Hack to support single-subject data:
        if isinstance(niimgs, (basestring, nibabel.Nifti1Image)):
            niimgs = [niimgs]
            # This is a very incomplete hack, as it won't work right for
            # single-subject list of 3D filenames
        # First, learn the mask
        if not isinstance(self.mask, MultiNiftiMasker):
            self.masker_ = MultiNiftiMasker(mask=self.mask,
                                            smoothing_fwhm=self.smoothing_fwhm,
                                            target_affine=self.target_affine,
                                            target_shape=self.target_shape,
                                            low_pass=self.low_pass,
                                            high_pass=self.high_pass,
                                            t_r=self.t_r,
                                            memory=self.memory,
                                            memory_level=self.memory_level)
        else:
            try:
                self.masker_ = clone(self.mask)
            except TypeError as e:
                # Workaround for a joblib bug: in joblib 0.6, a Memory object
                # with cachedir = None cannot be cloned.
                masker_memory = self.mask.memory
                if masker_memory.cachedir is None:
                    self.mask.memory = None
                    self.masker_ = clone(self.mask)
                    self.mask.memory = masker_memory
                    self.masker_.memory = Memory(cachedir=None)
                else:
                    # The error was raised for another reason
                    raise e

            for param_name in ['target_affine', 'target_shape',
                               'smoothing_fwhm', 'low_pass', 'high_pass',
                               't_r', 'memory', 'memory_level']:
                if getattr(self.masker_, param_name) is not None:
                    warnings.warn('Parameter %s of the masker overriden'
                                  % param_name)
                setattr(self.masker_, param_name,
                        getattr(self, param_name))
        if self.masker_.mask is None:
            self.masker_.fit(niimgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        parameters = get_params(MultiNiftiMasker, self)
        parameters['detrend'] = True
        parameters['standardize'] = True

        # Now do the subject-level signal extraction (i.e. data-loading +
        # PCA)

        subject_pcas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(session_pca)(
                niimg,
                self.masker_.mask_img_,
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
                for subject_pca, subject_svd_val in \
                        zip(subject_pcas, subject_svd_vals):
                    subject_pca *= subject_svd_val[:, np.newaxis]
            data = np.empty((len(niimgs) * self.n_components,
                            subject_pcas[0].shape[1]),
                            dtype=subject_pcas[0].dtype)
            for index, subject_pca in enumerate(subject_pcas):
                if self.n_components > subject_pca.shape[0]:
                    raise ValueError('You asked for %i components.'
                                     'This is smaller than single-subject '
                                     'data size.' % self.n_components)
                data[index * self.n_components:
                     (index + 1) * self.n_components] = subject_pca
            data, variance, _ = randomized_svd(
                data.T, n_components=self.n_components)
            data = data.T
        else:
            data = subject_pcas[0]
        self.components_ = data
        return self

    def transform(self, niimgs, confounds=None):
        """ Project the data into a reduced representation

        Parameters
        ----------
        niimgs: nifti like images
            Data to be projected

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 4D/ list of 4D data?
        if confounds is None:
            confounds = itertools.repeat(None, len(niimgs))
        return [nifti_maps_masker.transform(niimg, confounds=confound)
                for niimg, confound in zip(niimgs, confounds)]

    def inverse_transform(self, component_signals):
        """ Transform regions signals into voxel signals

        Parameters
        ----------
        component_signals: list of numpy array (n_samples x n_components)
            Component signals to tranform back into voxel signals
        """
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 2D/ list of 2D data?
        return [nifti_maps_masker.inverse_transform(signal)
                for signal in component_signals]
