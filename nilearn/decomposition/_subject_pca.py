"""
PCA dimension reduction on single subjects
"""
import warnings

import numpy as np
from scipy import linalg
import nibabel
from sklearn.base import BaseEstimator, clone
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_random_state

from .._utils.class_inspect import get_params
from ..input_data import NiftiMasker, MultiNiftiMasker
from ..input_data.base_masker import filter_and_mask
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.compat import _basestring

def session_pca(imgs, mask_img, parameters,
                n_components=20,
                confounds=None,
                memory_level=0,
                memory=Memory(cachedir=None),
                verbose=0,
                copy=True,
                random_state=0):
    """Filter, mask and compute PCA on Niimg-like objects

    This is an helper function whose first call `base_masker.filter_and_mask`
    and then apply a PCA to reduce the number of time series.

    Parameters
    ----------
    imgs: list of Niimg-like objects
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        List of subject data

    mask_img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Mask to apply on the data

    parameters: dictionary
        Dictionary of parameters passed to `filter_and_mask`. Please see the
        documentation of the `NiftiMasker` for more informations.

    confounds: CSV file path or 2D matrix
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    n_components: integer, optional
        Number of components to be extracted by the PCA

    memory_level: integer, optional
        Integer indicating the level of memorization. The higher, the more
        function calls are cached.

    memory: joblib.Memory
        Used to cache the function calls.

    verbose: integer, optional
        Indicate the level of verbosity (0 means no messages).

    copy: boolean, optional
        Whether or not data should be copied

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    random_state:
    """

    data, affine = cache(
        filter_and_mask, memory,
        func_memory_level=2, memory_level=memory_level,
        ignore=['verbose', 'memory', 'memory_level', 'copy'])(
            imgs, mask_img, parameters,
            memory_level=memory_level,
            memory=memory,
            verbose=verbose,
            confounds=confounds,
            copy=copy)
    if n_components <= data.shape[0] // 4:
        U, S, _ = cache(randomized_svd, memory, memory_level=memory_level,
                        func_memory_level=2)(
            data.T, n_components, random_state=random_state)
    else:
        U, S, _ = cache(linalg.svd, memory, memory_level=memory_level,
                        func_memory_level=2)(
            data.T, full_matrices=False)
    U = U.T[:n_components].copy()
    S = S[:n_components]
    return U, S


class _SubjectPCA(BaseEstimator, CacheMixin):
    """Perform Single Subject Principal Component Analysis. Perform a PCA on each subject and stack the results in a list

    Parameters
    ----------
    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    mask: Niimg-like object, instance of NiftiMasker or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    whiten: boolean, optional
        Indicate whether components PCA must be whitened

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

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ----------
    `masker_`: instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_`: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `components_`: list of 2D numpy array (n_components x n-voxels)
        Arrays of masked extracted components, for each subject. They can be unmasked thanks to
        the `masker_` attribute.

    """
    def __init__(self, n_components=20, smoothing_fwhm=None, mask=None,
                 standardize=True, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None,
                 t_r=None, memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 whiten=False,
                 random_state=None
                 ):
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.n_components = n_components
        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.standardize = standardize
        self.random_state = random_state

        self.whiten = whiten

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        random_state = check_random_state(self.random_state)
        # Hack to support single-subject data:
        if isinstance(imgs, (_basestring, nibabel.Nifti1Image)):
            imgs = [imgs]
            # This is a very incomplete hack, as it won't work right for
            # single-subject list of 3D filenames
        if len(imgs) == 0:
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError('Need one or more Niimg-like objects as input, '
                             'an empty list was given.')
        if confounds is None:
            confounds = [None] * len(imgs)  # itertools.repeat(None, len(imgs))

        # First, learn the mask
        if not isinstance(self.mask, (NiftiMasker, MultiNiftiMasker)):
            self.masker_ = MultiNiftiMasker(mask_img=self.mask,
                                            smoothing_fwhm=self.smoothing_fwhm,
                                            target_affine=self.target_affine,
                                            target_shape=self.target_shape,
                                            standardize=self.standardize,
                                            low_pass=self.low_pass,
                                            high_pass=self.high_pass,
                                            mask_strategy='epi',
                                            t_r=self.t_r,
                                            memory=self.memory,
                                            memory_level=self.memory_level,
                                            n_jobs=self.n_jobs,
                                            verbose=max(0, self.verbose - 1))
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
                our_param = getattr(self, param_name)
                if our_param is None:
                    # Default value
                    continue
                if getattr(self.masker_, param_name) is not None:
                    warnings.warn('Parameter %s of the masker overriden'
                                  % param_name)
                setattr(self.masker_, param_name, our_param)

        # Masker warns if it has a mask_img and is passed
        # imgs to fit().  Avoid the warning by being careful
        # when calling fit.
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        # Now do the subject-level signal extraction (i.e. data-loading +
        # PCA)
        if self.verbose:
            print("[SinglePCA] Learning subject level PCAs")
        subject_pcas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._cache(session_pca, func_memory_level=1))(
                img,
                self.masker_.mask_img_,
                self._get_filter_and_mask_parameters(),
                n_components=self.n_components,
                memory=self.memory,
                memory_level=self.memory_level,
                confounds=confound,
                verbose=self.verbose,
                random_state=random_state,
            )
            for img, confound in zip(imgs, confounds))
        subject_pcas, subject_svd_vals = zip(*subject_pcas)
        if not self.whiten:
            for subject_pca, subject_svd_val in \
                    zip(subject_pcas, subject_svd_vals):
                subject_pca *= subject_svd_val[:, np.newaxis]

        self.components_ = list(subject_pcas)
        self.variance_ = list(subject_svd_vals)

        if len(subject_pcas) == 1:
            self.components_ = self.components_[0]
        else:
            self.components_ = np.empty((len(imgs) * self.n_components,
                            subject_pcas[0].shape[1]),
                            dtype=subject_pcas[0].dtype)
            for index, subject_pca in enumerate(subject_pcas):
                if self.n_components > subject_pca.shape[0]:
                    raise ValueError('You asked for %i components. '
                                     'This is larger than the single-subject '
                                     'data size (%d).' % (self.n_components,
                                                          subject_pca.shape[0]))
                self.components_[index * self.n_components:
                                 (index + 1) * self.n_components] = subject_pca

        return self

    def _get_filter_and_mask_parameters(self):
        parameters = get_params(MultiNiftiMasker, self)
        # Remove non specific and redudent parameters
        for param_name in ['memory', 'memory_level', 'confounds',
                           'verbose', 'n_jobs']:
            parameters.pop(param_name, None)

        parameters['detrend'] = True
        return parameters

