"""
PCA dimension reduction on multiple subjects.
This is a good initialization method for ICA.
"""
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.utils.extmath import randomized_svd

from .base import BaseDecomposition


class MultiPCA(BaseDecomposition):
    """Perform Multi Subject Principal Component Analysis.

    Perform a PCA on each subject, stack the results, and reduce them
    at group level. An optional Canonical Correlation Analysis can be
    performed at group level. This is a good initialization method for ICA.

    Parameters
    ----------
    n_components: int
        Number of components to extract. By default n_components=20.

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

    mask_strategy: {'background', 'epi' or 'template'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, 'epi' if they
        are raw EPI images, or you could use 'template' which will
        extract the gray matter part of your data by resampling the MNI152
        brain mask for your data's field of view.
        Depending on this value, the mask will be computed from
        masking.compute_background_mask, masking.compute_epi_mask or
        masking.compute_gray_matter_mask. Default is 'epi'.

    mask_args: dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    detrend : boolean, optional
        If detrend is True, the time-series will be detrended before
        components extraction.

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
        Indicate the level of verbosity. By default, nothing is printed.

    Attributes
    ----------
    `masker_` : instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_` : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `components_` : 2D numpy array (n_components x n-voxels)
        Array of masked extracted components. They can be unmasked thanks to
        the `masker_` attribute.

        Deprecated since version 0.4.1. Use `components_img_` instead.

    `components_img_` : 4D Nifti image
        4D image giving the extracted PCA components. Each 3D image is a
        component.

        New in version 0.4.1.

    `variance_` : numpy array (n_components,)
        The amount of variance explained by each of the selected components.

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
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1,
                 verbose=0
                 ):
        self.n_components = n_components
        self.do_cca = do_cca

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

    def _raw_fit(self, data):
        """Helper function that directly process unmasked data"""
        if self.do_cca:
            S = np.sqrt(np.sum(data ** 2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]
        components_, self.variance_, _ = self._cache(
            randomized_svd, func_memory_level=2)(
            data.T, n_components=self.n_components,
            transpose=True,
            random_state=self.random_state, n_iter=3)
        if self.do_cca:
            data *= S[:, np.newaxis]
        self.components_ = components_.T
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(
                components_.T)
        return components_
