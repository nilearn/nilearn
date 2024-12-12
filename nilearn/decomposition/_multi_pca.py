"""PCA dimension reduction on multiple subjects.

This is a good initialization method for ICA.
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd

from nilearn._utils import fill_doc

from ._base import _BaseDecomposition


@fill_doc
class _MultiPCA(_BaseDecomposition):
    """Perform Multi Subject Principal Component Analysis.

    Perform a PCA on each subject, stack the results, and reduce them
    at group level. An optional Canonical Correlation Analysis can be
    performed at group level. This is a good initialization method for ICA.

    Parameters
    ----------
    n_components : int, default=20
        Number of components to extract.

    do_cca : boolean, default=True
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.
    %(smoothing_fwhm)s
    mask : Niimg-like object, instance of NiftiMasker
        or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.
    %(mask_strategy)s

        .. note::
             Depending on this value, the mask will be computed from
             :func:`nilearn.masking.compute_background_mask`,
             :func:`nilearn.masking.compute_epi_mask`, or
             :func:`nilearn.masking.compute_brain_mask`.

        Default='epi'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        :func:`nilearn.masking.compute_background_mask`,
        or :func:`nilearn.masking.compute_epi_mask`
        to fine-tune mask computation.
        Please see the related documentation for details.

    standardize : boolean, default=False
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    standardize_confounds : boolean, default=True
        If standardize_confounds is True, the confounds are z-scored:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend : boolean, default=False
        If detrend is True, the time-series will be detrended before
        components extraction.

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    memory : instance of joblib.Memory or string, default=None
        Used to cache the masking process.
        By default, no caching is done.
        If a string is given, it is the path to the caching directory.
        If ``None`` is passed will default to ``Memory(location=None)``.

    memory_level : integer, default=0
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs : integer, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose : integer, default=0
        Indicate the level of verbosity. By default, nothing is printed.

    Attributes
    ----------
    masker_ : instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in ``mask`` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of ``mask`` and other NiftiMasker related parameters as initialization.

    mask_img_ : Niimg-like object
        See :ref:`extracting_data`.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    components_ : 2D numpy array (n_components x n-voxels)
        Array of masked extracted components.

        .. note::

            Use attribute ``components_img_`` rather than manually unmasking
            ``components_`` with ``masker_`` attribute.

    components_img_ : 4D Nifti image
        4D image giving the extracted PCA components. Each 3D image is a
        component.

        .. versionadded:: 0.4.1

    variance_ : numpy array (n_components,)
        The amount of variance explained by each of the selected components.

    """

    def __init__(
        self,
        n_components=20,
        mask=None,
        smoothing_fwhm=None,
        do_cca=True,
        random_state=None,
        standardize=False,
        standardize_confounds=True,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="epi",
        mask_args=None,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        self.n_components = n_components
        self.do_cca = do_cca

        _BaseDecomposition.__init__(
            self,
            n_components=n_components,
            random_state=random_state,
            mask=mask,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            standardize_confounds=standardize_confounds,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            target_affine=target_affine,
            target_shape=target_shape,
            mask_strategy=mask_strategy,
            mask_args=mask_args,
            memory=memory,
            memory_level=memory_level,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _raw_fit(self, data):
        """Process unmasked data directly."""
        if self.do_cca:
            S = np.sqrt(np.sum(data**2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]
        components_, self.variance_, _ = self._cache(
            randomized_svd, func_memory_level=2
        )(
            data.T,
            n_components=self.n_components,
            transpose=True,
            random_state=self.random_state,
            n_iter=3,
        )
        if self.do_cca:
            data *= S[:, np.newaxis]
        self.components_ = components_.T
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(
                components_.T
            )
        return components_
