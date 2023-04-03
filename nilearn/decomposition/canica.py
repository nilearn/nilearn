"""Canonical Independent Component Analysis."""

# Author: Alexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause

import warnings as _warnings
from operator import itemgetter

import numpy as np
from joblib import Memory, Parallel, delayed
from nilearn._utils import fill_doc
from scipy.stats import scoreatpercentile
from sklearn.decomposition import fastica
from sklearn.utils import check_random_state

from ._multi_pca import _MultiPCA


@fill_doc
class CanICA(_MultiPCA):
    """Perform Canonical Independent Component Analysis [1]_ [2]_.

    Parameters
    ----------
    mask : Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components : int, optional
        Number of components to extract. Default=20.
    %(smoothing_fwhm)s
        Default=6mm.

    do_cca : boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA. Default=True.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.
        Default=True.

    standardize_confounds : boolean, optional
        If standardize_confounds is True, the confounds are zscored:
        their mean is put to 0 and their variance to 1 in the time dimension.
        Default=True.

    detrend : boolean, optional
        If detrend is True, the time-series will be detrended before
        components extraction. Default=True.

    threshold : None, 'auto' or float, optional
        If None, no thresholding is applied. If 'auto',
        then we apply a thresholding that will keep the n_voxels,
        more intense voxels across all the maps, n_voxels being the number
        of voxels in a brain volume. A float value indicates the
        ratio of voxels to keep (2. means that the maps will together
        have 2 x n_voxels non-zero voxels ). The float value
        must be bounded by [0. and n_components].
        Default='auto'.

    n_init : int, optional
        The number of times the fastICA algorithm is restarted
        Default=10.

    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.

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

    %(mask_strategy)s

        .. note::
             Depending on this value, the mask will be computed from
             :func:`nilearn.masking.compute_background_mask`,
             :func:`nilearn.masking.compute_epi_mask`, or
             :func:`nilearn.masking.compute_brain_mask`.

        Default='epi'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory : instance of joblib.Memory or string, optional
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
        Default=Memory(location=None).

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching. Default=0.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on. Default=1.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed
        Default=0.

    Attributes
    ----------
    `components_` : 2D numpy array (n_components x n-voxels)
        Masked ICA components extracted from the input images.

        .. note::

            Use attribute `components_img_` rather than manually unmasking
            `components_` with `masker_` attribute.

    `components_img_` : 4D Nifti image
        4D image giving the extracted ICA components. Each 3D image is a
        component.

        .. versionadded:: 0.4.1

    `masker_` : instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_` : Niimg-like object
        See :ref:`extracting_data`.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    References
    ----------
    .. [1] G. Varoquaux et al. "A group model for stable multi-subject ICA on
       fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    .. [2] G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
       datasets", IEEE ISBI 2010, p. 1177

    """

    def __init__(
        self,
        mask=None,
        n_components=20,
        smoothing_fwhm=6,
        do_cca=True,
        threshold="auto",
        n_init=10,
        random_state=None,
        standardize=True,
        standardize_confounds=True,
        detrend=True,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="epi",
        mask_args=None,
        memory=Memory(location=None),
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        super().__init__(
            n_components=n_components,
            do_cca=do_cca,
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

        if isinstance(threshold, float) and threshold > n_components:
            raise ValueError(
                "Threshold must not be higher than number "
                "of maps. "
                f"Number of maps is {n_components} and you provided "
                f"threshold={threshold}"
            )
        self.threshold = threshold
        self.n_init = n_init

    def _unmix_components(self, components):
        """Core function of CanICA than rotate components_ to maximize \
        independence."""
        random_state = check_random_state(self.random_state)

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        # Note: fastICA is very unstable, hence we use 64bit on it
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._cache(fastica, func_memory_level=2))(
                components.astype(np.float64),
                whiten="arbitrary-variance",
                fun="cube",
                random_state=seed,
            )
            for seed in seeds
        )

        ica_maps_gen_ = (result[2].T for result in results)
        ica_maps_and_sparsities = (
            (ica_map, np.sum(np.abs(ica_map), axis=1).max())
            for ica_map in ica_maps_gen_
        )
        ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == "auto":
            ratio = 1.0
        elif self.threshold is not None:
            raise ValueError(
                "Threshold must be None, "
                f"'auto' or float. You provided {self.threshold}."
            )
        if ratio is not None:
            abs_ica_maps = np.abs(ica_maps)
            percentile = 100.0 - (100.0 / len(ica_maps)) * ratio
            if percentile <= 0:
                _warnings.warn(
                    "Nilearn's decomposition module "
                    "obtained a critical threshold "
                    f"(= {percentile} percentile).\n"
                    "No threshold will be applied. "
                    "Threshold should be decreased or "
                    "number of components should be adjusted.",
                    UserWarning,
                    stacklevel=4,
                )
            else:
                threshold = scoreatpercentile(abs_ica_maps, percentile)
                ica_maps[abs_ica_maps < threshold] = 0.0
        # We make sure that we keep the dtype of components
        self.components_ = ica_maps.astype(self.components_.dtype)

        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if component.max() < -component.min():
                component *= -1
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(
                self.components_
            )

    # Overriding _MultiPCA._raw_fit overrides _MultiPCA.fit behavior
    def _raw_fit(self, data):
        """Process unmasked data directly.

        Useful when called by another estimator that has already
        unmasked data.

        Parameters
        ----------
        data : ndarray or memmap
            Unmasked data to process

        """
        components = _MultiPCA._raw_fit(self, data)
        self._unmix_components(components)
        return self
