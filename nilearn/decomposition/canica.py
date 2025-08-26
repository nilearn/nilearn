"""Canonical Independent Component Analysis."""

import warnings as _warnings
from operator import itemgetter

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import scoreatpercentile
from sklearn.decomposition import fastica
from sklearn.utils import check_random_state

from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn.decomposition._multi_pca import _MultiPCA


@fill_doc
class CanICA(_MultiPCA):
    """Perform :term:`Canonical Independent Component Analysis<CanICA>`.

    See :footcite:t:`Varoquaux2010c` and :footcite:t:`Varoquaux2010d`.

    Parameters
    ----------
    mask : Niimg-like object, :obj:`~nilearn.maskers.MultiNiftiMasker` or \
           :obj:`~nilearn.surface.SurfaceImage` or \
           :obj:`~nilearn.maskers.SurfaceMasker` object, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given, for Nifti images,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters; for surface images, all the vertices will be used.

    n_components : :obj:`int`, default=20
        Number of components to extract.

    %(smoothing_fwhm)s
        Default=6mm.

    do_cca : :obj:`bool`, default=True
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    threshold : None, 'auto' or :obj:`float`, default='auto'
        If None, no thresholding is applied. If 'auto',
        then we apply a thresholding that will keep the n_voxels,
        more intense voxels across all the maps, n_voxels being the number
        of voxels in a brain volume. A float value indicates the
        ratio of voxels to keep (2. means that the maps will together
        have 2 x n_voxels non-zero voxels ). The float value
        must be bounded by [0. and n_components].

    n_init : :obj:`int`, default=10
        The number of times the fastICA algorithm is restarted

    %(random_state)s

    %(standardize)s

    %(standardize_confounds)s

    %(detrend)s

    %(low_pass)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(high_pass)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(t_r)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.


    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(mask_strategy)s

        Default='epi'.

        .. note::
            These strategies are only relevant for Nifti images and the
            parameter is ignored for SurfaceImage objects.

    mask_args : :obj:`dict`, optional
        If mask is None, these are additional parameters passed to
        :func:`nilearn.masking.compute_background_mask`,
        or :func:`nilearn.masking.compute_epi_mask`
        to fine-tune mask computation.
        Please see the related documentation for details.

    %(memory)s

    %(memory_level)s

    %(n_jobs)s

    %(verbose0)s

    %(base_decomposition_fit_attributes)s

    %(multi_pca_fit_attributes)s

    variance_ : numpy array (n_components,)
        The amount of variance explained
        by each of the selected components.

    References
    ----------
    .. footbibliography::

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
        memory=None,
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

        self.threshold = threshold
        self.n_init = n_init

    def _unmix_components(self, components):
        """Core function of CanICA than rotate components_ to maximize \
        independence.
        """
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
                    stacklevel=find_stack_level(),
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
        if (
            isinstance(self.threshold, float)
            and self.threshold > self.n_components
        ):
            raise ValueError(
                "Threshold must not be higher than number of maps. "
                f"Number of maps is {self.n_components} "
                f"and you provided threshold={self.threshold}."
            )
        components = _MultiPCA._raw_fit(self, data)

        self._unmix_components(components)
        return self
