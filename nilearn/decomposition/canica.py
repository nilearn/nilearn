"""
CanICA
"""

# Author: Alexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause

from operator import itemgetter

import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory, delayed, Parallel
from sklearn.utils import check_random_state

from .multi_pca import MultiPCA


class CanICA(MultiPCA):
    """Perform Canonical Independent Component Analysis.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    threshold: None, 'auto' or float
        If None, no thresholding is applied. If 'auto',
        then we apply a thresholding that will keep the n_voxels,
        more intense voxels across all the maps, n_voxels being the number
        of voxels in a brain volume. A float value indicates the
        ratio of voxels to keep (2. means that the maps will together
        have 2 x n_voxels non-zero voxels ). The float value
        must be bounded by [0. and n_components].

    n_init: int, optional
        The number of times the fastICA algorithm is restarted

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

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
        Indicate the level of verbosity. By default, nothing is printed

    References
    ----------
    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

    """

    def __init__(self, mask=None, n_components=20, smoothing_fwhm=6,
                 do_cca=True,
                 threshold='auto',
                 n_init=10,
                 random_state=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0
                 ):

        super(CanICA, self).__init__(
            n_components=n_components,
            do_cca=do_cca,
            random_state=random_state,
            # feature_compression=feature_compression,
            mask=mask, smoothing_fwhm=smoothing_fwhm,
            standardize=standardize, detrend=detrend,
            low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, mask_args=mask_args,
            memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

        if isinstance(threshold, float) and threshold > n_components:
            raise ValueError("Threshold must not be higher than number "
                             "of maps. "
                             "Number of maps is %s and you provided "
                             "threshold=%s" %
                             (str(n_components), str(threshold)))
        self.threshold = threshold
        self.n_init = n_init

    def _unmix_components(self):
        """Core function of CanICA than rotate components_ to maximize
        independance"""
        random_state = check_random_state(self.random_state)

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._cache(fastica, func_memory_level=2))
            (self.components_.T, whiten=True, fun='cube',
             random_state=seed)
            for seed in seeds)

        ica_maps_gen_ = (result[2].T for result in results)
        ica_maps_and_sparsities = ((ica_map,
                                    np.sum(np.abs(ica_map), axis=1).max())
                                   for ica_map in ica_maps_gen_)
        ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        elif self.threshold is not None:
            raise ValueError("Threshold must be None, "
                             "'auto' or float. You provided %s." %
                             str(self.threshold))
        if ratio is not None:
            abs_ica_maps = np.abs(ica_maps)
            threshold = scoreatpercentile(
                abs_ica_maps,
                100. - (100. / len(ica_maps)) * ratio)
            ica_maps[abs_ica_maps < threshold] = 0.
        self.components_ = ica_maps

        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if component.max() < -component.min():
                component *= -1

    # Overriding MultiPCA._raw_fit overrides MultiPCA.fit behavior
    def _raw_fit(self, data):
        """Helper function that directly process unmasked data.

        Useful when called by another estimator that has already
        unmasked data.

        Parameters
        ----------
        data: ndarray or memmap
            Unmasked data to process

        """
        MultiPCA._raw_fit(self, data)
        self._unmix_components()
        return self
