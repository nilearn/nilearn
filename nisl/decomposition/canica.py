"""
CanICA
"""

# Author: Alexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause
import distutils

import numpy as np

import sklearn
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state

from .multi_pca import MultiPCA
from .._utils.cache_mixin import CacheMixin


class CanICA(MultiPCA, CacheMixin):
    """Perform Canonical Independent Component Analysis.

    Parameters
    ----------
    data: array-like, shape = [[n_samples, n_features], ...]
        Training vector, where n_samples is the number of samples,
        n_features is the number of features. There is one vector per
        subject.

    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    References
    ----------
    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177
    """

    def __init__(self, mask=None, n_components=20,
                 smoothing_fwhm=6, do_cca=True,
                 threshold='auto', random_state=0,
                 target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 # Common options
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
             ):
        super(CanICA, self).__init__(
            mask=mask, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose, do_cca=do_cca,
            n_components=n_components, smoothing_fwhm=smoothing_fwhm,
            target_affine=target_affine, target_shape=target_shape)
        self.threshold = threshold
        self.random_state = random_state
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

    def fit(self, niimgs, y=None, confounds=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nisl.signal.clean. Please see the
            related documentation for details
        """
        MultiPCA.fit(self, niimgs, y=y, confounds=confounds)
        random_state = check_random_state(self.random_state)

        if (distutils.version.LooseVersion(sklearn.__version__).version
                > [0, 12]):
            # random_state in fastica was added in 0.13
            ica_maps = self._cache(fastica, memory_level=6)(
                                self.components_.T,
                                whiten=False,
                                fun='cube',
                                random_state=random_state)[2]
        else:
            ica_maps = self.cache(fastica, memory_level=6)(
                                self.components_.T, whiten=False,
                                            fun='cube')[2]
        ica_maps = ica_maps.T

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        if self.threshold is not None:
            raveled = np.abs(ica_maps).ravel()
            argsort = np.argsort(raveled)
            n_voxels = ica_maps[0].size
            threshold = raveled[argsort[- ratio * n_voxels]]
            ica_maps[np.abs(ica_maps) < threshold] = 0.

        self.components_ = ica_maps
        # For the moment, store also the components_img
        self.components_img_ = \
            self.mask_.inverse_transform(ica_maps)

        return self
