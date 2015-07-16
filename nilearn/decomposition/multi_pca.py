"""
PCA dimension reduction on multiple subjects
"""
import itertools

import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.utils.extmath import randomized_svd
from sklearn.base import TransformerMixin

from ..input_data import NiftiMapsMasker
from .._utils.cache_mixin import CacheMixin
from .._utils import as_ndarray
from .single_pca import SinglePCA


class MultiPCA(SinglePCA, TransformerMixin, CacheMixin):
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

    mask: Niimg-like object, instance of NiftiMasker or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

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

    keep_data_mem: boolean,
        Keep unmasked data in memory (useful to reuse unmasked data from super classes)

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

    `components_`: 2D numpy array (n_components x n-voxels)
        Array of masked extracted components. They can be unmasked thanks to
        the `masker_` attribute.
    """

    def __init__(self, n_components=20, smoothing_fwhm=None, mask=None,
                 do_cca=True, standardize=True, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None,
                 t_r=None, memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):

        SinglePCA.__init__(self, n_components=n_components,
                           smoothing_fwhm=smoothing_fwhm,
                           mask=mask,
                           standardize=standardize, target_affine=target_affine,
                           target_shape=target_shape,
                           low_pass=low_pass, high_pass=high_pass,
                           t_r=t_r, memory=memory, memory_level=memory_level,
                           n_jobs=n_jobs, verbose=verbose, do_cca=do_cca)

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        SinglePCA.fit(self, imgs, confounds=confounds)

        subject_pcas = self.components_list_
        subject_svd_vals = self.variance_list_

        if len(subject_pcas) > 1:
            data = np.empty((len(imgs) * self.n_components,
                            subject_pcas[0].shape[1]),
                            dtype=subject_pcas[0].dtype)
            for index, subject_pca in enumerate(subject_pcas):
                if self.n_components > subject_pca.shape[0]:
                    raise ValueError('You asked for %i components. '
                                     'This is larger than the single-subject '
                                     'data size (%d).' % (self.n_components,
                                                          subject_pca.shape[0]))
                data[index * self.n_components:
                     (index + 1) * self.n_components] = subject_pca
            data, variance, _ = self._cache(
                randomized_svd, func_memory_level=3)(
                    data.T, n_components=self.n_components)
            # as_ndarray is to get rid of memmapping
            data = as_ndarray(data.T)
        else:
            data = subject_pcas[0]
            variance = subject_svd_vals[0]

        self.components_ = data
        self.variance_ = variance

        return self

    def transform(self, imgs, confounds=None):
        """ Project the data into a reduced representation

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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
            confounds = itertools.repeat(None, len(imgs))
        return [nifti_maps_masker.transform(img, confounds=confound)
                for img, confound in zip(imgs, confounds)]

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
