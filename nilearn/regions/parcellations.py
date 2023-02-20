"""Parcellation tools such as KMeans or Ward for fMRI images
"""

import warnings
import numpy as np

from sklearn.base import clone
from sklearn.feature_extraction import image
from joblib import Memory, delayed, Parallel

from .rena_clustering import ReNA
from .hierarchical_kmeans_clustering import HierarchicalKMeans
from ..decomposition._multi_pca import _MultiPCA
from nilearn.maskers import NiftiLabelsMasker
from .._utils.niimg import _safe_get_data
from .._utils.niimg_conversions import _iter_check_niimg
from .._utils import fill_doc
from .._utils import stringify_path


def _estimator_fit(data, estimator, method=None):
    """Estimator to fit on the data matrix

    Parameters
    ----------
    data : numpy array
        Data matrix.

    estimator : instance of estimator from sklearn
        MiniBatchKMeans or AgglomerativeClustering.

    method: str,
    {'kmeans', 'ward', 'complete', 'average', 'rena', 'hierarchical_kmeans'},
    optional

        A method to choose between for brain parcellations.

    Returns
    -------
    labels_ : numpy.ndarray
        labels_ estimated from estimator.

    """
    if method == 'rena':
        rena = ReNA(mask_img=estimator.mask_img,
                    n_clusters=estimator.n_clusters,
                    scaling=estimator.scaling,
                    n_iter=estimator.n_iter,
                    threshold=estimator.threshold,
                    memory=estimator.memory,
                    memory_level=estimator.memory_level,
                    verbose=estimator.verbose)
        rena.fit(data)
        labels_ = rena.labels_

    else:
        estimator = clone(estimator)
        estimator.fit(data.T)
        labels_ = estimator.labels_

    return labels_


def _check_parameters_transform(imgs, confounds):
    """A helper function to check the parameters and prepare for processing
    as a list.

    """
    imgs = stringify_path(imgs)
    confounds = stringify_path(confounds)
    if not isinstance(imgs, (list, tuple)) or \
            isinstance(imgs, str):
        imgs = [imgs, ]
        single_subject = True
    elif isinstance(imgs, (list, tuple)) and len(imgs) == 1:
        single_subject = True
    else:
        single_subject = False

    if confounds is None and isinstance(imgs, (list, tuple)):
        confounds = [None] * len(imgs)

    if confounds is not None:
        if not isinstance(confounds, (list, tuple)) or \
                isinstance(confounds, str):
            confounds = [confounds, ]

    if len(confounds) != len(imgs):
        raise ValueError("Number of confounds given does not match with "
                         "the given number of images.")
    return imgs, confounds, single_subject


def _labels_masker_extraction(img, masker, confound):
    """Helper function for parallelizing NiftiLabelsMasker extractor
    on list of Nifti images.

    Parameters
    ----------
    img : 4D Nifti image like object
        Image to process.

    masker : instance of NiftiLabelsMasker
        Used for extracting signals with fit_transform.

    confound : csv file, numpy ndarray or pandas DataFrame
        Confound used for signal cleaning while extraction.
        Passed to signal.clean.

    Returns
    -------
    signals : numpy array
        Signals extracted on given img.

    """
    masker = clone(masker)
    signals = masker.fit_transform(img, confounds=confound)
    return signals


@fill_doc
class Parcellations(_MultiPCA):
    """Learn :term:`parcellations<parcellation>`
    on :term:`fMRI` images.

    Five different types of clustering methods can be used:
    kmeans, ward, complete, average and rena.
    kmeans will call MiniBatchKMeans whereas
    ward, complete, average are used within in Agglomerative Clustering and
    rena will call ReNA.
    kmeans, ward, complete, average are leveraged from scikit-learn.
    rena is built into nilearn.

    .. versionadded:: 0.4.1

    Parameters
    ----------
    method: :obj:`str`, {'kmeans', 'ward', 'complete', 'average', 'rena',
        'hierarchical_kmeans'}
        A method to choose between for brain parcellations.
        For a small number of parcels, kmeans is usually advisable.
        For a large number of parcellations (several hundreds, or thousands),
        ward and rena are the best options. Ward will give higher quality
        parcels, but with increased computation time. ReNA is most useful as a
        fast data-reduction step, typically dividing the signal size by ten.
    %(n_parcels)s
    %(random_state)s
        Default=0.

    mask : Niimg-like object or :class:`nilearn.maskers.NiftiMasker`,\
 :class:`nilearn.maskers.MultiNiftiMasker`, optional
        Mask/Masker used for masking the data.
        If mask image if provided, it will be used in the MultiNiftiMasker.
        If an instance of MultiNiftiMasker is provided, then this instance
        parameters will be used in masking the data by overriding the default
        masker parameters.
        If None, mask will be automatically computed by a MultiNiftiMasker
        with default parameters.
    %(smoothing_fwhm)s
        Default=4.0.
    %(standardize_false)s
    %(detrend)s

        .. note::
            This parameter is passed to :func:`nilearn.signal.clean`.
            Please see the related documentation for details.

        Default=False.
    %(low_pass)s

        .. note::
            This parameter is passed to :func:`nilearn.signal.clean`.
            Please see the related documentation for details.

    %(high_pass)s

        .. note::
            This parameter is passed to :func:`nilearn.signal.clean`.
            Please see the related documentation for details.

    %(t_r)s

        .. note::
            This parameter is passed to :func:`nilearn.signal.clean`.
            Please see the related documentation for details.

    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.
            Please see the related documentation for details.

        .. note::
            The given affine will be considered as same for all
            given list of images.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.
            Please see the related documentation for details.

    %(mask_strategy)s

        .. note::
             Depending on this value, the mask will be computed from
             :func:`nilearn.masking.compute_background_mask`,
             :func:`nilearn.masking.compute_epi_mask`, or
             :func:`nilearn.masking.compute_brain_mask`.

        Default='epi'.

    mask_args : :obj:`dict`, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    scaling : :obj:`bool`, optional
        Used only when the method selected is 'rena'. If scaling is True, each
        cluster is scaled by the square root of its size, preserving the
        l2-norm of the image. Default=False.

    n_iter : :obj:`int`, optional
        Used only when the method selected is 'rena'. Number of iterations of
        the recursive neighbor agglomeration. Default=10.
    %(memory)s
    %(memory_level)s
    %(n_jobs)s
    %(verbose0)s

    Attributes
    ----------
    `labels_img_` : :class:`nibabel.nifti1.Nifti1Image`
        Labels image to each parcellation learned on fmri images.

    `masker_` : :class:`nilearn.maskers.NiftiMasker` or\
 :class:`nilearn.maskers.MultiNiftiMasker`
        The masker used to mask the data.

    `connectivity_` : :class:`numpy.ndarray`
        Voxel-to-voxel connectivity matrix computed from a mask.
        Note that this attribute is only seen if selected methods are
        Agglomerative Clustering type, 'ward', 'complete', 'average'.

    Notes
    -----
    * Transforming list of Nifti images to data matrix takes few steps.
      Reducing the data dimensionality using randomized SVD, build brain
      parcellations using KMeans or various Agglomerative methods.

    * This object uses spatially-constrained AgglomerativeClustering for
      method='ward' or 'complete' or 'average' and spatially-constrained
      ReNA clustering for method='rena'. Spatial connectivity matrix
      (voxel-to-voxel) is built-in object which means no need of explicitly
      giving the matrix.

    """
    VALID_METHODS = ['kmeans', 'ward', 'complete',
                     'average', 'rena', 'hierarchical_kmeans']

    def __init__(self, method, n_parcels=50,
                 random_state=0, mask=None, smoothing_fwhm=4.,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 scaling=False, n_iter=10,
                 memory=Memory(location=None),
                 memory_level=0, n_jobs=1, verbose=1):

        self.method = method
        self.n_parcels = n_parcels
        self.scaling = scaling
        self.n_iter = n_iter

        _MultiPCA.__init__(self, n_components=200,
                           random_state=random_state,
                           mask=mask, memory=memory,
                           smoothing_fwhm=smoothing_fwhm,
                           standardize=standardize, detrend=detrend,
                           low_pass=low_pass, high_pass=high_pass,
                           t_r=t_r, target_affine=target_affine,
                           target_shape=target_shape,
                           mask_strategy=mask_strategy,
                           mask_args=mask_args,
                           memory_level=memory_level,
                           n_jobs=n_jobs,
                           verbose=verbose)

    def _raw_fit(self, data):
        """Fits the parcellation method on this reduced data.

        Data are coming from a base decomposition estimator which computes
        the mask and reduces the dimensionality of images using
        randomized_svd.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            Shape (n_samples, n_features)

        Returns
        -------
        labels : :class:`numpy.ndarray`
            Labels to each cluster in the brain.

        connectivity : :class:`numpy.ndarray`
            Voxel-to-voxel connectivity matrix computed from a mask.
            Note that, this attribute is returned only for selected methods
            such as 'ward', 'complete', 'average'.

        """
        valid_methods = self.VALID_METHODS
        if self.method is None:
            raise ValueError("Parcellation method is specified as None. "
                             "Please select one of the method in "
                             "{0}".format(valid_methods))
        if self.method is not None and self.method not in valid_methods:
            raise ValueError("The method you have selected is not implemented "
                             "'{0}'. Valid methods are in {1}"
                             .format(self.method, valid_methods))

        # we delay importing Ward or AgglomerativeClustering and same
        # time import plotting module before that.

        # Because sklearn.cluster imports scipy hierarchy and hierarchy imports
        # matplotlib. So, we force import matplotlib first using our
        # plotting to avoid backend display error with matplotlib
        # happening in Travis
        try:
            from nilearn import plotting
        except Exception:
            pass

        components = _MultiPCA._raw_fit(self, data)

        mask_img_ = self.masker_.mask_img_
        if self.verbose:
            print("[{0}] computing {1}".format(self.__class__.__name__,
                                               self.method))

        if self.method == 'kmeans':
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=self.n_parcels,
                                     init='k-means++',
                                     random_state=self.random_state,
                                     verbose=max(0, self.verbose - 1))
            labels = self._cache(_estimator_fit,
                                 func_memory_level=1)(components.T, kmeans)
        elif self.method == 'hierarchical_kmeans':
            hkmeans = HierarchicalKMeans(self.n_parcels, init="k-means++",
                                         batch_size=1000, n_init=10,
                                         max_no_improvement=10,
                                         random_state=self.random_state,
                                         verbose=max(0, self.verbose - 1))
            # data ou data.T
            labels = self._cache(_estimator_fit,
                                 func_memory_level=1)(components.T, hkmeans)

        elif self.method == 'rena':
            rena = ReNA(mask_img_, n_clusters=self.n_parcels,
                        scaling=self.scaling, n_iter=self.n_iter,
                        memory=self.memory, memory_level=self.memory_level,
                        verbose=max(0, self.verbose - 1))
            method = 'rena'
            labels = \
                self._cache(_estimator_fit, func_memory_level=1)(components.T,
                                                                 rena, method)

        else:
            mask_ = _safe_get_data(mask_img_).astype(bool)
            shape = mask_.shape
            connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                               n_z=shape[2], mask=mask_)

            from sklearn.cluster import AgglomerativeClustering

            agglomerative = AgglomerativeClustering(
                n_clusters=self.n_parcels, connectivity=connectivity,
                linkage=self.method, memory=self.memory)

            labels = self._cache(_estimator_fit,
                                 func_memory_level=1)(components.T,
                                                      agglomerative)

            self.connectivity_ = connectivity
        # Avoid 0 label
        labels = labels + 1
        unique_labels = np.unique(labels)

        # Check that appropriate number of labels were created
        if len(unique_labels) != self.n_parcels:
            n_parcels_warning = ('The number of generated labels does not '
                                 'match the requested number of parcels.')
            warnings.warn(message=n_parcels_warning, category=UserWarning,
                          stacklevel=3)
        self.labels_img_ = self.masker_.inverse_transform(labels)

        return self

    def _check_fitted(self):
        """Helper function to check whether fit is called or not.
        """
        if not hasattr(self, 'labels_img_'):
            raise ValueError("Object has no labels_img_ attribute. "
                             "Ensure that fit() is called before transform.")

    @fill_doc
    def transform(self, imgs, confounds=None):
        """Extract signals from :term:`parcellations<parcellation>` learned
        on :term:`fMRI` images.

        Parameters
        ----------
        %(imgs)s
            Images to process.

        confounds : :obj:`list` of CSV files, arrays-like,\
 or :class:`pandas.DataFrame`, optional
            Each file or numpy array in a list should have shape
            (number of scans, number of confounds)
            Must be of same length as imgs.

            .. note::
                This parameter is passed to :func:`nilearn.signal.clean`.
                Please see the related documentation for details.

        Returns
        -------
        region_signals : :obj:`list` of or 2D :class:`numpy.ndarray`
            Signals extracted for each label for each image.
            Example, for single image shape will be
            (number of scans, number of labels)

        """
        self._check_fitted()
        imgs, confounds, single_subject = _check_parameters_transform(
            imgs, confounds)
        # Requires for special cases like extracting signals on list of
        # 3D images
        imgs_list = _iter_check_niimg(imgs, atleast_4d=True)

        masker = NiftiLabelsMasker(self.labels_img_,
                                   mask_img=self.masker_.mask_img_,
                                   smoothing_fwhm=self.smoothing_fwhm,
                                   standardize=self.standardize,
                                   detrend=self.detrend,
                                   low_pass=self.low_pass,
                                   high_pass=self.high_pass, t_r=self.t_r,
                                   resampling_target='data',
                                   memory=self.memory,
                                   memory_level=self.memory_level,
                                   verbose=self.verbose)

        region_signals = Parallel(n_jobs=self.n_jobs)(
            delayed(self._cache(_labels_masker_extraction,
                                func_memory_level=2))
            (img, masker, confound)
            for img, confound in zip(imgs_list, confounds))

        if single_subject:
            return region_signals[0]
        else:
            return region_signals

    @fill_doc
    def fit_transform(self, imgs, confounds=None):
        """Fit the images to :term:`parcellations<parcellation>` and
        then transform them.

        Parameters
        ----------
        %(imgs)s
            Images for process for fit as well for transform to signals.

        confounds : :obj:`list` of CSV files, arrays-like or\
 :class:`pandas.DataFrame`, optional
            Each file or numpy array in a list should have shape
            (number of scans, number of confounds).
            Given confounds should have same length as images if
            given as a list.

            .. note::
                This parameter is passed to :func:`nilearn.signal.clean`.
                Please see the related documentation for details.

            .. note::
                Confounds will be used for cleaning signals before
                learning parcellations.

        Returns
        -------
        region_signals : :obj:`list` of or 2D :class:`numpy.ndarray`
            Signals extracted for each label for each image.
            Example, for single image shape will be
            (number of scans, number of labels)

        """
        return self.fit(imgs, confounds=confounds).transform(imgs, confounds)

    @fill_doc
    def inverse_transform(self, signals):
        """Transform signals extracted from :term:`parcellations<parcellation>`
        back to brain images.

        Uses `labels_img_` (parcellations) built at fit() level.

        Parameters
        ----------
        signals : :obj:`list` of 2D :class:`numpy.ndarray`
            Each 2D array with shape (number of scans, number of regions).

        Returns
        -------
        %(imgs)s
            Brain image(s).

        """
        from .signal_extraction import signals_to_img_labels

        self._check_fitted()

        if not isinstance(signals, (list, tuple)) or\
                isinstance(signals, np.ndarray):
            signals = [signals, ]
            single_subject = True
        elif isinstance(signals, (list, tuple)) and len(signals) == 1:
            single_subject = True
        else:
            single_subject = False

        imgs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._cache(signals_to_img_labels, func_memory_level=2))
            (each_signal, self.labels_img_, self.mask_img_)
            for each_signal in signals)

        if single_subject:
            return imgs[0]
        else:
            return imgs
