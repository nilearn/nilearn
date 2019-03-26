"""Parcellation tools such as KMeans or Ward for fMRI images
"""

import numpy as np

from sklearn.base import clone
from sklearn.feature_extraction import image
from sklearn.externals.joblib import Memory, delayed, Parallel

from ..decomposition.multi_pca import MultiPCA
from ..input_data import NiftiLabelsMasker
from .._utils.compat import _basestring
from .._utils.niimg import _safe_get_data
from .._utils.niimg_conversions import _iter_check_niimg


def _estimator_fit(data, estimator):
    """ Estimator to fit on the data matrix

    Parameters
    ----------
    data : numpy array
        Data matrix

    estimator : instance of estimator from sklearn
        MiniBatchKMeans or AgglomerativeClustering

    Returns
    -------
    labels_ : numpy.ndarray
        labels_ estimated from estimator
    """
    estimator = clone(estimator)
    estimator.fit(data.T)

    return estimator.labels_


def _check_parameters_transform(imgs, confounds):
    """A helper function to check the parameters and prepare for processing
    as a list.
    """
    if not isinstance(imgs, (list, tuple)) or \
            isinstance(imgs, _basestring):
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
                isinstance(confounds, _basestring):
            confounds = [confounds, ]

    if len(confounds) != len(imgs):
        raise ValueError("Number of confounds given does not match with "
                         "the given number of images.")
    return imgs, confounds, single_subject


def _labels_masker_extraction(img, masker, confound):
    """ Helper function for parallelizing NiftiLabelsMasker extractor
    on list of Nifti images.

    Parameters
    ----------
    img : 4D Nifti image like object
        Image to process.

    masker : instance of NiftiLabelsMasker
        Used for extracting signals with fit_transform

    confound : csv file or numpy array
        Confound used for signal cleaning while extraction.
        Passed to signal.clean

    Returns
    -------
    signals : numpy array
        Signals extracted on given img
    """
    masker = clone(masker)
    signals = masker.fit_transform(img, confounds=confound)
    return signals


class Parcellations(MultiPCA):
    """Learn parcellations on fMRI images.

    Four different types of clustering methods can be used such as kmeans,
    ward, complete, average. Kmeans will call MiniBatchKMeans whereas
    ward, complete, average are used within in Agglomerative Clustering.
    All methods are leveraged from scikit-learn.

    .. versionadded:: 0.4.1

    Parameters
    ----------
    method : str, {'kmeans', 'ward', 'complete', 'average'}
        A method to choose between for brain parcellations.

    n_parcels : int, default=50
        Number of parcellations to divide the brain data into.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    mask : Niimg-like object or NiftiMasker, MultiNiftiMasker instance
        Mask/Masker used for masking the data.
        If mask image if provided, it will be used in the MultiNiftiMasker.
        If an instance of MultiNiftiMasker is provided, then this instance
        parameters will be used in masking the data by overriding the default
        masker parameters.
        If None, mask will be automatically computed by a MultiNiftiMasker
        with default parameters.

    smoothing_fwhm : float, optional default=4.
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend : boolean, optional
        Whether to detrend signals or not.
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details. The given affine will be
        considered as same for all given list of images.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    memory : instance of joblib.Memory or str
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    labels_img_ : Nifti1Image
        Labels image to each parcellation learned on fmri images.

    masker_ : instance of NiftiMasker or MultiNiftiMasker
        The masker used to mask the data

    connectivity_ : numpy.ndarray
        voxel-to-voxel connectivity matrix computed from a mask.
        Note that this attribute is only seen if selected methods are
        Agglomerative Clustering type, 'ward', 'complete', 'average'.

    Notes
    -----
        * Transforming list of Nifti images to data matrix takes few steps.
          Reducing the data dimensionality using randomized SVD, build brain
          parcellations using KMeans or various Agglomerative methods.

        * This object uses spatially-constrained AgglomerativeClustering for
          method='ward' or 'complete' or 'average'. Spatial connectivity matrix
          (voxel-to-voxel) is built-in object which means no need of explicitly
          giving the matrix.

    """
    VALID_METHODS = ['kmeans', 'ward', 'complete', 'average']

    def __init__(self, method, n_parcels=50,
                 random_state=0, mask=None, smoothing_fwhm=4.,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None),
                 memory_level=0, n_jobs=1, verbose=1):
        self.method = method
        self.n_parcels = n_parcels

        MultiPCA.__init__(self, n_components=200,
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
        """ Fits the parcellation method on this reduced data.

        Data are coming from a base decomposition estimator which computes
        the mask and reduces the dimensionality of images using
        randomized_svd.

        Parameters
        ----------
        data : ndarray
            Shape (n_samples, n_features)

        Returns
        -------
        labels_ : numpy.ndarray
            Labels to each cluster in the brain.

        connectivity_ : numpy.ndarray
            voxel-to-voxel connectivity matrix computed from a mask.
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
        except:
            pass

        components = MultiPCA._raw_fit(self, data)

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
        else:
            mask_ = _safe_get_data(mask_img_).astype(np.bool)
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
        self.labels_img_ = self.masker_.inverse_transform(labels)

        return self

    def _check_fitted(self):
        """Helper function to check whether fit is called or not.
        """
        if not hasattr(self, 'labels_img_'):
            raise ValueError("Object has no labels_img_ attribute. "
                             "Ensure that fit() is called before transform.")

    def transform(self, imgs, confounds=None):
        """Extract signals from parcellations learned on fmri images.

        Parameters
        ----------
        imgs : List of Nifti-like images
            See http://nilearn.github.io/manipulating_images/input_output.html.
            Images to process.

        confounds: List of CSV files or arrays-like, optional
            Each file or numpy array in a list should have shape
            (number of scans, number of confounds)
            This parameter is passed to signal.clean. Please see the related
            documentation for details. Must be of same length of imgs.

        Returns
        -------
        region_signals: List of or 2D numpy.ndarray
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

    def fit_transform(self, imgs, confounds=None):
        """Fit the images to parcellations and then transform them.

        Parameters
        ----------
        imgs : List of Nifti-like images
            See http://nilearn.github.io/manipulating_images/input_output.html.
            Images for process for fit as well for transform to signals.

        confounds : List of CSV files or arrays-like, optional
            Each file or numpy array in a list should have shape
            (number of scans, number of confounds).
            This parameter is passed to signal.clean. Given confounds
            should have same length as images if given as a list.

            Note: same confounds will used for cleaning signals before
            learning parcellations.

        Returns
        -------
        region_signals: List of or 2D numpy.ndarray
            Signals extracted for each label for each image.
            Example, for single image shape will be
            (number of scans, number of labels)
        """
        return self.fit(imgs, confounds=confounds).transform(imgs,
                                                             confounds=confounds)

    def inverse_transform(self, signals):
        """Transform signals extracted from parcellations back to brain
        images.

        Uses `labels_img_` (parcellations) built at fit() level.

        Parameters
        ----------
        signals : List of 2D numpy.ndarray
            Each 2D array with shape (number of scans, number of regions)

        Returns
        -------
        imgs : List of or Nifti-like image
            Brain image(s)
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
