"""Base class for decomposition estimators.

Utilities for masking and dimension reduction of group data
"""

import glob
import inspect
import itertools
import warnings
from math import ceil
from pathlib import Path
from string import Template

import numpy as np
from joblib import Parallel, delayed
from nibabel import Nifti1Image
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import check_is_fitted
from sklearn.utils.extmath import randomized_svd, svd_flip

import nilearn
from nilearn._utils import logger
from nilearn._utils.cache_mixin import CacheMixin
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import check_embedded_masker
from nilearn._utils.niimg import safe_get_data
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils.param_validation import check_params
from nilearn._utils.path_finding import resolve_globbing
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers import NiftiMapsMasker, SurfaceMapsMasker, SurfaceMasker
from nilearn.signal import row_sum_of_squares
from nilearn.surface import SurfaceImage


def _warn_ignored_surface_masker_params(estimator):
    """Warn about parameters that are ignored by SurfaceMasker.

    Only raise warning if parameters are different
    from the default value in the estimator __init__ signature.

    Parameters
    ----------
    estimator : _BaseDecomposition
        The estimator to check for ignored parameters.
    """
    params_to_ignore = ["mask_strategy", "target_affine", "target_shape"]

    tmp = dict(**inspect.signature(estimator.__init__).parameters)

    ignored_params = []
    for param in params_to_ignore:
        if param in tmp:
            if (
                tmp[param].default is None
                and getattr(estimator, param) is not None
            ):
                # this should catch when user passes a numpy array
                ignored_params.append(param)
            elif getattr(estimator, param) != tmp[param].default:
                ignored_params.append(param)

    if ignored_params:
        warnings.warn(
            Template(
                "The following parameters are not relevant when the input "
                "images and mask are SurfaceImages: "
                "${params}. They will be ignored."
            ).substitute(params=", ".join(ignored_params)),
            UserWarning,
            stacklevel=find_stack_level(),
        )


def _fast_svd(X, n_components, random_state=None):
    """Automatically switch between randomized and lapack SVD (heuristic \
    of scikit-learn).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data to decompose

    n_components : integer
        The order of the dimensionality of the truncated SVD

    %(random_state)s
        default=0

    Returns
    -------
    U : array, shape (n_samples, n_components)
        The first matrix of the truncated svd

    S : array, shape (n_components)
        The second matrix of the truncated svd

    V : array, shape (n_components, n_features)
        The last matrix of the truncated svd

    """
    random_state = check_random_state(random_state)
    # Small problem, just call full PCA
    if max(X.shape) <= 500:
        svd_solver = "full"
    elif 1 <= n_components < 0.8 * min(X.shape):
        svd_solver = "randomized"
    # This is also the case of n_components in (0,1)
    else:
        svd_solver = "full"

    # Call different fits for either full or truncated SVD
    if svd_solver == "full":
        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)
        # The "copy" are there to free the reference on the non reduced
        # data, and hence clear memory early
        U = U[:, :n_components].copy()
        S = S[:n_components]
        V = V[:n_components].copy()
    else:
        n_iter = "auto"

        U, S, V = randomized_svd(
            X,
            n_components=n_components,
            n_iter=n_iter,
            flip_sign=True,
            random_state=random_state,
        )
    return U, S, V


def _mask_and_reduce(
    masker,
    imgs,
    confounds=None,
    reduction_ratio="auto",
    n_components=None,
    random_state=None,
    n_jobs=1,
):
    """Mask and reduce provided 4D images with given masker.

    Uses a PCA (randomized for small reduction ratio) or a range finding matrix
    on time series to reduce data size in time direction. For multiple images,
    the concatenation of data is returned, either as an ndarray or a memorymap
    (useful for big datasets that do not fit in memory).

    Parameters
    ----------
    masker : :obj:`~nilearn.maskers.NiftiMasker` or \
        :obj:`~nilearn.maskers.MultiNiftiMasker` or \
        :obj:`~nilearn.maskers.SurfaceMasker`
        Instance used to mask provided data.

    imgs : list of 4D Niimg-like objects or list of \
        :obj:`~nilearn.surface.SurfaceImage`
        See :ref:`extracting_data`.
        List of subject data to mask, reduce and stack.

    confounds : CSV file path or numpy ndarray, or pandas DataFrame, optional
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    reduction_ratio : 'auto' or float between 0. and 1., default='auto'
        - Between 0. or 1. : controls data reduction in the temporal domain
        , 1. means no reduction, < 1. calls for an SVD based reduction.
        - if set to 'auto', estimator will set the number of components per
          reduced session to be n_components.

    n_components : integer, optional
        Number of components per subject to be extracted by dimension reduction

    %(random_state)s
        default=0

    n_jobs : integer, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    Returns
    -------
    data : ndarray or memorymap
        Concatenation of reduced data.

    """
    if not hasattr(imgs, "__iter__"):
        imgs = [imgs]

    if reduction_ratio == "auto":
        if n_components is None:
            # Reduction ratio is 1 if
            # neither n_components nor ratio is provided
            reduction_ratio = 1
    else:
        reduction_ratio = (
            1 if reduction_ratio is None else float(reduction_ratio)
        )
        if not 0 <= reduction_ratio <= 1:
            raise ValueError(
                "Reduction ratio should be between 0.0 and 1.0, "
                f"got {reduction_ratio:.2f}"
            )

    if confounds is None:
        confounds = itertools.repeat(confounds)

    if reduction_ratio == "auto":
        n_samples = n_components
        reduction_ratio = None
    else:
        # We'll let _mask_and_reduce_single decide on the number of
        # samples based on the reduction_ratio
        n_samples = None

    data_list = Parallel(n_jobs=n_jobs)(
        delayed(_mask_and_reduce_single)(
            masker,
            img,
            confound,
            reduction_ratio=reduction_ratio,
            n_samples=n_samples,
            random_state=random_state,
        )
        for img, confound in zip(imgs, confounds)
    )

    subject_n_samples = [subject_data.shape[0] for subject_data in data_list]

    n_samples = np.sum(subject_n_samples)
    # n_features is the number of True vertices in the mask if it is a surface
    if isinstance(masker, SurfaceMasker):
        n_features = masker.n_elements_
    # n_features is the number of True voxels in the mask if it is a volume
    else:
        n_features = int(np.sum(safe_get_data(masker.mask_img_)))
    dtype = np.float64 if data_list[0].dtype.type is np.float64 else np.float32
    data = np.empty((n_samples, n_features), order="F", dtype=dtype)

    current_position = 0
    for i, next_position in enumerate(np.cumsum(subject_n_samples)):
        data[current_position:next_position] = data_list[i]
        current_position = next_position
        # Clear memory as fast as possible: remove the reference on
        # the corresponding block of data
        data_list[i] = None
    return data


def _mask_and_reduce_single(
    masker,
    img,
    confound,
    reduction_ratio=None,
    n_samples=None,
    random_state=None,
):
    """Implement multiprocessing from MaskReducer."""
    if confound is not None and not isinstance(confound, list):
        confound = [confound]
    this_data = masker.transform(img, confound)
    this_data = np.atleast_2d(this_data)
    # Now get rid of the img as fast as possible, to free a
    # reference count on it, and possibly free the corresponding
    # data
    del img
    random_state = check_random_state(random_state)

    data_n_samples = this_data.shape[0]
    if reduction_ratio is None:
        assert n_samples is not None
        n_samples = min(n_samples, data_n_samples)
    else:
        n_samples = ceil(data_n_samples * reduction_ratio)

    U, S, V = masker._cache(_fast_svd, func_memory_level=3)(
        this_data.T, n_samples, random_state=random_state
    )
    U = U.T.copy()
    U = U * S[:, np.newaxis]
    return U


@fill_doc
class _BaseDecomposition(CacheMixin, TransformerMixin, BaseEstimator):
    """Base class for matrix factorization based decomposition estimators.

    Handles mask logic, provides transform and inverse_transform methods

     .. versionadded:: 0.2

    Parameters
    ----------
    n_components : int, default=20
        Number of components to extract,
        for each 4D-Niimage or each 2D surface image

    %(random_state)s

    mask : Niimg-like object,  :obj:`~nilearn.maskers.MultiNiftiMasker` or
           :obj:`~nilearn.surface.SurfaceImage` or
           :obj:`~nilearn.maskers.SurfaceMasker` object, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given, for Nifti images,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters; for surface images, all the vertices will be used.

    %(smoothing_fwhm)s

    standardize : boolean, default=True
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    standardize_confounds : boolean, default=True
        If standardize_confounds is True, the confounds are z-scored:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend : boolean, default=True
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

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

          These strategies are only relevant for Nifti images and the parameter
          is ignored for SurfaceImage objects.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        :func:`nilearn.masking.compute_background_mask`,
        or :func:`nilearn.masking.compute_epi_mask`
        to fine-tune mask computation.
        Please see the related documentation for details.

    memory : instance of joblib.Memory or str, default=None
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

    %(verbose0)s

    %(base_decomposition_fit_attributes)s
    """

    def __init__(
        self,
        n_components=20,
        random_state=None,
        mask=None,
        smoothing_fwhm=None,
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
        self.n_components = n_components
        self.random_state = random_state
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _more_tags(self):
        """Return estimator tags.

        TODO (sklearn >= 1.6.0) remove
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=True)
        return tags

    @fill_doc
    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the components across subjects.

        Parameters
        ----------
        imgs : list of Niimg-like objects or \
               list of :obj:`~nilearn.surface.SurfaceImage`
            See :ref:`extracting_data`.
            Data on which the mask is calculated. If this is a list,
            the affine (for Niimg-like objects) and mesh (for SurfaceImages)
            is considered the same for all

        %(y_dummy)s

        confounds : list of CSV file paths, numpy.ndarrays
            or pandas DataFrames, optional.
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
            Should match with the list of imgs given.

        Returns
        -------
        self : object
            Returns the instance itself. Contains attributes listed
            at the object level.

        """
        del y
        # Base fit for decomposition estimators : compute the embedded masker
        check_params(self.__dict__)

        if (
            isinstance(imgs, str)
            and nilearn.EXPAND_PATH_WILDCARDS
            and glob.has_magic(imgs)
        ):
            imgs = resolve_globbing(imgs)

        if isinstance(imgs, (str, Path)) or not hasattr(imgs, "__iter__"):
            # these classes are meant for list of 4D images
            # (multi-subject), we want it to work also on a single
            # subject, so we hack it.
            imgs = [
                imgs,
            ]

        if len(imgs) == 0:
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError(
                "Need one or more Niimg-like or SurfaceImage "
                "objects as input, "
                "an empty list was given."
            )

        if confounds is not None and len(confounds) != len(imgs):
            raise ValueError(
                f"Number of confounds ({len(confounds)=}) "
                f"must match number of images ({len(imgs)=})."
            )

        self._fit_cache()

        masker_type = "multi_nii"
        if isinstance(self.mask, (SurfaceMasker, SurfaceImage)) or any(
            isinstance(x, SurfaceImage) for x in imgs
        ):
            masker_type = "surface"
            _warn_ignored_surface_masker_params(self)
        self.masker_ = check_embedded_masker(self, masker_type=masker_type)
        self.masker_.memory_level = self.memory_level

        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        # _mask_and_reduce step for decomposition estimators i.e.
        # MultiPCA, CanICA and Dictionary Learning
        logger.log("Loading data", self.verbose)
        data = _mask_and_reduce(
            self.masker_,
            imgs,
            confounds=confounds,
            n_components=self.n_components,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._raw_fit(data)

        # Create and fit appropriate MapsMasker for transform
        # and inverse_transform
        if isinstance(self.masker_, SurfaceMasker):
            self.maps_masker_ = SurfaceMapsMasker(
                self.components_img_, self.masker_.mask_img_
            )
        else:
            self.maps_masker_ = NiftiMapsMasker(
                self.components_img_,
                self.masker_.mask_img_,
                resampling_target="maps",
            )
        self.maps_masker_.fit()

        self.n_elements_ = self.maps_masker_.n_elements_

        return self

    @property
    def nifti_maps_masker_(self):
        # TODO (nilearn >= 0.13.0) remove
        warnings.warn(
            message="The 'nifti_maps_masker_' attribute is deprecated "
            "and will be removed in Nilearn 0.13.0.\n"
            "Please use 'maps_masker_' instead.",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.maps_masker_

    def __sklearn_is_fitted__(self):
        return hasattr(self, "components_")

    def transform(self, imgs, confounds=None):
        """Project the data into a reduced representation.

        Parameters
        ----------
        imgs : iterable of Niimg-like objects or \
               :obj:`list` of :obj:`~nilearn.surface.SurfaceImage`
            See :ref:`extracting_data`.
            Data to be projected

        confounds : CSV file path or numpy.ndarray
            or pandas DataFrame, optional
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        Returns
        -------
        loadings : list of 2D ndarray,
            For each subject, each sample, loadings for each decomposition
            components
            shape: number of subjects * (number of scans, number of regions)

        """
        check_is_fitted(self)

        # XXX: dealing properly with 4D/ list of 4D data?
        if isinstance(imgs, (str, Path)):
            imgs = check_niimg(imgs)

        if isinstance(imgs, (SurfaceImage, Nifti1Image)):
            imgs = [imgs]

        if confounds is None:
            confounds = list(itertools.repeat(None, len(imgs)))
        elif len(confounds) != len(imgs):
            raise ValueError(
                f"Number of confounds ({len(confounds)=}) "
                f"must match number of images ({len(imgs)=})."
            )

        return [
            self.maps_masker_.transform(img, confounds=confound)
            for img, confound in zip(imgs, confounds)
        ]

    def inverse_transform(self, loadings):
        """Use provided loadings to compute corresponding linear component \
        combination in whole-brain voxel space.

        Parameters
        ----------
        loadings : list of numpy array (n_samples x n_components)
            Component signals to transform back into voxel signals

        Returns
        -------
        reconstructed_imgs : list of nibabel.Nifti1Image or \
            :class:`~nilearn.surface.SurfaceImage`

        For each loading, reconstructed Nifti1Image or SurfaceImage.

        """
        check_is_fitted(self)

        # XXX: dealing properly with 2D/ list of 2D data?
        if not isinstance(loadings, list):
            raise TypeError(
                "'loadings' must be a list of numpy arrays. "
                f"Got: {loadings.__class__.__name__}"
            )

        return [
            self.maps_masker_.inverse_transform(loading)
            for loading in loadings
        ]

    def _sort_by_score(self, data):
        """Sort components on the explained variance over data of estimator \
        components_.
        """
        components_score = self._raw_score(data, per_component=True)
        order = np.argsort(components_score)[::-1]
        self.components_ = self.components_[order]

    def _raw_score(self, data, per_component=True):
        """Return explained variance over data of estimator components_."""
        return self._cache(_explained_variance)(
            data, self.components_, per_component=per_component
        )

    def score(self, imgs, y=None, confounds=None, per_component=False):
        """Score function based on explained variance on imgs.

        Should only be used by DecompositionEstimator derived classes

        Parameters
        ----------
        imgs : iterable of Niimg-like objects or \
               :obj:`list` of :obj:`~nilearn.surface.SurfaceImage`
            See :ref:`extracting_data`.
            Data to be scored

        %(y_dummy)s

        confounds : CSV file path or numpy.ndarray
            or pandas DataFrame, optional
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        per_component : bool, default=False
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components.

        Returns
        -------
        score : float
            Holds the score for each subjects. Score is two dimensional
            if per_component is True. First dimension
            is squeezed if the number of subjects is one

        """
        del y
        check_is_fitted(self)

        data = _mask_and_reduce(
            self.masker_,
            imgs,
            confounds,
            reduction_ratio=1.0,
            random_state=self.random_state,
        )
        return self._raw_score(data, per_component=per_component)

    def set_output(self, *, transform=None):
        """Set the output container when ``"transform"`` is called.

        .. warning::

            This has not been implemented yet.
        """
        raise NotImplementedError()


def _explained_variance(X, components, per_component=True):
    """Score function based on explained variance.

    Parameters
    ----------
    X : ndarray
        Holds single subject data to be tested against components.

    components : array-like
        Represents the components estimated by the decomposition algorithm.

    per_component : bool, default=True
        Specify whether the explained variance ratio is desired for each
        map or for the global set of components_.

    Returns
    -------
    score : ndarray
        Holds the score for each subjects. score is two dimensional if
        per_component = True.

    """
    full_var = np.var(X)
    n_components = components.shape[0]
    S = np.sqrt(np.sum(components**2, axis=1))
    S[S == 0] = 1
    components = components / S[:, np.newaxis]
    projected_data = components.dot(X.T)
    if per_component:
        res_var = np.zeros(n_components)
        for i in range(n_components):
            res = X - np.outer(projected_data[i], components[i])
            res_var[i] = np.var(res)
            # Free some memory
            del res
        return np.maximum(0.0, 1.0 - res_var / full_var)
    else:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(components.T, X.T)
        res = X - lr.coef_.dot(components)
        res_var = row_sum_of_squares(res).sum()
        return np.maximum(0.0, 1.0 - res_var / row_sum_of_squares(X).sum())
