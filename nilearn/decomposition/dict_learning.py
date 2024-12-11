"""Dictionary learning estimator.

Perform a map learning algorithm by learning
a temporal dense dictionary along with sparse spatial loadings, that
constitutes output maps
"""

# Author: Arthur Mensch

import warnings

import numpy as np
import sklearn
from sklearn.decomposition import dict_learning_online
from sklearn.linear_model import Ridge

from nilearn._utils import fill_doc, logger
from nilearn._utils.helpers import _transfer_deprecated_param_vals

from ._base import _BaseDecomposition
from .canica import CanICA

# check_input=False is an optimization available in sklearn.
sparse_encode_args = {"check_input": False}


def _compute_loadings(components, data):
    ridge = Ridge(fit_intercept=False, alpha=1e-8)
    ridge.fit(components.T, np.asarray(data.T))
    loadings = ridge.coef_.T

    S = np.sqrt(np.sum(loadings**2, axis=0))
    S[S == 0] = 1
    loadings /= S[np.newaxis, :]
    return loadings


@fill_doc
class DictLearning(_BaseDecomposition):
    """Perform a map learning algorithm based on spatial component sparsity, \
    over a :term:`CanICA` initialization.

    This yields more stable maps than :term:`CanICA`.

    See :footcite:t:`Mensch2016`.

     .. versionadded:: 0.2

    Parameters
    ----------
    mask : Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components : int, default=20
        Number of components to extract.

    batch_size : int, default=20
        The number of samples to take in each batch.

    n_epochs : float, default=1
        Number of epochs the algorithm should run on the data.

    alpha : float, default=10
        Sparsity controlling parameter.

    dict_init : Niimg-like object, optional
        Initial estimation of dictionary maps. Would be computed from CanICA if
        not provided.

    reduction_ratio : 'auto' or float between 0. and 1., default='auto'
        - Between 0. or 1. : controls data reduction in the temporal domain.
          1. means no reduction, < 1. calls for an SVD based reduction.
        - if set to 'auto', estimator will set the number of components per
          reduced session to be n_components.

    method : {'cd', 'lars'}, default='cd'
        Coding method used by sklearn backend. Below are the possible values.
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.
    %(smoothing_fwhm)s
        Default=4mm.

    standardize : boolean, default=True
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    detrend : boolean, default=True
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
        documentation for details.

    high_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

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
    components_ : 2D numpy array (n_components x n-voxels)
        Masked dictionary components extracted from the input images.

        .. note::

            Use attribute `components_img_` rather than manually unmasking
            `components_` with `masker_` attribute.

    components_img_ : 4D Nifti image
        4D image giving the extracted components. Each 3D image is a component.

        .. versionadded:: 0.4.1

    masker_ : instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    mask_img_ : Niimg-like object
        See :ref:`extracting_data`.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        n_components=20,
        n_epochs=1,
        alpha=10,
        reduction_ratio="auto",
        dict_init=None,
        random_state=None,
        batch_size=20,
        method="cd",
        mask=None,
        smoothing_fwhm=4,
        standardize=True,
        detrend=True,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="epi",
        mask_args=None,
        n_jobs=1,
        verbose=0,
        memory=None,
        memory_level=0,
    ):
        _BaseDecomposition.__init__(
            self,
            n_components=n_components,
            random_state=random_state,
            mask=mask,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
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
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.method = method
        self.alpha = alpha
        self.reduction_ratio = reduction_ratio
        self.dict_init = dict_init

    def _init_dict(self, data):
        if self.dict_init is not None:
            components = self.masker_.transform(self.dict_init)
        else:
            canica = CanICA(
                n_components=self.n_components,
                # CanICA specific parameters
                do_cca=True,
                threshold=float(self.n_components),
                n_init=1,
                # mask parameter is not useful as we bypass masking
                mask=self.masker_,
                random_state=self.random_state,
                memory=self.memory,
                memory_level=self.memory_level,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # We use protected function _raw_fit as data
                # has already been unmasked
                canica._raw_fit(data)
            components = canica.components_
        S = (components**2).sum(axis=1)
        S[S == 0] = 1
        components /= S[:, np.newaxis]
        self.components_init_ = components

    def _init_loadings(self, data):
        self.loadings_init_ = self._cache(_compute_loadings)(
            self.components_init_, data
        )

    def _raw_fit(self, data):
        """Process unmasked data directly.

        Parameters
        ----------
        data : ndarray,
            Shape (n_samples, n_features)

        """
        logger.log("Learning initial components", self.verbose)
        self._init_dict(data)

        _, n_features = data.shape

        logger.log(
            "Computing initial loadings", verbose=self.verbose, stack_level=2
        )
        self._init_loadings(data)

        dict_init = self.loadings_init_

        max_iter = ((n_features - 1) // self.batch_size + 1) * self.n_epochs

        logger.log(" Learning dictionary", verbose=self.verbose, stack_level=2)

        # TODO: remove this when sklearn 1.0 not supported anymore;
        # replace kwargs with actual parameter name
        if sklearn.__version__ <= "1.0":
            kwargs = {"n_iter": max_iter}
        else:
            kwargs = _transfer_deprecated_param_vals(
                {"n_iter": "max_iter"}, {"max_iter": max_iter}
            )
        self.components_, _ = self._cache(dict_learning_online)(
            data.T,
            self.n_components,
            alpha=self.alpha,
            batch_size=self.batch_size,
            method=self.method,
            dict_init=dict_init,
            verbose=max(0, self.verbose - 1),
            random_state=self.random_state,
            return_code=True,
            shuffle=True,
            n_jobs=1,
            **kwargs,
        )
        self.components_ = self.components_.T
        # Unit-variance scaling
        S = np.sqrt(np.sum(self.components_**2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]

        # Flip signs in each composant so that positive part is l1 larger
        # than negative part. Empirically this yield more positive looking maps
        # than with setting the max to be positive.
        for component in self.components_:
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(
                self.components_
            )

        return self
