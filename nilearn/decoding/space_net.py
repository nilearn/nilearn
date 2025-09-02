"""sklearn-compatible implementation of spatially structured learners.

For example: TV-L1, Graph-Net, etc
"""

import collections
import time
import warnings
from functools import partial
from typing import ClassVar

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import _preprocess_data as center_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array, check_X_y
from sklearn.utils.estimator_checks import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot

from nilearn._utils import logger
from nilearn._utils.cache_mixin import CacheMixin
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import check_embedded_masker
from nilearn._utils.param_validation import (
    adjust_screening_percentile,
    check_params,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.decoding._mixin import _ClassifierMixin, _RegressorMixin
from nilearn.image import get_data
from nilearn.maskers import SurfaceMasker
from nilearn.masking import unmask_from_to_3d_array
from nilearn.surface import SurfaceImage

from .space_net_solvers import (
    graph_net_logistic,
    graph_net_squared_loss,
    tvl1_solver,
)


def _crop_mask(mask):
    """Crops input mask to produce tighter (i.e smaller) bounding box \
    with the same support (active voxels).
    """
    idx = np.where(mask)
    if idx[0].size == 0:
        raise ValueError(
            "Empty mask: if you have given a mask, it is "
            "empty, and if you have not given a mask, the "
            "mask-extraction routines have failed. Please "
            "provide an appropriate mask."
        )
    i_min = max(idx[0].min() - 1, 0)
    i_max = idx[0].max()
    j_min = max(idx[1].min() - 1, 0)
    j_max = idx[1].max()
    k_min = max(idx[2].min() - 1, 0)
    k_max = idx[2].max()
    return mask[i_min : i_max + 1, j_min : j_max + 1, k_min : k_max + 1]


@fill_doc
def _univariate_feature_screening(
    X, y, mask, is_classif, screening_percentile, smoothing_fwhm=2.0
):
    """Select the most import features, via a univariate test.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Response Vector.

    mask : ndarray or booleans, shape (nx, ny, nz)
        Mask defining brain Rois.

    is_classif : bool
        Flag telling whether the learning task is classification or regression.

    %(screening_percentile)s

    %(smoothing_fwhm)s
        Default=2.

    Returns
    -------
    X_ : ndarray, shape (n_samples, n_features_)
        Reduced design matrix with only columns corresponding to the voxels
        retained after screening.

    mask_ : ndarray of booleans, shape (nx, ny, nz)
        Mask with support reduced to only contain voxels retained after
        screening.

    support : ndarray of ints, shape (n_features_,)
        Support of the screened mask, as a subset of the support of the
        original mask.
    """
    # smooth the data (with isotropic Gaussian kernel) before screening
    if smoothing_fwhm > 0.0:
        sX = np.empty(X.shape)
        for sample in range(sX.shape[0]):
            sX[sample] = gaussian_filter(
                unmask_from_to_3d_array(
                    X[sample].copy(),  # avoid modifying X
                    mask,
                ),
                (smoothing_fwhm, smoothing_fwhm, smoothing_fwhm),
            )[mask]
    else:
        sX = X

    # do feature screening proper
    selector = SelectPercentile(
        f_classif if is_classif else f_regression,
        percentile=screening_percentile,
    ).fit(sX, y)
    support = selector.get_support()

    # erode and then dilate mask, thus obtaining a "cleaner" version of
    # the mask on which a spatial prior actually makes sense
    mask_ = mask.copy()
    mask_[mask] = support > 0
    mask_ = binary_dilation(binary_erosion(mask_)).astype(bool)
    mask_[np.logical_not(mask)] = 0
    support = mask_[mask]
    X = X[:, support]

    return X, mask_, support


def _space_net_alpha_grid(
    X, y, eps=1e-3, n_alphas=10, l1_ratio=1.0, logistic=False
):
    """Compute the grid of alpha values for TV-L1 and Graph-Net.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data (design matrix).

    y : ndarray, shape (n_samples,)
        Target / response vector.

    l1_ratio : float, default=1
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is purely a spatial prior
        (Graph-Net, TV, etc.). ``For l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1
        and a spatial prior.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=10
        Number of alphas along the regularization path.

    logistic : bool, default=False
        Indicates where the underlying loss function is logistic.

    """
    if logistic:
        # Computes the theoretical upper bound for the overall
        # regularization, as derived in "An Interior-Point Method for
        # Large-Scale l1-Regularized Logistic Regression", by Koh, Kim,
        # Boyd, in Journal of Machine Learning Research, 8:1519-1555,
        # July 2007.
        # url: https://web.stanford.edu/~boyd/papers/pdf/l1_logistic_reg.pdf
        m = float(y.size)
        m_plus = float(y[y == 1].size)
        m_minus = float(y[y == -1].size)
        b = np.zeros_like(y)
        b[y == 1] = m_minus / m
        b[y == -1] = -m_plus / m
        alpha_max = np.max(np.abs(X.T.dot(b)))

        # tt may happen that b is in the kernel of X.T!
        if alpha_max == 0.0:
            alpha_max = np.abs(np.dot(X.T, y)).max()
    else:
        alpha_max = np.abs(np.dot(X.T, y)).max()

    # prevent alpha_max from exploding when l1_ratio = 0
    if l1_ratio == 0.0:
        l1_ratio = 1e-3
    alpha_max /= l1_ratio

    if n_alphas == 1:
        return np.array([alpha_max])

    alpha_min = alpha_max * eps
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=n_alphas)[
        ::-1
    ]


class _EarlyStoppingCallback:
    """Out-of-bag early stopping.

    A callable that returns True when the test error starts
    rising. We use a Spearman correlation (between X_test.w and y_test)
    for scoring.
    """

    def __init__(self, X_test, y_test, is_classif, debias=False, verbose=0):
        self.X_test = X_test
        self.y_test = y_test
        self.is_classif = is_classif
        self.debias = debias
        self.verbose = verbose
        self.tol = -1e-4 if self.is_classif else -1e-2
        self.test_scores = []
        self.counter = 0.0

    def __call__(self, variables):
        """Perform callback."""
        # misc
        if not isinstance(variables, dict):
            variables = {"w": variables}
        self.counter += 1
        w = variables["w"]

        # use Spearman score as stopping criterion
        score = self.test_score(w)[0]

        self.test_scores.append(score)
        if self.counter <= 20 or self.counter % 10 != 2:
            return

        # check whether score increased on average over last 5 iterations
        if (
            len(self.test_scores) > 4
            and np.mean(np.diff(self.test_scores[-5:][::-1])) >= self.tol
        ):
            message = "."
            if self.verbose > 1:
                message = (
                    f"Early stopping.\nTest score: {score:.8f} {40 * '-'}"
                )
            logger.log(
                message,
                verbose=self.verbose,
            )
            return True

        logger.log(
            f"Test score: {score:.8f}", verbose=self.verbose, msg_level=1
        )
        return False

    def _debias(self, w):
        """Debias w by rescaling the coefficients by a fixed factor.

        Precisely, the scaling factor is: <y_pred, y_test> / ||y_test||^2.
        """
        y_pred = np.dot(self.X_test, w)
        scaling = np.dot(y_pred, y_pred)
        if scaling > 0.0:
            scaling = np.dot(y_pred, self.y_test) / scaling
            w *= scaling
        return w

    def test_score(self, w):
        """Compute test score for model, given weights map `w`.

        We use correlations between linear prediction and
        ground truth (y_test).

        We return 2 scores for model selection: one is the Spearman
        correlation, which captures ordering between input and
        output, but tends to have 'flat' regions. The other
        is the Pearson correlation, that we can use to disambiguate
        between regions with equivalent Spearman correlation.

        """
        if self.is_classif:
            w = w[:-1]
        if np.ptp(w) == 0:
            # constant map, there is nothing
            return (-np.inf, -np.inf)
        y_pred = np.dot(self.X_test, w)
        spearman_score = stats.spearmanr(y_pred, self.y_test)[0]
        pearson_score = np.corrcoef(y_pred, self.y_test)[1, 0]
        if self.is_classif:
            return spearman_score, pearson_score
        else:
            return pearson_score, spearman_score


@fill_doc
def path_scores(
    solver,
    X,
    y,
    mask,
    alphas,
    l1_ratios,
    train,
    test,
    solver_params,
    is_classif=False,
    n_alphas=10,
    eps=1e-3,
    key=None,
    debias=False,
    screening_percentile=20,
    verbose=1,
):
    """Compute scores of different alphas in regression \
    and classification used by CV objects.

    Parameters
    ----------
    X : 2D array of shape (n_samples, n_features)
        Design matrix, one row per sample point.

    y : 1D array of length n_samples
        Response vector; one value per sample.

    mask : 3D arrays of :obj:`bool`
        Mask defining brain regions that we work on.

    %(alphas)s

    train : array or :obj:`list` of :obj:`int`:
        List of indices for the train samples.

    test : array or :obj:`list` of :obj:`int`
        List of indices for the test samples.

    l1_ratios : :obj:`float` or :obj:`list` of floats in the interval [0, 1]
        Constant that mixes L1 and TV (resp. Graph-Net) penalization.
        l1_ratios == 0: just smooth. l1_ratios == 1: just lasso.

    eps : :obj:`float`, default=1e-3
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : :obj:`int`, default=10
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    solver : function handle
       See for example tv.TVl1Classifier documentation.

    solver_params : :obj:`dict`
       Dictionary of param-value pairs to be passed to solver.

    is_classif : :obj:`bool`, default=False
        Indicates whether the loss is a classification loss or a
        regression loss.

    key: ??? TODO: Add description.

    %(debias)s

    %(screening_percentile)s

    %(verbose)s

    """
    if l1_ratios is None:
        raise ValueError("l1_ratios must be specified!")

    # misc
    _, n_features = X.shape
    verbose = int(verbose if verbose is not None else 0)

    # Univariate feature screening. Note that if we have only as few as 100
    # features in the mask's support, then we should use all of them to
    # learn the model i.e disable this screening)
    do_screening = (n_features > 100) and screening_percentile < 100.0
    if do_screening:
        X, mask, support = _univariate_feature_screening(
            X, y, mask, is_classif, screening_percentile
        )

    # crop the mask to have a tighter bounding box
    mask = _crop_mask(mask)

    # get train and test data
    X_train, y_train = X[train].copy(), y[train].copy()
    X_test, y_test = X[test].copy(), y[test].copy()

    # it is essential to center the data in regression
    X_train, y_train, _, y_train_mean, _ = center_data(
        X_train, y_train, fit_intercept=True, copy=False
    )

    # misc
    if not isinstance(l1_ratios, collections.abc.Iterable):
        l1_ratios = [l1_ratios]
    l1_ratios = sorted(l1_ratios)[::-1]  # from large to small l1_ratios
    best_score = -np.inf
    best_secondary_score = -np.inf
    best_l1_ratio = l1_ratios[0]
    best_alpha = None
    best_init = None
    all_test_scores = []
    if len(test) > 0.0:
        # do l1_ratio path
        for l1_ratio in l1_ratios:
            this_test_scores = []

            # make alpha grid
            if alphas is None:
                alphas_ = _space_net_alpha_grid(
                    X_train,
                    y_train,
                    l1_ratio=l1_ratio,
                    eps=eps,
                    n_alphas=n_alphas,
                    logistic=is_classif,
                )
            else:
                alphas_ = alphas
            alphas_ = sorted(alphas_)[::-1]  # from large to small l1_ratios

            # do alpha path
            if best_alpha is None:
                best_alpha = alphas_[0]
            init = None
            path_solver_params = solver_params.copy()
            # Use a lighter tol during the path
            path_solver_params["tol"] = 2 * path_solver_params.get("tol", 1e-4)
            for alpha in alphas_:
                # setup callback mechanism for early stopping
                early_stopper = _EarlyStoppingCallback(
                    X_test,
                    y_test,
                    is_classif=is_classif,
                    debias=debias,
                    verbose=verbose,
                )
                w, _, init = solver(
                    X_train,
                    y_train,
                    alpha,
                    l1_ratio,
                    mask=mask,
                    init=init,
                    callback=early_stopper,
                    verbose=max(verbose - 1, 0.0),
                    **path_solver_params,
                )

                # We use 2 scores for model selection: the second one is to
                # disambiguate between regions of equivalent Spearman
                # correlations
                score, secondary_score = early_stopper.test_score(w)
                this_test_scores.append(score)
                if np.isfinite(score) and (
                    score > best_score
                    or (
                        score == best_score
                        and secondary_score > best_secondary_score
                    )
                ):
                    best_secondary_score = secondary_score
                    best_score = score
                    best_l1_ratio = l1_ratio
                    best_alpha = alpha
                    best_init = init.copy()
            all_test_scores.append(this_test_scores)
    else:
        if alphas is None:
            alphas_ = _space_net_alpha_grid(
                X_train,
                y_train,
                l1_ratio=best_l1_ratio,
                eps=eps,
                n_alphas=n_alphas,
                logistic=is_classif,
            )
        else:
            alphas_ = alphas
        best_alpha = alphas_[0]

    # re-fit best model to high precision (i.e without early stopping, etc.)
    best_w, _, init = solver(
        X_train,
        y_train,
        best_alpha,
        best_l1_ratio,
        mask=mask,
        init=best_init,
        verbose=max(verbose - 1, 0),
        **solver_params,
    )
    if debias:
        best_w = _EarlyStoppingCallback(
            X_test,
            y_test,
            is_classif=is_classif,
            debias=debias,
            verbose=verbose,
        )._debias(best_w)

    if len(test) == 0.0:
        all_test_scores.append(np.nan)

    # unmask univariate screening
    if do_screening:
        w_ = np.zeros(len(support))
        if is_classif:
            w_ = np.append(w_, best_w[-1])
            w_[:-1][support] = best_w[:-1]
        else:
            w_[support] = best_w
        best_w = w_

    if len(best_w) == n_features:
        # TODO: implement with Xmean
        best_w = np.append(best_w, 0.0)

    all_test_scores = np.array(all_test_scores)
    return (
        all_test_scores,
        best_w,
        best_alpha,
        best_l1_ratio,
        alphas_,
        y_train_mean,
        key,
    )


@fill_doc
class BaseSpaceNet(CacheMixin, LinearRegression):
    """Regression and classification learners with sparsity and spatial priors.

    `SpaceNet` implements Graph-Net and TV-L1 priors /
    penalties. Thus, the penalty is a sum of an L1 term and a spatial term. The
    aim of such a hybrid prior is to obtain weights maps which are structured
    (due to the spatial prior) and sparse (enforced by L1 norm).

    Parameters
    ----------
    penalty : :obj:`str`, default='graph-net'
        Penalty to used in the model. Can be 'graph-net' or 'tv-l1'.

    l1_ratios : :obj:`float` or :obj:`list` of floats in the interval [0, 1]; \
        default=0.5
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratios == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map. If list is provided,
        then the best value will be selected by cross-validation.

    %(alphas)s

    n_alphas : :obj:`int`, default=10
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : :obj:`float`, default=1e-3
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    mask : filename, niimg, NiftiMasker instance, default=None
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a NiftiMasker.

    %(target_affine)s
        An important use-case of this parameter is for downsampling the
        input data to a coarser resolution (to speed of the model fit).

    %(target_shape)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(screening_percentile)s

    standardize : :obj:`bool`, default=True
        If set, then the data (X, y) are centered to have mean zero along
        axis 0. This is here because nearly all linear models will want
        their data to be centered.

    fit_intercept : :obj:`bool`, default=True
        Fit or not an intercept.

    %(max_iter)s

    tol : :obj:`float`, default=5e-4
        Defines the tolerance for convergence for the backend FISTA solver.

    %(verbose)s

    %(n_jobs)s

    %(memory)s

    %(memory_level1)s

    cv : :obj:`int`, a cv generator instance, or None, default=8
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    %(debias)s

    positive : :obj:`bool`, default=False
        When set to ``True``, forces the coefficients to be positive.
        This option is only supported for dense arrays.

        .. versionadded:: 0.12.0

    %(spacenet_fit_attributes)s

    """

    SUPPORTED_PENALTIES: ClassVar[tuple[str, ...]] = ("graph-net", "tv-l1")

    def __init__(
        self,
        penalty="graph-net",
        l1_ratios=0.5,
        alphas=None,
        n_alphas=10,
        mask=None,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        max_iter=200,
        tol=5e-4,
        memory=None,
        memory_level=1,
        standardize=True,
        verbose=1,
        n_jobs=1,
        eps=1e-3,
        cv=8,
        fit_intercept=True,
        screening_percentile=20,
        debias=False,
        positive=False,
    ):
        self.penalty = penalty
        self.n_alphas = n_alphas
        self.eps = eps
        self.l1_ratios = l1_ratios
        self.alphas = alphas
        self.mask = mask
        self.fit_intercept = fit_intercept
        self.memory = memory
        self.memory_level = memory_level
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.standardize = standardize
        self.n_jobs = n_jobs
        self.cv = cv
        self.screening_percentile = screening_percentile
        self.debias = debias
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.positive = positive

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
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(require_y=True, niimg_like=True, surf_img=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags = InputTags(niimg_like=True, surf_img=False)
        return tags

    @property
    def _is_classification(self) -> bool:
        # TODO remove for sklearn>=1.6
        # this private method can probably be removed
        # when dropping sklearn>=1.5 and replaced by just:
        #   self.__sklearn_tags__().estimator_type == "classifier"
        if SKLEARN_LT_1_6:
            # TODO remove for sklearn>=1.6
            return self._estimator_type == "classifier"
        return self.__sklearn_tags__().estimator_type == "classifier"

    def _check_params(self):
        """Make sure parameters are sane."""
        if self.l1_ratios is not None:
            l1_ratios = self.l1_ratios
            if not isinstance(l1_ratios, collections.abc.Iterable):
                l1_ratios = [l1_ratios]
            for l1_ratio in l1_ratios:
                if not 0 <= l1_ratio <= 1.0:
                    raise ValueError(
                        "l1_ratio must be in the interval [0, 1]; "
                        f" got {l1_ratio:g}"
                    )
                elif l1_ratio in (0.0, 1.0):
                    warnings.warn(
                        f"Specified l1_ratio = {l1_ratio:g}. "
                        "It's advised to only specify values of l1_ratio "
                        "strictly between 0 and 1.",
                        stacklevel=find_stack_level(),
                    )
        if not (0.0 <= self.screening_percentile <= 100.0):
            raise ValueError(
                "screening_percentile should be in the interval [0, 100]. "
                f"Got {self.screening_percentile:g}."
            )
        if self.penalty not in self.SUPPORTED_PENALTIES:
            raise ValueError(
                "'penalty' parameter must be one of "
                f"{self.SUPPORTED_PENALTIES}. "
                f"Got {self.penalty}."
            )
        if self._is_classification:
            self._validate_loss(self.loss)

    def _set_coef_and_intercept(self, w):
        """Set the loadings vector (coef) and the intercept of the fitted \
        model.
        """
        self.w_ = np.array(w)
        if self.w_.ndim == 1:
            self.w_ = self.w_[np.newaxis, :]
        self.coef_ = self.w_[:, :-1]
        if self._is_classification:
            self.intercept_ = self.w_[:, -1]
        else:
            self._set_intercept(self.Xmean_, self.ymean_, self.Xstd_)

    def _return_loss_value(self):
        """Set loss value for instances where it is not defined.

        For SpaceNetRegressor it is always "mse".
        """
        loss = getattr(self, "loss", None)
        if loss is None:
            loss = "logistic"
            if not self._is_classification:
                loss = "mse"
        return loss

    def fit(self, X, y):
        """Fit the learner.

        Parameters
        ----------
        X : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        y : array or :obj:`list` of length n_samples
            The dependent variable (age, sex, QI, etc.).

        Notes
        -----
        self : `SpaceNet` object
            Model selection is via cross-validation with bagging.
        """
        check_params(self.__dict__)
        # sanity check on params
        self._check_params()
        if isinstance(X, SurfaceImage) or isinstance(self.mask, SurfaceMasker):
            raise NotImplementedError(
                "Running space net on surface objects is not supported."
            )

        # misc
        self._check_params()

        self._fit_cache()

        tic = time.time()

        self.masker_ = check_embedded_masker(self, masker_type="nii")
        self.masker_.memory_level = self.memory_level
        X = self.masker_.fit_transform(X)

        X, y = check_X_y(
            X,
            y,
            ["csr", "csc", "coo"],
            dtype=float,
            multi_output=True,
            y_numeric=not self._is_classification,
        )

        if not self._is_classification and np.all(np.diff(y) == 0.0):
            raise ValueError(
                "The given input y must have at least 2 targets"
                " to do regression analysis. You provided only"
                f" one target {np.unique(y)}"
            )

        # misc
        self.Xmean_ = X.mean(axis=0)
        self.Xstd_ = X.std(axis=0)
        self.Xstd_[self.Xstd_ < 1e-8] = 1
        self.mask_img_ = self.masker_.mask_img_
        self.mask_ = get_data(self.mask_img_).astype(bool)
        n_samples, _ = X.shape
        y = np.array(y).copy()
        l1_ratios = self.l1_ratios
        if not isinstance(l1_ratios, collections.abc.Iterable):
            l1_ratios = [l1_ratios]
        alphas = self.alphas
        if alphas is not None and not isinstance(
            alphas, collections.abc.Iterable
        ):
            alphas = [alphas]

        loss = self._return_loss_value()

        # set backend solver
        if self.penalty.lower() == "graph-net":
            if loss == "mse":
                solver = graph_net_squared_loss
            else:
                solver = graph_net_logistic
        elif loss == "mse":
            solver = partial(tvl1_solver, loss="mse")
        else:
            solver = partial(tvl1_solver, loss="logistic")

        # generate fold indices
        case1 = (None in [alphas, l1_ratios]) and self.n_alphas > 1
        case2 = (alphas is not None) and min(len(l1_ratios), len(alphas)) > 1
        if case1 or case2:
            self.cv_ = list(
                check_cv(
                    self.cv, y=y, classifier=self._is_classification
                ).split(X, y)
            )
        else:
            # no cross-validation needed, user supplied all params
            self.cv_ = [(np.arange(n_samples), [])]
        n_folds = len(self.cv_)

        # number of problems to solve
        y = (
            self._binarize_y(y)
            if self._is_classification
            else y[:, np.newaxis]
        )

        n_problems = (
            self.n_classes_
            if self._is_classification and self.n_classes_ > 2
            else 1
        )

        # standardize y
        self.ymean_ = np.zeros(y.shape[0])
        if n_problems == 1:
            y = y[:, 0]

        # scores & mean weights map over all folds
        self.cv_scores_ = [[] for _ in range(n_problems)]
        w = np.zeros((n_problems, X.shape[1] + 1))
        self.all_coef_ = np.ndarray((n_problems, n_folds, X.shape[1]))

        self.screening_percentile_ = adjust_screening_percentile(
            self.screening_percentile, self.mask_img_, verbose=self.verbose
        )

        # main loop: loop on classes and folds
        solver_params = {"tol": self.tol, "max_iter": self.max_iter}
        self.best_model_params_ = []
        self.alpha_grids_ = []
        for (
            test_scores,
            best_w,
            best_alpha,
            best_l1_ratio,
            alphas,
            y_train_mean,
            (cls, fold),
        ) in Parallel(n_jobs=self.n_jobs, verbose=2 * self.verbose)(
            delayed(self._cache(path_scores, func_memory_level=2))(
                solver,
                X,
                y[:, cls] if n_problems > 1 else y,
                self.mask_,
                alphas,
                l1_ratios,
                self.cv_[fold][0],
                self.cv_[fold][1],
                solver_params,
                n_alphas=self.n_alphas,
                eps=self.eps,
                is_classif=self._is_classification,
                key=(cls, fold),
                debias=self.debias,
                verbose=self.verbose,
                screening_percentile=self.screening_percentile_,
            )
            for cls in range(n_problems)
            for fold in range(n_folds)
        ):
            self.best_model_params_.append((best_alpha, best_l1_ratio))
            self.alpha_grids_.append(alphas)
            self.ymean_[cls] += y_train_mean
            self.all_coef_[cls, fold] = best_w[:-1]
            if len(np.atleast_1d(l1_ratios)) == 1:
                test_scores = test_scores[0]
            self.cv_scores_[cls].append(test_scores)
            w[cls] += best_w

        # misc
        self.cv_scores_ = np.array(self.cv_scores_)
        self.best_model_params_ = np.array(self.best_model_params_)
        self.alpha_grids_ = np.array(self.alpha_grids_)
        self.ymean_ /= n_folds
        if not self._is_classification:
            self.all_coef_ = np.array(self.all_coef_)
            w = w[0]
            self.ymean_ = self.ymean_[0]

        # bagging: average best weights maps over folds
        w /= n_folds

        # set coefs and intercepts
        self._set_coef_and_intercept(w)

        # unmask weights map as a niimg
        self.coef_img_ = self.masker_.inverse_transform(self.coef_)

        self.n_elements_ = self.coef_.shape[1]

        # report time elapsed
        duration = time.time() - tic
        logger.log(
            f"Time Elapsed: {duration} seconds, {duration / 60.0} minutes.",
            self.verbose,
        )

        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "masker_")

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` Niimg-like objects or \
            {array-like, sparse matrix}, shape = (n_samples, n_features)
            See :ref:`extracting_data`.
            Data on prediction is to be made. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class label per sample.
        """
        check_is_fitted(self)

        # cast X into usual 2D array
        if not isinstance(X, np.ndarray) or len(np.shape(X)) == 1:
            X = self.masker_.transform(X)

        X = check_array(X)
        if X.shape[1] != self.n_elements_:
            raise ValueError(
                f"X has {X.shape[1]} features per sample; "
                f"expecting {self.n_elements_}."
            )

        # handle regression (least-squared loss)
        if not self._is_classification:
            return LinearRegression.predict(self, X)

        # prediction proper
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]


@fill_doc
class SpaceNetClassifier(_ClassifierMixin, BaseSpaceNet):
    """Classification learners with sparsity and spatial priors.

    `SpaceNetClassifier` implements Graph-Net and TV-L1
    priors / penalties for classification problems. Thus, the penalty
    is a sum an L1 term and a spatial term. The aim of such a hybrid prior
    is to obtain weights maps which are structured (due to the spatial
    prior) and sparse (enforced by L1 norm).

    Parameters
    ----------
    penalty : :obj:`str`, default='graph-net'
        Penalty to used in the model. Can be 'graph-net' or 'tv-l1'.

    loss : :obj:`str`, default="logistic"
        Loss to be used in the classifier. Must be one of "mse", or "logistic".

    l1_ratios : :obj:`float` or :obj:`list` of floats in the interval [0, 1]; \
        default=0.5
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratios == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map. If list is provided,
        then the best value will be selected by cross-validation.

    %(alphas)s

    n_alphas : :obj:`int`, default=10
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    mask : filename, niimg, NiftiMasker instance, default=None
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a MultiNiftiMasker with default parameters.

    %(target_affine)s

    %(target_shape)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(max_iter)s

    tol : :obj:`float`, default=1e-4.
        Defines the tolerance for convergence.

    %(memory)s

    %(memory_level1)s

    standardize : :obj:`bool`, default=True
        If set, then we'll center the data (X, y) have mean zero along axis 0.
        This is here because nearly all linear models will want their data
        to be centered.

    %(verbose)s

    %(n_jobs)s

    eps : :obj:`float`, default=1e-3
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    cv : :obj:`int`, a cv generator instance, or None, default=8
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    fit_intercept : :obj:`bool`, default=True
        Fit or not an intercept.

    %(screening_percentile)s

    %(debias)s

    positive : :obj:`bool`, default=False
        When set to ``True``, forces the coefficients to be positive.
        This option is only supported for dense arrays.

        .. versionadded:: 0.12.1

    %(spacenet_fit_attributes)s

    classes_ : ndarray of labels (`n_classes_`)
        Labels of the classes

    n_classes_ : int
        Number of classes

    See Also
    --------
    nilearn.decoding.SpaceNetRegressor: Graph-Net and TV-L1 priors/penalties

    """

    SUPPORTED_LOSSES: ClassVar[tuple[str, ...]] = ("mse", "logistic")

    def __init__(
        self,
        penalty="graph-net",
        loss="logistic",
        l1_ratios=0.5,
        alphas=None,
        n_alphas=10,
        mask=None,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        max_iter=200,
        tol=1e-4,
        memory=None,
        memory_level=1,
        standardize=True,
        verbose=1,
        n_jobs=1,
        eps=1e-3,
        cv=8,
        fit_intercept=True,
        screening_percentile=20,
        debias=False,
        positive=False,
    ):
        super().__init__(
            penalty=penalty,
            l1_ratios=l1_ratios,
            alphas=alphas,
            n_alphas=n_alphas,
            target_shape=target_shape,
            low_pass=low_pass,
            high_pass=high_pass,
            mask=mask,
            t_r=t_r,
            max_iter=max_iter,
            tol=tol,
            memory=memory,
            memory_level=memory_level,
            n_jobs=n_jobs,
            eps=eps,
            cv=cv,
            debias=debias,
            fit_intercept=fit_intercept,
            standardize=standardize,
            screening_percentile=screening_percentile,
            target_affine=target_affine,
            verbose=verbose,
            positive=positive,
        )
        self.loss = loss

        # TODO (sklearn  >= 1.6.0) remove
        self._estimator_type = "classifier"

    def _validate_loss(self, value):
        if value is not None and value not in self.SUPPORTED_LOSSES:
            raise ValueError(
                f"'loss' parameter must be one of {self.SUPPORTED_LOSSES}. "
                f"Got {value}."
            )

    def _binarize_y(self, y):
        """Encode target classes as -1 and 1.

        Helper function invoked just before fitting a classifier.
        """
        y = np.array(y)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` Niimg-like objects or \
            {array-like, sparse matrix}, shape = (n_samples, n_features)
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        y : array or :obj:`list` of length n_samples.
            Labels.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X)  w.r.t y.
        """
        check_is_fitted(self)
        return accuracy_score(y, self.predict(X))

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : Niimg-like, :obj:`list` Niimg-like objects or \
            {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for `self.classes_[1]` where >0 means this
            class would be predicted.
        """
        check_is_fitted(self)

        # for backwards compatibility - apply masker transform if X is
        # niimg-like or a list of strings
        if not isinstance(X, np.ndarray) or len(np.shape(X)) == 1:
            X = self.masker_.transform(X)

        X = check_array(X)
        if X.shape[1] != self.n_elements_:
            raise ValueError(
                f"X has {X.shape[1]} features per sample; "
                f"expecting {self.n_elements_}."
            )

        scores = (
            safe_sparse_dot(X, self.coef_.T, dense_output=True)
            + self.intercept_
        )
        return scores.ravel() if scores.shape[1] == 1 else scores


@fill_doc
class SpaceNetRegressor(_RegressorMixin, BaseSpaceNet):
    """Regression learners with sparsity and spatial priors.

    `SpaceNetRegressor` implements Graph-Net and TV-L1 priors / penalties
    for regression problems. Thus, the penalty is a sum an L1 term and a
    spatial term. The aim of such a hybrid prior is to obtain weights maps
    which are structured (due to the spatial prior) and sparse (enforced
    by L1 norm).

    Parameters
    ----------
    penalty : :obj:`str`, default='graph-net'
        Penalty to used in the model. Can be 'graph-net' or 'tv-l1'.

    l1_ratios : :obj:`float` or :obj:`list` of floats in the interval [0, 1]; \
        default=0.5
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratios == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map. If list is provided,
        then the best value will be selected by cross-validation.

    %(alphas)s

    n_alphas : :obj:`int`, default=10
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.`

    mask : filename, niimg, NiftiMasker instance, default=None
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a MultiNiftiMasker with default parameters.

    %(target_affine)s

    %(target_shape)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(max_iter)s

    tol : :obj:`float`, default=1e-4
        Defines the tolerance for convergence.

    %(memory)s

    %(memory_level1)s

    standardize : :obj:`bool`, default=True
        If set, then we'll center the data (X, y) have mean zero along axis 0.
        This is here because nearly all linear models will want their data
        to be centered.

    %(verbose)s

    %(n_jobs)s

    eps : :obj:`float`, default=1e-3
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    cv : :obj:`int`, a cv generator instance, or None, default=8
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    fit_intercept : :obj:`bool`, default=True
        Fit or not an intercept.

    %(screening_percentile)s

    %(debias)s

    positive : :obj:`bool`, default=False
        When set to ``True``, forces the coefficients to be positive.
        This option is only supported for dense arrays.

        .. versionadded:: 0.12.1


    %(spacenet_fit_attributes)s


    See Also
    --------
    nilearn.decoding.SpaceNetClassifier: Graph-Net and TV-L1 priors/penalties

    """

    def __init__(
        self,
        penalty="graph-net",
        l1_ratios=0.5,
        alphas=None,
        n_alphas=10,
        mask=None,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        max_iter=200,
        tol=1e-4,
        memory=None,
        memory_level=1,
        standardize=True,
        verbose=1,
        n_jobs=1,
        eps=1e-3,
        cv=8,
        fit_intercept=True,
        screening_percentile=20,
        debias=False,
        positive=False,
    ):
        super().__init__(
            penalty=penalty,
            l1_ratios=l1_ratios,
            alphas=alphas,
            n_alphas=n_alphas,
            target_shape=target_shape,
            low_pass=low_pass,
            high_pass=high_pass,
            mask=mask,
            t_r=t_r,
            max_iter=max_iter,
            tol=tol,
            memory=memory,
            memory_level=memory_level,
            n_jobs=n_jobs,
            eps=eps,
            cv=cv,
            debias=debias,
            fit_intercept=fit_intercept,
            standardize=standardize,
            screening_percentile=screening_percentile,
            target_affine=target_affine,
            verbose=verbose,
            positive=positive,
        )
