"""
sklearn-compatible implementation of spatially structured learners (
TV-L1, Graph-Net, etc.)

"""
# Author: DOHMATOB Elvis Dopgima,
#         PIZARRO Gaspar,
#         VAROQUAUX Gael,
#         GRAMFORT Alexandre,
#         EICKENBERG Michael,
#         THIRION Bertrand
# License: simplified BSD

from distutils.version import LooseVersion
import sklearn
import warnings
import numbers
import time
import sys
from functools import partial
import numpy as np
from scipy import stats, ndimage
from sklearn.base import RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot
try:
    from sklearn.utils import atleast2d_or_csr
except ImportError: # sklearn 0.15
    from sklearn.utils import check_array as atleast2d_or_csr
from sklearn.linear_model.base import LinearModel, center_data
from sklearn.feature_selection import (SelectPercentile, f_regression,
                                       f_classif)
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from ..input_data.masker_validation import check_embedded_nifti_masker
from .._utils.param_validation import _adjust_screening_percentile
from .._utils.fixes import check_X_y
from .._utils.fixes import check_cv
from .._utils.compat import _basestring
from .._utils.cache_mixin import CacheMixin
from .objective_functions import _unmask
from .space_net_solvers import (tvl1_solver, _graph_net_logistic,
                                _graph_net_squared_loss)


def _crop_mask(mask):
    """Crops input mask to produce tighter (i.e smaller) bounding box with
    the same support (active voxels)"""
    idx = np.where(mask)
    if idx[0].size == 0:
        raise ValueError("Empty mask: if you have given a mask, it is "
                         "empty, and if you have not given a mask, the "
                         "mask-extraction routines have failed. Please "
                         "provide an appropriate mask.")
    i_min = max(idx[0].min() - 1, 0)
    i_max = idx[0].max()
    j_min = max(idx[1].min() - 1, 0)
    j_max = idx[1].max()
    k_min = max(idx[2].min() - 1, 0)
    k_max = idx[2].max()
    return mask[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1]


def _univariate_feature_screening(
        X, y, mask, is_classif, screening_percentile, smoothing_fwhm=2.):
    """
    Selects the most import features, via a univariate test

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Response Vector.

    mask: ndarray or booleans, shape (nx, ny, nz)
        Mask defining brain Rois.

    is_classif: bool
        Flag telling whether the learning task is classification or regression.

    screening_percentile : float in the closed interval [0., 100.]
        Only the `screening_percentile * 100" percent most import voxels will
        be retained.

    smoothing_fwhm : float, optional (default 2.)
        FWHM for isotropically smoothing the data X before F-testing. A value
        of zero means "don't smooth".

    Returns
    -------
    X_: ndarray, shape (n_samples, n_features_)
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
    if smoothing_fwhm > 0.:
        sX = np.empty(X.shape)
        for sample in range(sX.shape[0]):
            sX[sample] = ndimage.gaussian_filter(
                _unmask(X[sample].copy(),  # avoid modifying X
                        mask), (smoothing_fwhm, smoothing_fwhm,
                                smoothing_fwhm))[mask]
    else:
        sX = X

    # do feature screening proper
    selector = SelectPercentile(f_classif if is_classif else f_regression,
                                percentile=screening_percentile).fit(sX, y)
    support = selector.get_support()

    # erode and then dilate mask, thus obtaining a "cleaner" version of
    # the mask on which a spatial prior actually makes sense
    mask_ = mask.copy()
    mask_[mask] = (support > 0)
    mask_ = ndimage.binary_dilation(ndimage.binary_erosion(
        mask_)).astype(np.bool)
    mask_[np.logical_not(mask)] = 0
    support = mask_[mask]
    X = X[:, support]

    return X, mask_, support


def _space_net_alpha_grid(X, y, eps=1e-3, n_alphas=10, l1_ratio=1.,
                          logistic=False):
    """Compute the grid of alpha values for TV-L1 and Graph-Net.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data (design matrix).

    y : ndarray, shape (n_samples,)
        Target / response vector.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is purely a spatial prior
        (Graph-Net, TV, etc.). ``For l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1
        and a spatial prior.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    logistic : bool, optional (default False)
        Indicates where the underlying loss function is logistic.

    """

    if logistic:
        # Computes the theoretical upper bound for the overall
        # regularization, as derived in "An Interior-Point Method for
        # Large-Scale l1-Regularized Logistic Regression", by Koh, Kim,
        # Boyd, in Journal of Machine Learning Research, 8:1519-1555,
        # July 2007.
        # url: http://www.stanford.edu/~boyd/papers/pdf/l1_logistic_reg.pdf
        m = float(y.size)
        m_plus = float(y[y == 1].size)
        m_minus = float(y[y == -1].size)
        b = np.zeros_like(y)
        b[y == 1] = m_minus / m
        b[y == -1] = - m_plus / m
        alpha_max = np.max(np.abs(X.T.dot(b)))

        # tt may happen that b is in the kernel of X.T!
        if alpha_max == 0.:
            alpha_max = np.abs(np.dot(X.T, y)).max()
    else:
        alpha_max = np.abs(np.dot(X.T, y)).max()

    # prevent alpha_max from exploding when l1_ratio = 0
    if l1_ratio == 0.:
        l1_ratio = 1e-3
    alpha_max /= l1_ratio

    if n_alphas == 1:
        return np.array([alpha_max])

    alpha_min = alpha_max * eps
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                       num=n_alphas)[::-1]


class _EarlyStoppingCallback(object):
    """Out-of-bag early stopping

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
        self.counter = 0.

    def __call__(self, variables):
        """The callback proper """
        # misc
        if not isinstance(variables, dict):
            variables = dict(w=variables)
        self.counter += 1
        w = variables['w']

        # use Spearman score as stopping criterion
        score = self.test_score(w)[0]

        self.test_scores.append(score)
        if not (self.counter > 20 and (self.counter % 10) == 2):
            return

        # check whether score increased on average over last 5 iterations
        if len(self.test_scores) > 4:
            if np.mean(np.diff(self.test_scores[-5:][::-1])) >= self.tol:
                if self.verbose:
                    if self.verbose > 1:
                        print('Early stopping. Test score: %.8f %s' % (
                              score, 40 * '-'))
                    else:
                        sys.stderr.write('.')
                return True

        if self.verbose > 1:
            print('Test score: %.8f' % score)
        return False

    def _debias(self, w):
        """"Debias w by rescaling the coefficients by a fixed factor.

        Precisely, the scaling factor is: <y_pred, y_test> / ||y_test||^2.
        """
        y_pred = np.dot(self.X_test, w)
        scaling = np.dot(y_pred, y_pred)
        if scaling > 0.:
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
        if w.ptp() == 0:
            # constant map, there is nothing
            return (-np.inf, -np.inf)
        y_pred = np.dot(self.X_test, w)
        spearman_score = stats.spearmanr(y_pred, self.y_test)[0]
        pearson_score = np.corrcoef(y_pred, self.y_test)[1, 0]
        if self.is_classif:
            return spearman_score, pearson_score
        else:
            return pearson_score, spearman_score


def path_scores(solver, X, y, mask, alphas, l1_ratios, train, test,
                solver_params, is_classif=False, n_alphas=10, eps=1E-3,
                key=None, debias=False, Xmean=None,
                screening_percentile=20., verbose=1):
    """Function to compute scores of different alphas in regression and
    classification used by CV objects

    Parameters
    ----------
    X : 2D array of shape (n_samples, n_features)
        Design matrix, one row per sample point.

    y : 1D array of length n_samples
        Response vector; one value per sample.

    mask : 3D arrays of boolean
        Mask defining brain regions that we work on.

    alphas : list of floats
        List of regularization parameters being considered.

    train : array or list of integers
        List of indices for the train samples.

    test : array or list of integers
        List of indices for the test samples.

    l1_ratio : float in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and TV (resp. Graph-Net) penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    solver : function handle
       See for example tv.TVl1Classifier documentation.

    solver_params: dict
       Dictionary of param-value pairs to be passed to solver.
    """
    if l1_ratios is None:
        raise ValueError("l1_ratios must be specified!")

    # misc
    _, n_features = X.shape
    verbose = int(verbose if verbose is not None else 0)

    # Univariate feature screening. Note that if we have only as few as 100
    # features in the mask's support, then we should use all of them to
    # learn the model i.e disable this screening)
    do_screening = (n_features > 100) and screening_percentile < 100.
    if do_screening:
        X, mask, support = _univariate_feature_screening(
            X, y, mask, is_classif, screening_percentile)

    # crop the mask to have a tighter bounding box
    mask = _crop_mask(mask)

    # get train and test data
    X_train, y_train = X[train].copy(), y[train].copy()
    X_test, y_test = X[test].copy(), y[test].copy()

    # it is essential to center the data in regression
    X_train, y_train, _, y_train_mean, _ = center_data(
        X_train, y_train, fit_intercept=True, normalize=False,
        copy=False)

    # misc
    if isinstance(l1_ratios, numbers.Number):
        l1_ratios = [l1_ratios]
    l1_ratios = sorted(l1_ratios)[::-1]  # from large to small l1_ratios
    best_score = -np.inf
    best_secondary_score = -np.inf
    best_l1_ratio = l1_ratios[0]
    best_alpha = None
    best_init = None
    all_test_scores = []
    if len(test) > 0.:
        # do l1_ratio path
        for l1_ratio in l1_ratios:
            this_test_scores = []

            # make alpha grid
            if alphas is None:
                alphas_ = _space_net_alpha_grid(
                    X_train, y_train, l1_ratio=l1_ratio, eps=eps,
                    n_alphas=n_alphas, logistic=is_classif)
            else:
                alphas_ = alphas
            alphas_ = sorted(alphas_)[::-1]  # from large to small l1_ratios

            # do alpha path
            if best_alpha is None:
                best_alpha = alphas_[0]
            init = None
            for alpha in alphas_:
                # setup callback mechanism for early stopping
                early_stopper = _EarlyStoppingCallback(
                    X_test, y_test, is_classif=is_classif, debias=debias,
                    verbose=verbose)
                w, _, init = solver(
                    X_train, y_train, alpha, l1_ratio, mask=mask, init=init,
                    callback=early_stopper, verbose=max(verbose - 1, 0.),
                    **solver_params)

                # We use 2 scores for model selection: the second one is to
                # disambiguate between regions of equivalent Spearman
                # correlations
                score, secondary_score = early_stopper.test_score(w)
                this_test_scores.append(score)
                if (np.isfinite(score) and
                        (score > best_score
                         or (score == best_score and
                             secondary_score > best_secondary_score))):
                    best_secondary_score = secondary_score
                    best_score = score
                    best_l1_ratio = l1_ratio
                    best_alpha = alpha
                    best_init = init.copy()
            all_test_scores.append(this_test_scores)
    else:
        if alphas is None:
            alphas_ = _space_net_alpha_grid(
                X_train, y_train, l1_ratio=best_l1_ratio, eps=eps,
                n_alphas=n_alphas, logistic=is_classif)
        else:
            alphas_ = alphas
        best_alpha = alphas_[0]

    # re-fit best model to high precision (i.e without early stopping, etc.)
    best_w, _, init = solver(X_train, y_train, best_alpha, best_l1_ratio,
                             mask=mask, init=best_init,
                             verbose=max(verbose - 1, 0), **solver_params)
    if debias:
        best_w = _EarlyStoppingCallback(
            X_test, y_test, is_classif=is_classif, debias=debias,
            verbose=verbose)._debias(best_w)

    if len(test) == 0.:
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
        if Xmean is None:
            Xmean = np.zeros(n_features)
        best_w = np.append(best_w, 0.)

    all_test_scores = np.array(all_test_scores)
    return (all_test_scores, best_w, best_alpha, best_l1_ratio, alphas_,
            y_train_mean, key)


class BaseSpaceNet(LinearModel, RegressorMixin, CacheMixin):
    """
    Regression and classification learners with sparsity and spatial priors

    `SpaceNet` implements Graph-Net and TV-L1 priors /
    penalties. Thus, the penalty is a sum an L1 term and a spatial term. The
    aim of such a hybrid prior is to obtain weights maps which are structured
    (due to the spatial prior) and sparse (enforced by L1 norm).

    Parameters
    ----------
    penalty : string, optional (default 'graph-net')
        Penalty to used in the model. Can be 'graph-net' or 'tv-l1'.

    loss : string, optional (default "mse")
        Loss to be used in the model. Must be an one of "mse", or "logistic".

    is_classif : bool, optional (default False)
        Flag telling whether the learning task is classification or regression.

    l1_ratios : float or list of floats in the interval [0, 1];
    optional (default .5)
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map. If list is provided,
        then the best value will be selected by cross-validation.

    alphas : float or list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.
        If None or list of floats is provided, then the best value will be
        selected by cross-validation.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    mask : filename, niimg, NiftiMasker instance, optional default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a NiftiMasker.

    target_affine : 3x3 or 4x4 matrix, optional (default None)
        This parameter is passed to image.resample_img. An important use-case
        of this parameter is for downsampling the input data to a coarser
        resolution (to speed of the model fit). Please see the related
        documentation for details.

    target_shape : 3-tuple of integers, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 20)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime to correspond to the volume of the user-supplied mask
        (which is typically smaller). If '100' is given, all the features
        are used, regardless of the number of voxels.

    standardize : bool, optional (default True):
        If set, then the data (X, y) are centered to have mean zero along
        axis 0. This is here because nearly all linear models will want
        their data to be centered.

    fit_intercept : bool, optional (default True)
        Fit or not an intercept.

    max_iter : int (default 1000)
        Defines the iterations for the solver.

    tol : float, optional (default 5e-4)
        Defines the tolerance for convergence for the backend FISTA solver.

    verbose : int, optional (default 1)
        Verbosity level.

    n_jobs : int, optional (default 1)
        Number of jobs in solving the sub-problems.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional (default 1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    cv : int, a cv generator instance, or None (default 8)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    debias : bool, optional (default False)
        If set, then the estimated weights maps will be debiased.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation.

    `coef_` : ndarray, shape (n_classes-1, n_features)
        Coefficient of the features in the decision function.

    `masker_` : instance of NiftiMasker
        The nifti masker used to mask the data.

    `mask_img_` : Nifti like image
        The mask of the data. If no mask was supplied by the user,
        this attribute is the mask image computed automatically from the
        data `X`.

    `intercept_` : narray, shape (nclasses -1,)
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `cv_` : list of pairs of lists
         List of the (n_folds,) folds. For the corresponding fold,
         each pair is composed of two lists of indices,
         one for the train samples and one for the test samples.

    `cv_scores_` : ndarray, shape (n_alphas, n_folds) or
                   (n_l1_ratios, n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    `screening_percentile_` : float
        Screening percentile corrected according to volume of mask,
        relative to the volume of standard brain.
    """
    SUPPORTED_PENALTIES = ["graph-net", "tv-l1"]
    SUPPORTED_LOSSES = ["mse", "logistic"]

    def __init__(self, penalty="graph-net", is_classif=False, loss=None,
                 l1_ratios=.5, alphas=None, n_alphas=10, mask=None,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None, max_iter=1000, tol=5e-4,
                 memory=None, memory_level=1, standardize=True, verbose=1,
                 mask_args=None,
                 n_jobs=1, eps=1e-3, cv=8, fit_intercept=True,
                 screening_percentile=20., debias=False):
        self.penalty = penalty
        self.is_classif = is_classif
        self.loss = loss
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
        self.mask_args = mask_args

        # sanity check on params
        self.check_params()

    def check_params(self):
        """Makes sure parameters are sane"""
        if self.l1_ratios is not None:
            l1_ratios = self.l1_ratios
            if isinstance(l1_ratios, numbers.Number):
                l1_ratios = [l1_ratios]
            for l1_ratio in l1_ratios:
                if not 0 <= l1_ratio <= 1.:
                    raise ValueError(
                        "l1_ratio must be in the interval [0, 1]; got %g" % (
                            l1_ratio))
                elif l1_ratio == 0. or l1_ratio == 1.:
                    warnings.warn(
                        ("Specified l1_ratio = %g. It's advised to only "
                         "specify values of l1_ratio strictly between 0 "
                         "and 1." % l1_ratio))
        if not (0. <= self.screening_percentile <= 100.):
            raise ValueError(
                ("screening_percentile should be in the interval"
                 " [0, 100], got %g" % self.screening_percentile))
        if self.penalty not in self.SUPPORTED_PENALTIES:
            raise ValueError(
                "'penalty' parameter must be one of %s%s or %s; got %s" % (
                    ",".join(self.SUPPORTED_PENALTIES[:-1]), "," if len(
                        self.SUPPORTED_PENALTIES) > 2 else "",
                    self.SUPPORTED_PENALTIES[-1], self.penalty))
        if not (self.loss is None or self.loss in self.SUPPORTED_LOSSES):
            raise ValueError(
                "'loss' parameter must be one of %s%s or %s; got %s" % (
                    ",".join(self.SUPPORTED_LOSSES[:-1]), "," if len(
                        self.SUPPORTED_LOSSES) > 2 else "",
                    self.SUPPORTED_LOSSES[-1], self.loss))
        if self.loss is not None and not self.is_classif and (
                self.loss == "logistic"):
            raise ValueError(
                ("'logistic' loss is only available for classification "
                 "problems."))

    def _set_coef_and_intercept(self, w):
        """Sets the loadings vector (coef) and the intercept of the fitted
        model."""
        self.w_ = np.array(w)
        if self.w_.ndim == 1:
            self.w_ = self.w_[np.newaxis, :]
        self.coef_ = self.w_[:, :-1]
        if self.is_classif:
            self.intercept_ = self.w_[:, -1]
        else:
            self._set_intercept(self.Xmean_, self.ymean_, self.Xstd_)

    def fit(self, X, y):
        """Fit the learner

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        y : array or list of length n_samples
            The dependent variable (age, sex, QI, etc.).

        Notes
        -----
        self : `SpaceNet` object
            Model selection is via cross-validation with bagging.
        """
        # misc
        self.check_params()
        if self.memory is None or isinstance(self.memory, _basestring):
            self.memory_ = Memory(self.memory,
                                  verbose=max(0, self.verbose - 1))
        else:
            self.memory_ = self.memory
        if self.verbose:
            tic = time.time()

        # nifti masking
        self.masker_ = check_embedded_nifti_masker(self, multi_subject=False)
        X = self.masker_.fit_transform(X)

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float,
                         multi_output=True, y_numeric=True)

        # misc
        self.Xmean_ = X.mean(axis=0)
        self.Xstd_ = X.std(axis=0)
        self.Xstd_[self.Xstd_ < 1e-8] = 1
        self.mask_img_ = self.masker_.mask_img_
        self.mask_ = self.mask_img_.get_data().astype(np.bool)
        n_samples, _ = X.shape
        y = np.array(y).copy()
        l1_ratios = self.l1_ratios
        if isinstance(l1_ratios, numbers.Number):
            l1_ratios = [l1_ratios]
        alphas = self.alphas
        if isinstance(alphas, numbers.Number):
            alphas = [alphas]
        if self.loss is not None:
            loss = self.loss
        elif self.is_classif:
            loss = "logistic"
        else:
            loss = "mse"

        # set backend solver
        if self.penalty.lower() == "graph-net":
            if not self.is_classif or loss == "mse":
                solver = _graph_net_squared_loss
            else:
                solver = _graph_net_logistic
        else:
            if not self.is_classif or loss == "mse":
                solver = partial(tvl1_solver, loss="mse")
            else:
                solver = partial(tvl1_solver, loss="logistic")

        # generate fold indices
        case1 = (None in [alphas, l1_ratios]) and self.n_alphas > 1
        case2 = (alphas is not None) and min(len(l1_ratios), len(alphas)) > 1
        if case1 or case2:
            if LooseVersion(sklearn.__version__) >= LooseVersion('0.18'):
                # scikit-learn >= 0.18
                self.cv_ = list(check_cv(
                    self.cv, y=y, classifier=self.is_classif).split(X, y))
            else:
                # scikit-learn < 0.18
                self.cv_ = list(check_cv(self.cv, X=X, y=y,
                                         classifier=self.is_classif))
        else:
            # no cross-validation needed, user supplied all params
            self.cv_ = [(np.arange(n_samples), [])]
        n_folds = len(self.cv_)

        # number of problems to solve
        if self.is_classif:
            y = self._binarize_y(y)
        else:
            y = y[:, np.newaxis]
        if self.is_classif and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1

        # standardize y
        self.ymean_ = np.zeros(y.shape[0])
        if n_problems == 1:
            y = y[:, 0]

        # scores & mean weights map over all folds
        self.cv_scores_ = [[] for i in range(n_problems)]
        w = np.zeros((n_problems, X.shape[1] + 1))
        self.all_coef_ = np.ndarray((n_problems, n_folds, X.shape[1]))

        self.screening_percentile_ = _adjust_screening_percentile(
            self.screening_percentile, self.mask_img_, verbose=self.verbose)

        # main loop: loop on classes and folds
        solver_params = dict(tol=self.tol, max_iter=self.max_iter)
        self.best_model_params_ = []
        self.alpha_grids_ = []
        for (test_scores, best_w, best_alpha, best_l1_ratio, alphas,
             y_train_mean, (cls, fold)) in Parallel(
            n_jobs=self.n_jobs, verbose=2 * self.verbose)(
                delayed(self._cache(path_scores, func_memory_level=2))(
                solver, X, y[:, cls] if n_problems > 1 else y, self.mask_,
                alphas, l1_ratios, self.cv_[fold][0], self.cv_[fold][1],
                solver_params, n_alphas=self.n_alphas, eps=self.eps,
                is_classif=self.loss == "logistic", key=(cls, fold),
                debias=self.debias, verbose=self.verbose,
                screening_percentile=self.screening_percentile_,
                ) for cls in range(n_problems) for fold in range(n_folds)):
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
        self.alpha_grids_ = np.array(self.alpha_grids_)
        self.ymean_ /= n_folds
        if not self.is_classif:
            self.all_coef_ = np.array(self.all_coef_)
            w = w[0]
            self.ymean_ = self.ymean_[0]

        # bagging: average best weights maps over folds
        w /= n_folds

        # set coefs and intercepts
        self._set_coef_and_intercept(w)

        # unmask weights map as a niimg
        self.coef_img_ = self.masker_.inverse_transform(self.coef_)

        # report time elapsed
        if self.verbose:
            duration = time.time() - tic
            print("Time Elapsed: %g seconds, %i minutes." % (
                duration, duration / 60.))

        return self

    def decision_function(self, X):
        """Predict confidence scores for samples

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        # handle regression (least-squared loss)
        if not self.is_classif:
            return LinearModel.decision_function(self, X)

        X = atleast2d_or_csr(X)
        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on prediction is to be made. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class label per sample.
        """
        # cast X into usual 2D array
        if not hasattr(self, "masker_"):
            raise RuntimeError("This %s instance is not fitted yet!" % (
                self.__class__.__name__))
        X = self.masker_.transform(X)

        # handle regression (least-squared loss)
        if not self.is_classif:
            return LinearModel.predict(self, X)

        # prediction proper
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]


class SpaceNetClassifier(BaseSpaceNet):
    """Classification learners with sparsity and spatial priors.

    `SpaceNetClassifier` implements Graph-Net and TV-L1
    priors / penalties for classification problems. Thus, the penalty
    is a sum an L1 term and a spatial term. The aim of such a hybrid prior
    is to obtain weights maps which are structured (due to the spatial
    prior) and sparse (enforced by L1 norm).

    Parameters
    ----------
    penalty : string, optional (default 'graph-net')
        Penalty to used in the model. Can be 'graph-net' or 'tv-l1'.

    loss : string, optional (default "logistic")
        Loss to be used in the classifier. Must be one of "mse", or "logistic".

    l1_ratios : float or list of floats in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map. If list is provided,
        then the best value will be selected by cross-validation.

    alphas : float or list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.
        If None or list of floats is provided, then the best value will be
        selected by cross-validation.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    mask : filename, niimg, NiftiMasker instance, optional default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a MultiNiftiMasker with default parameters.

    target_affine : 3x3 or 4x4 matrix, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    screening_percentile : float in the interval [0, 100]; Optional (default 20)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.  If '100' is given,
        all the features are used, regardless of the number of voxels.

    standardize : bool, optional (default True):
        If set, then we'll center the data (X, y) have mean zero along axis 0.
        This is here because nearly all linear models will want their data
        to be centered.

    fit_intercept : bool, optional (default True)
        Fit or not an intercept.

    max_iter : int (default 1000)
        Defines the iterations for the solver.

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 1)
        Verbosity level.

    n_jobs : int, optional (default 1)
        Number of jobs in solving the sub-problems.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional (default 1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    cv : int, a cv generator instance, or None (default 8)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    debias : bool, optional (default False)
        If set, then the estimated weights maps will be debiased.

    Attributes
    ----------
    `alpha_` : float
        Best alpha found by cross-validation.

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

    `masker_` : instance of NiftiMasker
        The nifti masker used to mask the data.

    `mask_img_` : Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `intercept_` : array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    `cv_` : list of pairs of lists
        Each pair are the list of indices for the train and test
        samples for the corresponding fold.

    `cv_scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold.

    `screening_percentile_` : float
        Screening percentile corrected according to volume of mask,
        relative to the volume of standard brain.

    """

    def __init__(self, penalty="graph-net", loss="logistic",
                 l1_ratios=.5, alphas=None, n_alphas=10, mask=None,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None, max_iter=1000, tol=1e-4,
                 memory=Memory(None), memory_level=1, standardize=True,
                 verbose=1, n_jobs=1, eps=1e-3,
                 cv=8, fit_intercept=True, screening_percentile=20.,
                 debias=False):
        super(SpaceNetClassifier, self).__init__(
            penalty=penalty, is_classif=True, l1_ratios=l1_ratios,
            alphas=alphas, n_alphas=n_alphas, target_shape=target_shape,
            low_pass=low_pass, high_pass=high_pass, mask=mask, t_r=t_r,
            max_iter=max_iter, tol=tol, memory=memory,
            memory_level=memory_level,
            n_jobs=n_jobs, eps=eps, cv=cv, debias=debias,
            fit_intercept=fit_intercept, standardize=standardize,
            screening_percentile=screening_percentile, loss=loss,
            target_affine=target_affine, verbose=verbose)

    def _binarize_y(self, y):
        """Helper function invoked just before fitting a classifier."""
        y = np.array(y)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        y : array or list of length n_samples.
            Labels.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X)  w.r.t y.
        """
        return accuracy_score(y, self.predict(X))


class SpaceNetRegressor(BaseSpaceNet):
    """Regression learners with sparsity and spatial priors.

    `SpaceNetClassifier` implements Graph-Net and TV-L1 priors / penalties
    for regression problems. Thus, the penalty is a sum an L1 term and a
    spatial term. The aim of such a hybrid prior is to obtain weights maps
    which are structured (due to the spatial prior) and sparse (enforced
    by L1 norm).

    Parameters
    ----------
    penalty : string, optional (default 'graph-net')
        Penalty to used in the model. Can be 'graph-net' or 'tv-l1'.

    l1_ratios : float or list of floats in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map. If list is provided,
        then the best value will be selected by cross-validation.

    alphas : float or list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.
        If None or list of floats is provided, then the best value will be
        selected by cross-validation.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    mask : filename, niimg, NiftiMasker instance, optional default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a MultiNiftiMasker with default parameters.

    target_affine : 3x3 or 4x4 matrix, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    screening_percentile : float in the interval [0, 100]; Optional (default 20)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime to correspond to the volume of the user-supplied mask
        (which is typically smaller).

    standardize : bool, optional (default True):
        If set, then we'll center the data (X, y) have mean zero along axis 0.
        This is here because nearly all linear models will want their data
        to be centered.

    fit_intercept : bool, optional (default True)
        Fit or not an intercept.

    max_iter : int (default 1000)
        Defines the iterations for the solver.

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 1)
        Verbosity level.

    n_jobs : int, optional (default 1)
        Number of jobs in solving the sub-problems.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional (default 1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    cv : int, a cv generator instance, or None (default 8)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    debias: bool, optional (default False)
        If set, then the estimated weights maps will be debiased.

    Attributes
    ----------
    `alpha_` : float
        Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

    `masker_` : instance of NiftiMasker
        The nifti masker used to mask the data.

    `mask_img_` : Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `intercept_` : array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    `cv_scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    `screening_percentile_` : float
        Screening percentile corrected according to volume of mask,
        relative to the volume of standard brain.
    """

    def __init__(self, penalty="graph-net", l1_ratios=.5, alphas=None,
                 n_alphas=10, mask=None, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None, t_r=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), memory_level=1,
                 standardize=True, verbose=1, n_jobs=1, eps=1e-3, cv=8,
                 fit_intercept=True, screening_percentile=20., debias=False):
        super(SpaceNetRegressor, self).__init__(
            penalty=penalty, is_classif=False, l1_ratios=l1_ratios,
            alphas=alphas, n_alphas=n_alphas, target_shape=target_shape,
            low_pass=low_pass, high_pass=high_pass, mask=mask, t_r=t_r,
            max_iter=max_iter, tol=tol, memory=memory,
            memory_level=memory_level,
            n_jobs=n_jobs, eps=eps, cv=cv, debias=debias,
            fit_intercept=fit_intercept, standardize=standardize,
            screening_percentile=screening_percentile,
            target_affine=target_affine, verbose=verbose)
