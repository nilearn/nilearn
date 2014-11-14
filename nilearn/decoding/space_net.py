"""
sklearn-compatible implementation of spatially structured learners (
TV-L1, S-LASSO, etc.)

"""
# Author: DOHMATOB Elvis Dopgima,
#         PIZARRO Gaspar,
#         VAROQUAUX Gael,
#         GRAMFORT Alexandre,
#         EICKENBERG Michael,
#         THIRION Bertrand
# License: simplified BSD

import warnings
import numbers
import time
from functools import partial
import numpy as np
from scipy import stats, ndimage
from sklearn.base import RegressorMixin, clone
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model.base import LinearModel
from sklearn.feature_selection import (SelectPercentile, f_regression,
                                       f_classif)
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.cross_validation import check_cv
from ..input_data import NiftiMasker
from .._utils.fixes import (center_data, LabelBinarizer, roc_auc_score,
                            atleast2d_or_csr)
from .objective_functions import _sigmoid, _unmask
from .space_net_solvers import (tvl1_solver, smooth_lasso_logistic,
                                smooth_lasso_squared_loss)


# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1827243.
X_ = None


def _get_mask_volume(mask):
    """Computes the volume of a brain mask in mm^3

    Parameters
    ----------
    mask : nibabel image object
        Input image whose voxel dimensions are to be computed.

    Returns
    -------
    vol : float
        The computed volume.
    """
    vox_dims = mask.get_header().get_zooms()[:3]
    return 1. * np.prod(vox_dims) * mask.get_data().astype(np.bool).sum()


def _crop_mask(mask):
    """Crops input mask to produce tighter (i.e smaller) bounding box with
    the same support (active voxels)"""
    idx = np.where(mask)
    i_min = max(idx[0].min() - 1, 0)
    i_max = idx[0].max()
    j_min = max(idx[1].min() - 1, 0)
    j_max = idx[1].max()
    k_min = max(idx[2].min() - 1, 0)
    k_max = idx[2].max()
    return mask[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1]


def _univariate_feature_screening(
        X, y, mask, is_classif, screening_percentile, smooth=2.):
    """
    Selects the most import features, via a univariate test

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Response Vector.

    mask: ndarray or booleans, shape (nx, ny, nz)
        Mask definining brain Rois.

    is_classif: bool
        Flag telling whether the learning task is classification or regression.

    screening_percentile : float in the closed interval [0., 100.]
        Only the `screening_percentile * 100" percent most import voxels will
        be retained.

    smooth : float, optional (default 2.)
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
    if smooth > 0.:
        sX = np.empty(X.shape)
        for sample in xrange(sX.shape[0]):
            sX[sample] = ndimage.gaussian_filter(
                _unmask(X[sample].copy(),  # avoid modifying X
                        mask), (smooth, smooth, smooth))[mask]
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


def _space_net_alpha_grid(
        X, y, eps=1e-3, n_alphas=10, l1_ratio=1., alpha_min=0.,
        standardize=False, normalize=False, fit_intercept=False,
        logistic=False):
    """Compute the grid of alpha values for TV-L1 and S-Lasso.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data (design matrix).

    y : ndarray, shape (n_samples,)
        Target / response vector.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is purely a spatial prior
        (S-LASSO, TV, etc.). ``For l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1
        and a spatial prior

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path.

    fit_intercept : bool
        Fit or not an intercept.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
    """
    if standardize:
        X, y, _, _, _ = center_data(X, y, fit_intercept=fit_intercept,
                                    normalize=normalize, copy=True)

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

        # It may happen that b is in the kernel of X.T!
        if alpha_max == 0.:
            alpha_max = np.abs(np.dot(X.T, y)).max()
    else:
        alpha_max = np.abs(np.dot(X.T, y)).max()
    alpha_max /= (X.shape[0] * l1_ratio)

    if n_alphas == 1:
        return np.array([alpha_max])
    if not alpha_min:
        alpha_min = alpha_max * eps
    else:
        if not (0 <= alpha_min < alpha_max):
            raise RuntimeError(
                ("Expecting 0 <= alpha_min < alpha_max. Got alpha_min=%g "
                 "and alpha_max=%g" % (alpha_min, alpha_max)))
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                      num=n_alphas)[::-1]


class EarlyStoppingCallback(object):
    """Out-of-bag early stopping

        A callable that returns True when the test error starts
        rising. We use a Spearman correlation (btween X_test.w and y_test)
        for scoring.
    """
    def __init__(self, X_test, y_test, is_classif, debias=False, ymean=0.,
                 verbose=0):
        self.X_test = X_test
        self.y_test = y_test
        self.is_classif = is_classif
        self.debias = debias
        self.ymean = ymean
        self.verbose = verbose
        self.test_errors = []
        self.counter = 0.

    def __call__(self, variables):
        """The callback proper """
        if not isinstance(variables, dict):
            variables = dict(w=variables)

        if self.is_classif:
            variables['w'] = variables['w'][:-1]  # strip off intercept
        self.counter += 1

        if self.counter == 0:
            # Reset the test_errors list
            self.test_errors = list()
        w = variables['w']
        w = np.ravel(w)

        # Correlation (Spearman) to output
        y_pred = np.dot(self.X_test, w)
        error = .5 * (1. - stats.spearmanr(y_pred, self.y_test)[0])
        self.test_errors.append(error)
        if not (self.counter > 20 and (self.counter % 10) == 2):
            return
        if len(self.test_errors) > 4:
            if np.mean(np.diff(self.test_errors[-5:])) >= -1e-4:
                if self.verbose:
                    print('Early stopping. Test error: %.8f %s' % (
                            error, 40 * '-'))

                # Error is steadily increasing
                return True
        if self.verbose > 1:
            print('Test error: %.8f' % error)

        return False

    def test_score(self, w):
        """Compute test score for model, given weights map `w`."""
        if self.is_classif:
            return 1. - roc_auc_score(
                (self.y_test > 0.), _sigmoid(
                    np.dot(self.X_test, w[:-1]) + w[-1]))
        else:
            if self.debias:
                y_pred = np.dot(self.X_test, w)
                scaling = np.dot(y_pred, y_pred)
                if scaling > 0.:
                    scaling = np.dot(y_pred, self.y_test) / scaling
                    w *= scaling
            y_pred = np.dot(self.X_test, w) + self.ymean  # the intercept!
            return .5 * np.mean((self.y_test - y_pred) ** 2)


def path_scores(solver, X, y, mask, alphas, l1_ratio, train,
                test, solver_params, is_classif=False, init=None, key=None,
                debias=False, Xmean=None, Xstd=1., ymean=0.,
                screening_percentile=20., verbose=1.):
    """Function to compute scores of different alphas in regression and
    classification used by CV objects

    Parameters
    ----------
    X : 2D array of shape (n_samples, n_features)
        Design matrix, one row per sample point.

    y : 1D array of length n_samples
        Response vector; one value per sample.

    nifti_masker : NiftiMasker instance
        Mask defining brain ROIs.

    alphas : list of floats
        List of regularization parameters being considered.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV (resp. Smooth Lasso) penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    solver : function handle
       See for example tv.TVl1Classifier documentation.

    solver_params: dict
       Dictionary of param-value pairs to be passed to solver.
    """
    # make local copy of mask
    mask = mask.copy()

    # misc
    _, n_features = X.shape
    verbose = int(verbose if verbose is not None else 0)
    alphas = sorted(alphas)[::-1]

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
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    test_scores = []

    # do alpha path
    best_init = init
    best_alpha = alphas[0]
    if len(test) > 0.:
        # setup callback mechanism for early stopping
        early_stopper = EarlyStoppingCallback(
            X_test, y_test, is_classif=is_classif, debias=debias, ymean=ymean,
            verbose=verbose)

        # score the alphas by model fit
        best_score = np.inf
        for alpha in alphas:
            w, _, init = solver(
                X_train, y_train, alpha, l1_ratio, mask=mask, init=init,
                callback=early_stopper, verbose=max(verbose - 1, 0.),
                **solver_params)
            score = early_stopper.test_score(w)
            test_scores.append(score)
            if score <= best_score:
                best_score = score
                best_alpha = alpha
                best_init = init.copy()

    # re-fit best model to high precision (i.e without early stopping, etc.)
    best_w, _, init = solver(X_train, y_train, best_alpha, l1_ratio,
                             mask=mask, init=best_init,
                             verbose=max(verbose - 1, 0), **solver_params)

    if len(test) == 0.:
        test_scores.append(np.nan)

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
        best_w = np.append(best_w, ymean - np.dot(Xmean, best_w))

    return test_scores, best_w, key


class SpaceNet(LinearModel, RegressorMixin):
    """
    Regression and classification learners with sparsity and spatial priors

    `SpaceNet` implements Smooth-LASSO (aka Graph-Net) and TV-L1 priors (aka
    penalties). Thus, the penalty is a sum an L1 term and a spatial term. The
    aim of such a hybrid prior is to obtain weights maps which are structured
    (due to the spatial prior) and sparse (enforced by L1 norm)

    Parameters
    ----------
    penalty : string, optional (default 'smooth-lasso')
        Penalty to used in the model. Can be 'smooth-lasso' or 'tv-l1'.

    is_classif : bool, optional (default False)
        Flag telling whether the learning task is classification or regression.

    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default None)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map.

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

    low_pass : False or float, optional, (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : False or float, optional (default None)
        This parameter is passed to signal. Clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    screening_percentile : float in the interval [0, 100]; Optional (
    default 20)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime to correspond to the volume of the user-supplied mask
        (which is typically smaller).

    standardize : bool, optional (default True):
        If set, then the data (X, y) are centered to have mean zero along
        axis 0. This is here because nearly all linear models will want
        their data to be centered.

    normalize : boolean, optional (default False)
        If True, then the data (X, y) will be normalized (to have unit std)
        before regression.

    fit_intercept : bool
        Fit or not an intercept.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float, optional (default 1e-4)
        Defines the tolerance for convergence for the backend fista solver.

    verbose : int, optional (default 0)
        Verbosity level.

    n_jobs : int, optional (default 1)
        Number of jobs in solving the sub-problems.

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
         Best alpha found by cross-validation

    `coef_` : ndarray, shape (n_classes-1, n_features)
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

    `masker_` : instance of NiftiMasker
        The nifti masker used to mask the data.

    `mask_img_` : Nifti like image
        The mask of the data. If no mask was supplied by the user,
        this attribute is the mask image computed automatically from the
        data `X`.

    `intercept_` : narray, shape (nclasses -1,)
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `cv_scores_` : ndarray, shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    `screening_percentile_` : float
        Screening percentile corrected according to volume of mask,
        relative to the volume of standard brain.
    """
    SUPPORTED_PENALTIES = ["smooth-lasso", "tv-l1"]
    SUPPORTED_LOSSES = ["mse", "logistic"]

    def __init__(self, penalty="smooth-lasso", is_classif=False, loss=None,
                 alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None, max_iter=1000, tol=1e-4,
                 memory=Memory(None), copy_data=True, standardize=True,
                 alpha_min=None, verbose=0, n_jobs=1, n_alphas=10, eps=1e-3,
                 cv=8, fit_intercept=True, screening_percentile=20.,
                 debias=False):
        super(SpaceNet, self).__init__()
        self.penalty = penalty
        self.is_classif = is_classif
        self.loss = loss
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.l1_ratio = l1_ratio
        self.mask = mask
        self.fit_intercept = fit_intercept
        self.memory = memory
        self.max_iter = max_iter
        self.tol = tol
        self.copy_data = copy_data
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

        # sanity check on params
        self.check_params()

    def check_params(self):
        """Makes sure parameters are sane"""
        for param in ["alpha", "l1_ratio"]:
            value = getattr(self, param)
            if not (value is None or isinstance(value, numbers.Number)):
                raise ValueError(
                    "'%s' parameter must be None or a float; got %s" % (
                        param, value))
        if not 0 <= self.l1_ratio <= 1.:
            raise ValueError(
                "l1_ratio must be in the interval [0, 1]; got %g" % (
                    self.l1_ratio))
        if not (0. <= self.screening_percentile <= 100.):
            raise ValueError(
                ("screening_percentile should be in the interval"
                 " [0, 100], got %g" % self.screening_percentile))
        if self.penalty.lower() not in self.SUPPORTED_PENALTIES:
            raise ValueError(
                "'penalty' parameter must be one of %s%s or %s; got %s" % (
                    ",".join(self.SUPPORTED_PENALTIES[:-1]), "," if len(
                        self.SUPPORTED_PENALTIES) > 2 else "",
                    self.SUPPORTED_PENALTIES[-1], self.penalty))
        if not (self.loss is None or
                self.loss.lower() in self.SUPPORTED_LOSSES):
            raise ValueError(
                "'loss' parameter must be one of %s%s or %s; got %s" % (
                    ",".join(self.SUPPORTED_LOSSES[:-1]), "," if len(
                        self.SUPPORTED_LOSSES) > 2 else "",
                    self.SUPPORTED_LOSSES[-1], self.loss))
        if not self.loss is None and not self.is_classif and (
                self.loss.lower() == "logistic"):
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
            self._set_intercept(self.Xmean, self.ymean, self.Xstd)

    def _standardize_X(self, X, copy=False):
        """Standardize data so that it each sample point has 0 mean and 0
        variance.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Sample points to be standardized.

        copy : bool, optional (default False)
            If False, then X will be modified inplace.

        Returns
        -------
        X_ : ndarray, shape (n_samples, n_features)
            Standardized data.

        Xmean : ndarray, shape (n_samples,)
            Mean along axis 0 of input data `X`.

        Xstd : ndarray, shape (n_samples,)
            Standard deviation (std) along axis 0 of input data `X`.
        """
        if copy:
            X = X.copy()
        Xmean = X.mean(axis=0)
        Xstd = X.std(axis=0)
        X -= Xmean
        X /= Xstd
        return X, Xmean, Xstd

    def _standardize_y(slef, y, copy=False):
        """Standardize response vector y so that it has 0 mean and unit
        variance.

        Parameters
        ----------
        y : ndarray, shape (n_samples,)
            Response vector to be standardize, one value per sample point.

        copy : bool, optional (default False)
            If False, then `y` will be modified inplace.

        Returns
        -------
        y_ : ndarray, shape (n_samples,)
            Standardized version of `y`.

        ymean : ndarray, shape (n_samples,)
            Mean along input `y`.
        """

        ymean = y.mean()
        y -= ymean
        return y, ymean

    def fit(self, X, y):
        """Fit the learner

        Parameters
        ----------
        X : list of filenames or NiImages of length n_samples, or 2D array of
           shape (n_samples, n_features)
            Brain images on which the which a structured weights map is to be
            learned. This is the independent variable (e.g gray-matter maps
            from VBM analysis, etc.)

        y : array or list of length n_samples
            The dependent variable (age, sex, QI, etc.)

        Notes
        -----
        self : `SpaceNet` object
            Model selection is via cross-validation with bagging.
        """
        # sanity check on params
        self.check_params()

        # sanitize object's memory
        if self.memory is None or isinstance(self.memory, basestring):
            self.memory_ = Memory(self.memory)
        else:
            self.memory_ = self.memory

        if self.verbose:
            tic = time.time()

        # compute / sanitize mask
        if isinstance(self.mask, NiftiMasker):
            self.masker_ = clone(self.mask)
        else:
            # compute mask
            self.masker_ = NiftiMasker(mask_img=self.mask,
                                       target_affine=self.target_affine,
                                       target_shape=self.target_shape,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       mask_strategy='epi', t_r=self.t_r,
                                       memory=self.memory_)
        X = self.masker_.fit_transform(X)
        self.mask_img_ = self.masker_.mask_img_
        self.mask_ = self.mask_img_.get_data().astype(np.bool)
        n_samples, _ = X.shape

        y = np.array(y).copy()

        # misc
        if not self.loss is None:
            self.loss_ = self.loss.lower()
        elif self.is_classif:
            self.loss_ = "logistic"
        else:
            self.loss_ = "mse"

        # set backend solver
        if self.penalty.lower() == "smooth-lasso":
            if not self.is_classif or self.loss == "mse":
                solver = smooth_lasso_squared_loss
            else:
                solver = smooth_lasso_logistic
        else:
            if not self.is_classif or self.loss == "mse":
                solver = partial(tvl1_solver, loss="mse")
            else:
                solver = partial(tvl1_solver, loss="logistic")

        if self.is_classif:
            y = self._binarize_y(y)
        else:
            y = y[:, np.newaxis]
        if self.is_classif and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1

        # scaling: standardize data (X, y)
        ymean = np.zeros(y.shape[0])
        if self.standardize:
            X, Xmean, Xstd = self._standardize_X(X)
            if not self.is_classif:
                for c in xrange(y.shape[1]):
                    y[:, c], ymean[c] = self._standardize_y(y[:, c])
        else:
            Xmean = np.zeros(X.shape[1])
            Xstd = np.ones(X.shape[1])
        if not self.is_classif:
            self.Xmean = Xmean
            self.Xstd = Xstd
            self.ymean = ymean[0]
        if n_problems == 1:
            y = y[:, 0]

        # make / sanitize alpha grid
        if self.alpha is not None:
            alphas = [self.alpha]
        elif self.alphas is None:
            alphas = _space_net_alpha_grid(
                X, y, l1_ratio=self.l1_ratio, eps=self.eps,
                n_alphas=self.n_alphas, standardize=self.standardize,
                normalize=True, alpha_min=self.alpha_min,
                fit_intercept=self.fit_intercept, logistic=self.is_classif)
        else:
            alphas = np.array(self.alphas)

        # always sort alphas from largest to smallest
        alphas = np.sort(alphas)[::-1]

        # generate fold indices
        if len(alphas) > 1:
            cv = list(check_cv(self.cv, X=X, y=y, classifier=self.is_classif))
        else:
            cv = [(range(n_samples), [])]  # single fold
        self.n_folds_ = len(cv)

        # scores & mean weights map over all folds
        self.cv_scores_ = [[] for _ in range(n_problems)]
        w = np.zeros((n_problems, X.shape[1] + 1))

        # correct screening_percentile according to the volume of the data mask
        mask_volume = _get_mask_volume(self.mask_img_)
        print "Mask volume = %gmm^3 = %gcm^3" % (
            mask_volume, mask_volume / 1000.)
        print "Standard brain volume = %gmm^3 = %gcm^3" % (
            MNI152_BRAIN_VOLUME, MNI152_BRAIN_VOLUME / 1000.)
        if mask_volume > MNI152_BRAIN_VOLUME:
            warnings.warn(
                "Brain mask is bigger than volume of standard brain!")
        self.screening_percentile_ = self.screening_percentile * (
            mask_volume / MNI152_BRAIN_VOLUME)
        if self.verbose:
            print "Original screening-percentile: %g" % (
                self.screening_percentile)
            print "Volume-corrected screening-percentile: %g" % (
                self.screening_percentile_)

        # main loop: loop on classes and folds
        solver_params = dict(tol=self.tol, max_iter=self.max_iter,
                             rescale_alpha=True)
        for test_scores, best_w, c in Parallel(n_jobs=self.n_jobs)(
            delayed(self.memory_.cache(path_scores))(
                solver, X, y[:, c] if n_problems > 1 else y, self.mask_,
                alphas, self.l1_ratio, train, test, solver_params,
                is_classif=self.loss == "logistic", key=c,
                debias=self.debias, Xmean=Xmean, ymean=ymean[c], Xstd=Xstd,
                verbose=self.verbose,
                screening_percentile=self.screening_percentile_
                ) for c in xrange(n_problems) for (train, test) in cv):
            test_scores = np.reshape(test_scores, (-1, 1))
            if not len(self.cv_scores_[c]):
                self.cv_scores_[c] = test_scores
            else:
                self.cv_scores_[c] = np.hstack((self.cv_scores_[c],
                                                test_scores))
            w[c] += best_w

        # keep best alpha, for historical reasons
        self.alphas_ = alphas
        self.i_alpha_ = [np.argmin(np.mean(self.cv_scores_[c], axis=-1))
                         for c in xrange(n_problems)]
        if n_problems == 1:
            self.i_alpha_ = self.i_alpha_[0]
        self.alpha_ = alphas[self.i_alpha_]

        # bagging: average best weights maps over folds
        w /= self.n_folds_

        # set coefs and intercepts
        self._set_coef_and_intercept(w)

        # unmask weights map as a niimg
        self.coef_img_ = self.masker_.inverse_transform(self.coef_)

        # report time elapsed
        if self.verbose:
            duration = time.time() - tic
            print "Time Elapsed: %g seconds, %i minutes."  % (duration,
                                                              duration / 60.)

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
        X : ndarray, shape(n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class label per sample.
        """
        # cast X into usual 2D array
        X = self.masker_.transform(X)

        # standardize X ?
        if self.standardize:
            X, _, _ = self._standardize_X(X, copy=True)

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


class SpaceNetClassifier(SpaceNet):
    """
    Classification learners with sparsity and spatial priors.

    `SpaceNetClassifier` implements Smooth-LASSO (aka Graph-Net) and TV-L1
    priors (aka penalties) for classification problems. Thus, the penalty
    is a sum an L1 term and a spatial term. The aim of such a hybrid prior
    is to obtain weights maps which are structured (due to the spatial
    prior) and sparse (enforced by L1 norm)

    Parameters
    ----------
    penalty : string, optional (default 'smooth-lasso')
        Penalty to used in the model. Can be 'smooth-lasso' or 'tv-l1'.

    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    loss: string, optional (default "logistic"):
        Loss to use in the classification problems. Must be one of "mse" and
        "logistic".

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default None)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map.

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

    low_pass : False or float, optional, (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : False or float, optional (default None)
        This parameter is passed to signal. Clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    screening_percentile : float in the interval [0, 100]; Optional (
    default 20)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.

    standardize : bool, optional (default True):
        If set, then we'll center the data (X, y) have mean zero along axis 0.
        This is here because nearly all linear models will want their data
        to be centered.

    normalize : boolean, optional (default False)
        If True, then the data (X, y) will be normalized (to have unit std)
        before regression.

    fit_intercept : bool
        Fit or not an intercept.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    n_jobs : int, optional (default 1)
        Number of jobs in solving the sub-problems.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    debias : bool, optional (default False)
        If set, then the estimated weights maps will be debiased.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

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
    def __init__(self, penalty="smooth-lasso", loss="logistic",
                 alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None, max_iter=1000, tol=1e-4,
                 memory=Memory(None), copy_data=True, standardize=True,
                 alpha_min=None, verbose=0, n_jobs=1, n_alphas=10, eps=1e-3,
                 cv=10, fit_intercept=True, screening_percentile=20.,
                 debias=False):
        super(SpaceNetClassifier, self).__init__(
            penalty=penalty, is_classif=True, alpha=alpha, alpha_min=alpha_min,
            target_shape=target_shape, low_pass=low_pass, high_pass=high_pass,
            alphas=alphas, n_alphas=n_alphas, l1_ratio=l1_ratio, mask=mask,
            t_r=t_r, max_iter=max_iter, tol=tol, memory=memory,
            copy_data=copy_data, n_jobs=n_jobs, eps=eps, cv=cv, debias=debias,
            fit_intercept=fit_intercept, standardize=standardize,
            screening_percentile=screening_percentile, loss=loss,
            target_affine=target_affine, verbose=verbose)

    def _binarize_y(self, y):
        """Helper function invoked just before fitting a classifier"""
        y = np.array(y)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y


class SpaceNetRegressor(SpaceNet):
    """
    Regression learners with sparsity and spatial priors.

    `SpaceNetClassifier` implements Smooth-LASSO (aka Graph-Net) and TV-L1
    priors (aka penalties) for regression problems. Thus, the penalty
    is a sum an L1 term and a spatial term. The aim of such a hybrid prior
    is to obtain weights maps which are structured (due to the spatial
    prior) and sparse (enforced by L1 norm)

    Parameters
    ----------
    penalty : string, optional (default 'smooth-lasso')
        Penalty to used in the model. Can be 'smooth-lasso' or 'tv-l1'.

    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional (default 1e-3)
        Length of the path. For example, ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default None)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and spatial prior terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO. The larger the value of this
        parameter, the sparser the estimated weights map.

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

    low_pass : False or float, optional, (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : False or float, optional (default None)
        This parameter is passed to signal. Clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    screening_percentile : float in the interval [0, 100]; Optional (
    default 20)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime to correspond to the volume of the user-supplied mask
        (which is typically smaller).

    standardize : bool, optional (default True):
        If set, then we'll center the data (X, y) have mean zero along axis 0.
        This is here because nearly all linear models will want their data
        to be centered.

    normalize : boolean, optional (default False)
        If True, then the data (X, y) will be normalized (to have unit std)
        before regression.

    fit_intercept : bool
        Fit or not an intercept.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    n_jobs : int, optional (default 1)
        Number of jobs in solving the sub-problems.

    cv : int, a cv generator instance, or None (default 10)
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

        `coef_` is readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

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
    def __init__(self, penalty="smooth-lasso", alpha=None, alphas=None,
                 l1_ratio=.5, mask=None, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None, t_r=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), copy_data=True,
                 standardize=True, alpha_min=None, verbose=0,
                 n_jobs=1, n_alphas=10, eps=1e-3, cv=10, fit_intercept=True,
                 screening_percentile=20., debias=False):
        super(SpaceNetRegressor, self).__init__(
            penalty=penalty, is_classif=False, alpha=alpha,
            alpha_min=alpha_min, target_shape=target_shape, low_pass=low_pass,
            high_pass=high_pass, alphas=alphas, n_alphas=n_alphas,
            l1_ratio=l1_ratio, mask=mask, t_r=t_r, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, n_jobs=n_jobs, eps=eps, cv=cv,
            debias=debias, fit_intercept=fit_intercept,
            standardize=standardize, screening_percentile=screening_percentile,
            target_affine=target_affine, verbose=verbose)
