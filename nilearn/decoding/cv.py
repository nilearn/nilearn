"""
sklearn-compatible Cross-Validation module for TV-l1, S-LASSO, etc. models

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from functools import partial
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    f_regression, f_classif, SelectPercentile)
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.cross_validation import check_cv
from .._utils.fixes import center_data, LabelBinarizer, roc_auc_score
from .objective_functions import _sigmoid
from .estimators import _BaseRegressor, _BaseClassifier, _BaseEstimator
from .space_net_solvers import smooth_lasso_logistic, smooth_lasso_squared_loss
from .tv import tvl1_solver
from .._utils.fixes._ravel import _ravel

is_classifier = lambda obj: "Classifier" in obj.__class__.__name__


def _squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). Faster than norm(x) ** 2.
    """
    x = _ravel(x)
    return np.dot(x, x)


class EarlyStoppingCallback(object):
    """ Out-of-bag early stopping.

        A callable that returns True when the test error starts
        rising. We use a Spearman correlation (btween X_test.w and y_test)
        for scoring.
    """

    def __init__(self, X_test, y_test, verbose=False):
        self.y_test = y_test
        self.X_test = X_test
        self.test_errors = list()
        self.verbose = verbose

    def __call__(self, variables):
        i = variables['counter']
        if i == 0:
            # Reset the test_errors list
            self.test_errors = list()
        w = variables['w']
        w = np.ravel(w)

        # Correlation (Spearman) to output
        y_pred = np.dot(self.X_test, w)
        y_pred -= y_pred.mean()
        error = .5 * (1. - stats.spearmanr(y_pred, self.y_test)[0])
        self.test_errors.append(error)
        if not (i > 20 and (i % 10) == 2):
            return
        if len(self.test_errors) > 4:
            if np.mean(np.diff(self.test_errors[-5:])) >= 1e-4:
                if self.verbose:
                    print('Early stopping. Test error: %.8f %s' % (
                            error, 40 * '-'))

                # Error is steadily increasing
                return True
        if self.verbose > 1:
            print('Test error: %.8f' % error)

        return False


class _BaseFeatureSelector(BaseEstimator):
    def __init__(self, score_func, percentile=10., mask=None):
        self.score_func = score_func
        self.percentile = percentile
        self.mask = mask

    def fit_transform(self, X, y):
        if self.mask is None:
            self.mask_ = None
        else:
            self.mask_ = np.array(self.mask, dtype=np.bool)

        if self.percentile < 100. and X.shape[1] > X.shape[0]:
            self.support_ = SelectPercentile(self.score_func,
                                             percentile=self.percentile
                                             ).fit(X, y).get_support()
            X = X[:, self.support_]
            if not self.mask_ is None:
                self.mask_[self.mask] = (self.support_ > 0)
        else:
            self.support_ = np.ones(X.shape[1]).astype(np.bool)

        return X

    def inverse_transform(self, w):
        """Unmasks the input vector `w`, according to the mask learned by
        Univariate screening.

        """

        if self.support_.sum() < len(self.support_):
            w_ = np.zeros(len(self.support_))
            w_[self.support_] = w
        else:
            w_ = w

        return w_


class RegressorFeatureSelector(_BaseFeatureSelector):
    """Univariate feature selector for spatial regression models with
    mask (as in fMRI analysis).

    """

    def __init__(self, percentile=10., mask=None):
        super(RegressorFeatureSelector, self).__init__(
            f_regression, percentile=percentile, mask=mask)


class ClassifierFeatureSelector(_BaseFeatureSelector):
    """Univariate feature selector for spatial classification models with
    mask (as in fMRI analysis).

    """

    def __init__(self, percentile=10., mask=None):
        super(ClassifierFeatureSelector, self).__init__(
            f_classif, percentile=percentile, mask=mask)

    def inverse_transform(self, w):
        """Unmasks the input vector `w`, according to the mask learned by
        Univariate screening.

        """

        if self.support_.sum() < len(self.support_):
            w_ = np.zeros(len(self.support_))
            w_ = np.append(w_, w[-1])
            w_[:-1][self.support_] = w[:-1]
        else:
            w_ = w

        return w_


def _my_alpha_grid(X, y, eps=1e-3, n_alphas=10, l1_ratio=1., alpha_min=0.,
                   standardize=False, normalize=False, fit_intercept=False,
                   logistic=False):
    """Compute the grid of alpha values for TV-l1 and S-Lasso.

    Parameters
    ----------
    X : 2d array, shape (n_samples, n_features)
        Training data (design matrix).

    y : ndarray, shape (n_samples,)
        Target / response vector.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.

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
        # XXX uncovered / untested code!
        m = float(y.size)
        m_plus = float(y[y == 1].size)
        m_minus = float(y[y == -1].size)
        b = np.zeros(y.size)
        b[y == 1] = m_minus / m
        b[y == -1] = - m_plus / m
        alpha_max = np.max(np.abs(X.T.dot(b)))

        # XXX It may happen that b is in the kernel of X.T!
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
        assert 0 <= alpha_min < alpha_max
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                      num=n_alphas)[::-1]


def path_scores(solver, X, y, alphas, l1_ratio, train,
                test, logistic=False, tol=1e-4, max_iter=1000, init=None,
                mask=None, verbose=0, key=None, debias=False, ymean=0.,
                screening_percentile=10., **kwargs):
    """Function to compute scores of different alphas in regression and
    classification used by CV objects.

    Parameters
    ----------
    alphas : list of floats
        List of regularization parameters being considered.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV (resp. Smooth Lasso) penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    solver : function handle
       See for example tv.TVl1Classifier documentation.

    """

    alphas = sorted(alphas)[::-1]

    # univariate feature screening
    if logistic:
        selector = ClassifierFeatureSelector(percentile=screening_percentile,
                                             mask=mask)
    else:
        selector = RegressorFeatureSelector(percentile=screening_percentile,
                                            mask=mask)

    X = selector.fit_transform(X, y)
    mask = selector.mask_

    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    test_scores = []

    best_alpha = alphas[0]
    if len(test) > 0.:
        if logistic:
            def _test_score(w):
                return 1. - roc_auc_score(
                    (y_test > 0.), _sigmoid(np.dot(X_test, w[:-1]) + w[-1]))
        else:
            def _test_score(w):
                # debias to correct for DoF
                if debias:
                    y_pred = np.dot(X_test, w)
                    scaling = np.dot(y_pred, y_pred)
                    if scaling > 0.:
                        scaling = np.dot(y_pred, y_test) / scaling
                        w *= scaling
                y_pred = np.dot(X_test, w) + ymean  # the intercept!
                score = .5 * np.mean((y_test - y_pred) ** 2)
                return score

        # setup callback mechanism for early stopping
        earlystopper = EarlyStoppingCallback(X_test, y_test, verbose=verbose)
        env = dict(counter=0)

        def _callback(_env):
            # our callback
            if not isinstance(_env, dict):
                _env = dict(w=_env)

            if logistic:
                _env['w'] = _env['w'][:-1]  # strip off intercept
            env["counter"] += 1
            _env["counter"] = env["counter"]
            return earlystopper(_env)

        best_score = np.inf
        for alpha in alphas:
            w, _, init = solver(
                X_train, y_train, alpha, l1_ratio, mask=mask, tol=tol,
                max_iter=max_iter, init=init, verbose=verbose,
                callback=_callback, **kwargs)
            score = _test_score(w)
            test_scores.append(score)
            if score <= best_score:
                best_score = score
                best_alpha = alpha

    # Re-fit best model to high precision (i.e without early stopping, etc.).
    # N.B: This work is cached, just in case another worker on another fold
    # reports the same best alpha. Also note that the re-fit is done only on
    # the train (i.e X_train), a piece of the design X.
    best_w, _, init = solver(
        X_train, y_train, best_alpha, l1_ratio, mask=mask, tol=tol,
        max_iter=max_iter, verbose=verbose, **kwargs)

    if len(test) == 0.:
        test_scores.append(np.nan)

    # unmask univariate screening
    best_w = selector.inverse_transform(best_w)

    return test_scores, best_w, key


class _BaseCV(_BaseEstimator):
    """
    Parameters
    ----------
    alphas: list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    standardize : bool, optional (default False):
       If set, then input data (X, y) will be standardized (i.e converted to
       standard Gaussian) before model is fitted.

    normalize : boolean, optional, default False
        Parameter passed to sklearn's `center_data` function for centralizing
        the input data (X, y)

    fit_intercept : bool
        Fit or not an intercept.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    def __init__(self, alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), copy_data=True,
                 standardize=False, normalize=False, alpha_min=1e-6,
                 verbose=0, n_jobs=1, n_alphas=10, eps=1e-3,
                 fit_intercept=True, cv=10, backtracking=False,
                 screening_percentile=10., solver=None, path_scores_func=None):
        super(_BaseCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            normalize=normalize, standardize=standardize)
        self.n_jobs = n_jobs
        self.cv = cv
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.solver = solver
        self.path_scores_func = path_scores
        if not (0. <= screening_percentile <= 100.):
            raise ValueError(
                ("screening_percentile should be in the interval"
                 " [0, 100], got %g" % screening_percentile))
        self.screening_percentile = screening_percentile

    @property
    def short_name(self):
        return '%s(l1_ratio=%g)' % (self.__class__.__name__, self.l1_ratio)

    def fit(self, X, y):
        assert self.solver is not None

        # sanitize path_scores_func
        if self.path_scores_func is None:
            raise ValueError(
                "Class '%s' doesn't have a `path_scores_func` attribute!" % (
                    self.__class__.__name__))

        X = np.array(X)
        y = np.array(y).ravel()
        n_samples, _ = X.shape

        self.__class__.__name__.endswith("CV")
        solver = self.solver
        path_scores_func = self.path_scores_func
        special_kwargs = {}
        if hasattr(self, "debias"):
            special_kwargs["debias"] = getattr(self, "debias")

        # always a good idea to centralize / normalize data
        ymean = 0.
        if self.standardize:
            X, y, Xmean, ymean, Xstd = center_data(
                X, y, copy=True, normalize=self.normalize,
                fit_intercept=self.fit_intercept)
            if not is_classifier(self):
                special_kwargs["ymean"] = ymean

        # make / sanitize alpha grid
        if self.alpha is not None:
            alphas = [self.alpha]
        elif self.alphas is None:
            # XXX Are these alphas reasonable ?
            alphas = _my_alpha_grid(X, y, l1_ratio=self.l1_ratio,
                                    eps=self.eps, n_alphas=self.n_alphas,
                                    standardize=self.standardize,
                                    normalize=self.normalize,
                                    alpha_min=self.alpha_min,
                                    fit_intercept=self.fit_intercept,
                                    logistic=is_classifier(self))
        else:
            alphas = np.array(self.alphas)

        # always sort alphas from largest to smallest
        alphas = np.sort(alphas)[::-1]

        if len(alphas) > 1:
            cv = list(check_cv(self.cv, X=X, y=y, classifier=True))
        else:
            cv = [(range(n_samples), [])]
        self.n_folds_ = len(cv)

        # misc (different for classifier and regressor)
        if is_classifier(self):
            X, y = self._pre_fit(X, y)
        if is_classifier(self) and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1
            y = y.ravel()
        self.scores_ = [[] for _ in xrange(n_problems)]
        w = np.zeros((n_problems, X.shape[1] + int(is_classifier(self))))

        # parameter to path_scores function
        path_params = dict(mask=self.mask, tol=self.tol, verbose=self.verbose,
                           max_iter=self.max_iter, rescale_alpha=True,
                           backtracking=self.backtracking,
                           screening_percentile=self.screening_percentile)
        path_params.update(special_kwargs)

        _ovr_y = lambda c: y[:, c] if is_classifier(
            self) and self.n_classes_ > 2 else y

        # main loop: loop on classes and folds
        for test_scores, best_w, c in Parallel(n_jobs=self.n_jobs)(
            delayed(self.memory.cache(path_scores_func))(
                solver, X, _ovr_y(c), alphas, self.l1_ratio, train, test,
                key=c, **path_params) for c in xrange(n_problems) for (
                train, test) in cv):
            test_scores = np.reshape(test_scores, (-1, 1))
            if not len(self.scores_[c]):
                self.scores_[c] = test_scores
            else:
                self.scores_[c] = np.hstack((self.scores_[c], test_scores))
            w[c] += best_w

        self.alphas_ = alphas
        self.i_alpha_ = [np.argmin(np.mean(self.scores_[c], axis=-1))
                         for c in xrange(n_problems)]
        if n_problems == 1:
            self.i_alpha_ = self.i_alpha_
        self.alpha_ = alphas[self.i_alpha_]

        w /= self.n_folds_

        if is_classifier(self):
            self._set_coef_and_intercept(w)
        else:
            self.coef_ = w
            if self.standardize:
                self._set_intercept(Xmean, ymean, Xstd)
            else:
                self.intercept_ = 0.

        if not is_classifier(self):
            self.coef_ = self.coef_[0]
            self.scores_ = self.scores_[0]

        return self


class _BaseRegressorCV(_BaseCV, _BaseRegressor):
    """
    Parameters
    ----------
    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    standardize : bool, optional (default False):
       If set, then input data (X, y) will be standardized (i.e converted to
       standard Gaussian) before model is fitted.

    normalize : boolean, optional, default False
        Parameter passed to sklearn's `center_data` function for centralizing
        the input data (X, y)

    fit_intercept : bool
        Fit or not an intercept.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True,
                 verbose=0, n_jobs=1, n_alphas=10, eps=1e-3,
                 fit_intercept=True, cv=10, debias=False, normalize=True,
                 backtracking=False, standardize=True, alpha_min=1e-6):
        super(_BaseRegressorCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            standardize=standardize, normalize=normalize)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.debias = debias
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.path_scores_func = partial(path_scores, logistic=False)


class _BaseClassifierCV(_BaseClassifier, _BaseCV):
    """
    Parameters
    ----------
    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    Notes
    -----
    XXX For now, CV only works for two-class problems!

    """

    def __init__(self, alphas=None, l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True, eps=1e-3,
                 verbose=0, n_jobs=1, n_alphas=10,
                 alpha_min=1e-6, fit_intercept=True, cv=10, backtracking=False
                 ):
        super(_BaseClassifierCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking)
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.path_scores_func = partial(path_scores, logistic=True)

    def _pre_fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)

        if self.mask is not None:
            self.n_features_ = np.prod(self.mask.shape)
        else:
            self.n_features_ = X.shape[1]

        return X, y

    def _set_coef_and_intercept(self, w):
        self.w_ = np.array(w)
        if self.w_.ndim == 1:
            self.w_ = self.w_[np.newaxis, :]
        self.coef_ = self.w_[:, :-1]
        self.intercept_ = self.w_[:, -1]


class SmoothLassoClassifierCV(_BaseClassifierCV):
    """
    Cross-validated Smooth-Lasso logistic regression model with L1 + L2
    regularization.

    w = argmin - (1 / n_samples) * log(sigmoid(y * w.T * X)) +
          w      alpha * (l1_ratio ||w||_1 (1-l1_ratio) * .5 * <Gw, Gw>)
    where G is the spatial gradient operator

    Parameters
    ----------
    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float
        Constant that mixes L1 and G2 penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

     n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    Notes
    -----
    XXX For now, CV only works for two-class problems!

    """

    def __init__(self, alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), copy_data=True,
                 eps=1e-3, verbose=0, n_jobs=1, n_alphas=10,
                 alpha_min=1e-6, fit_intercept=True, cv=10, backtracking=False,
                 solver=smooth_lasso_logistic, screening_percentile=10.):
        super(SmoothLassoClassifierCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter,
            tol=tol, memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking)
        self.solver = solver
        self.n_jobs = n_jobs
        self.cv = cv
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.screening_percentile = screening_percentile


class SmoothLassoRegressorCV(_BaseRegressorCV):
    """
    Cross-Validated Smooth-Lasso logistic regression model with L1 + L2
    regularization.

    w = argmin  n_samples^(-1) * || y - X w ||^2 + alpha * l1_ratio ||w||_1
           w      + alpha * (1 - l1_ratio) * ||Gw||^2_2

    Parameters
    ----------
    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float
        Constant that mixes L1 and G2 penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.
        Defaults to 0.5.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    standardize : bool, optional (default False):
       If set, then input data (X, y) will be standardized (i.e converted to
       standard Gaussian) before model is fitted.

    normalize : boolean, optional, default False
        Parameter passed to sklearn's `center_data` function for centralizing
        the input data (X, y)

    fit_intercept : bool
        Fit or not an intercept.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    def __init__(self, alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), copy_data=True,
                 eps=1e-3, verbose=0, n_jobs=1, debias=False,
                 fit_intercept=True, normalize=True, n_alphas=10,
                 standardize=True, cv=10, backtracking=False, alpha_min=1e-6,
                 solver=smooth_lasso_squared_loss, screening_percentile=10.):
        super(SmoothLassoRegressorCV, self).__init__(
            self, l1_ratio=l1_ratio, mask=mask, max_iter=max_iter,
            tol=tol, memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            normalize=normalize, standardize=standardize)
        self.solver = solver
        self.n_jobs = n_jobs
        self.cv = cv
        self.debias = debias
        self.eps = eps
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.screening_percentile = screening_percentile


class TVl1ClassifierCV(_BaseClassifierCV):
    """Cross-validated TV-l1 penalized logisitic regression.

    The underlying optimization problem is the following:

        w = argmin_w -(1 / n_samples) * log(sigmoid(y * w.T * X))
                 +  alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_TV)

    Parameters
    ----------
    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    prox_max_iter : int, optional (default 5000)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    Notes
    -----
    XXX For now, CV only works for two-class problems!

    """

    def __init__(self, alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), copy_data=True,
                 eps=1e-3, verbose=0, n_jobs=1, n_alphas=10,
                 fit_intercept=True, cv=10, backtracking=False, alpha_min=1e-6,
                 solver=partial(tvl1_solver, loss='logistic'),
                 screening_percentile=10.):
        super(TVl1ClassifierCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking)
        self.n_jobs = n_jobs
        self.cv = cv
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.screening_percentile = screening_percentile
        self.solver = solver


class TVl1RegressorCV(_BaseRegressorCV):
    """Cross-validated TV-l1 penalized logisitic regression.

    The underlying optimization problem is the following:

        w = argmin_w -(1 / n_samples) * log(sigmoid(y * w.T * X))
                 +  alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_TV)

    Parameters
    ----------
    alphas : list of floats, optional (default None)
        Choices for the constant that scales the overall regularization term.
        This parameter is mutually exclusive with the `n_alphas` parameter.

    n_alphas : int, optional (default 10).
        Generate this number of alphas per regularization path.
        This parameter is mutually exclusive with the `alphas` parameter.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    alpha_min : float, optional (default 1e-6)
        Minimum value of alpha to consider. This is mutually exclusive with the
        `eps` parameter.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    standardize : bool, optional (default False):
       If set, then input data (X, y) will be standardized (i.e converted to
       standard Gaussian) before model is fitted.

    normalize : boolean, optional, default False
        Parameter passed to sklearn's `center_data` function for centralizing
        the input data (X, y)

    fit_intercept : bool
        Fit or not an intercept.

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means "keep all features".

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    prox_max_iter : int, optional (default 5000)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    def __init__(self, alpha=None, alphas=None, l1_ratio=.5, mask=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None),
                 copy_data=True, verbose=0, n_jobs=1,
                 solver=partial(tvl1_solver, loss='mse'), n_alphas=10,
                 eps=1e-3, fit_intercept=True, cv=10, debias=False,
                 normalize=True, backtracking=False, standardize=True,
                 alpha_min=1e-6, screening_percentile=10.):
        super(TVl1RegressorCV, self).__init__(
            l1_ratio=l1_ratio, mask=mask, max_iter=max_iter, tol=tol,
            memory=memory, copy_data=copy_data, verbose=verbose,
            fit_intercept=fit_intercept, backtracking=backtracking,
            standardize=standardize, normalize=normalize)
        self.solver = solver
        self.n_jobs = n_jobs
        self.cv = cv
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.debias = debias
        self.eps = eps
        self.alphas = alphas
        self.alpha_min = alpha_min
        self.screening_percentile = screening_percentile

    def fit(self, X, y):
        return _BaseRegressorCV.fit(self, X, y)


def plot_cv_scores(cvobj, i_best_alpha=None, title=None, ylabel=None,
                   errorbars=True, show_best=False):
    """Plots CV scores against regularization parameter values (alpha grid).

    `cvobj` can be a CV object (_BaseCV instance) or a pair (alphas, scores).

    """

    import pylab as pl

    if hasattr(cvobj, '__class__'):
        alphas = cvobj.alphas_
        scores = cvobj.scores_
        i_best_alpha = cvobj.i_alpha_
        if title is None:
            title = cvobj.short_name
        if ylabel is None:
            if is_classifier(cvobj):
                ylabel = "1 - AUC"
            else:
                ylabel = "Mean-Squared Error"

        if is_classifier(cvobj) and cvobj.n_classes_ > 2:
            for c in xrange(cvobj.n_classes_):
                i = None if i_best_alpha is None else i_best_alpha[c]
                plot_cv_scores((alphas, scores[c]), i_best_alpha=i,
                               title="class %i vrs rest: %s" % (c, title),
                               ylabel=ylabel, errorbars=errorbars)
            return
    else:
        assert hasattr(cvobj, "__iter__")
        alphas, scores = cvobj
        if ylabel is None:
            ylabel = "Error (Out-of-Bag)"

    pl.figure()
    lalphas_ = -np.log10(alphas)
    if errorbars:
        means = np.mean(scores, axis=-1)
        stds = np.std(scores, axis=-1)
        pl.errorbar(lalphas_, means, yerr=stds)
        if i_best_alpha is not None:
            pl.axhline(means[i_best_alpha], linestyle="--")
    else:
        pl.plot(lalphas_, scores, "s-")
    if not i_best_alpha is None and show_best:
        pl.axvline(lalphas_[i_best_alpha], linestyle="--",
                   label="Best alpha: %.1e" % alphas[i_best_alpha])
    pl.ylabel(ylabel)
    pl.xlabel("-Log10(alpha)")
    pl.legend(loc="best")
    if title:
        pl.title(title)
    pl.grid('on')


TVl1Regressor = TVl1RegressorCV
SmoothLassoRegressor = SmoothLassoRegressorCV
SmoothLassoClassifier = SmoothLassoClassifierCV
TVl1Classifier = TVl1ClassifierCV
