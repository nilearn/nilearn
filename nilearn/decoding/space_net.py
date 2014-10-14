"""
sklearn-compatible implementation of spatially structured learners (
TV-l1, S-LASSO, etc.)

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Michael Eickenberg,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

import numbers
import time
from functools import partial
import numpy as np
from scipy import stats
from sklearn.base import RegressorMixin, clone
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model.base import LinearModel
from sklearn.feature_selection import (f_regression, f_classif,
                                       SelectPercentile)
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.cross_validation import check_cv
from ..input_data import NiftiMasker
from ..image.image import _fast_smooth_array
from .._utils.fixes import (center_data, LabelBinarizer, roc_auc_score,
                            atleast2d_or_csr)
from .objective_functions import _sigmoid, _unmask
from .space_net_solvers import (tvl1_solver, smooth_lasso_logistic,
                                smooth_lasso_squared_loss)


def _space_net_alpha_grid(
        X, y, eps=1e-3, n_alphas=10, l1_ratio=1., alpha_min=0.,
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
        m = float(y.size)
        m_plus = float(y[y == 1].size)
        m_minus = float(y[y == -1].size)
        b = np.zeros(y.size)
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
        assert 0 <= alpha_min < alpha_max
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                      num=n_alphas)[::-1]


class EarlyStoppingCallback(object):
    """ Out-of-bag early stopping.

        A callable that returns True when the test error starts
        rising. We use a Spearman correlation (btween X_test.w and y_test)
        for scoring.
    """

    def __init__(self, X_test, y_test, is_classif, debias=False, ymean=0.,
                 verbose=False):
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
            if np.mean(np.diff(self.test_errors[-5:])) >= 1e-4:
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
            score = .5 * np.mean((self.y_test - y_pred) ** 2)
            return score


def path_scores(solver, X, y, mask, alphas, l1_ratio, train,
                test, solver_params, is_classif=False, init=None, key=None,
                debias=False, ymean=0., screening_percentile=10.):
    """Function to compute scores of different alphas in regression and
    classification used by CV objects.

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

    n_samples, _ = X.shape

    # make local copy of mask
    mask = mask.copy()

    # misc
    verbose = solver_params.get('verbose', 0)
    alphas = sorted(alphas)[::-1]

    # univariate feature screening
    if screening_percentile < 100.:
        # smooth the data before screening
        sX = np.empty(list(mask.shape) + [n_samples])
        for row in xrange(n_samples):
            sX[:, :, :, row] = _unmask(X[row], mask)
        sX = _fast_smooth_array(sX)
        sX = np.array([sX[:, :, :, row][mask] for row in xrange(n_samples)])
        selector = SelectPercentile(f_classif if is_classif else f_regression,
                                    percentile=screening_percentile).fit(sX, y)
        support = selector.get_support()
        mask[mask] = (support > 0)
        X = X[:, support]

    # get train and test data
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    test_scores = []

    # do alpha path
    best_alpha = alphas[0]
    if len(test) > 0.:
        # setup callback mechanism for early stopping
        earlystopper = EarlyStoppingCallback(
            X_test, y_test, is_classif=is_classif, debias=debias, ymean=ymean,
            verbose=verbose)

        # score the alphas by model fit
        best_score = np.inf
        for alpha in alphas:
            w, _, init = solver(
                X_train, y_train, alpha, l1_ratio, mask=mask, init=init,
                callback=earlystopper, **solver_params)
            score = earlystopper.test_score(w)
            test_scores.append(score)
            if score <= best_score:
                best_score = score
                best_alpha = alpha

    # re-fit best model to high precision (i.e without early stopping, etc.)
    best_w, _, init = solver(X_train, y_train, best_alpha, l1_ratio,
                             mask=mask, **solver_params)

    if len(test) == 0.:
        test_scores.append(np.nan)

    # unmask univariate screening
    if screening_percentile < 100. and support.sum() < len(support):
        w_ = np.zeros(len(support))
        if is_classif:
            w_ = np.append(w_, best_w[-1])
            w_[:-1][support] = best_w[:-1]
        else:
            w_[support] = best_w
        best_w = w_

    return test_scores, best_w, key


class SpaceNet(LinearModel, RegressorMixin):
    """
    Cross-validated regression and classification learners with sparsity and
    spatial penalties (like TVl1, Smooth-Lasso, etc.).

    Parameters
    ----------
    penalty: string, optional (default 'smooth-lasso')
        Penalty to used in the model. Can be 'smooth-lasso' or 'tvl1'

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
        Constant that mixes L1 and TV (resp. SL) terms in penalization.
        l1_ratio == 1 corresponds to pure LASSO.

    mask: filename, niimg, NiftiMasker instance, optional default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a MultiNiftiMasker with default parameters.

    target_affine: 3x3 or 4x4 matrix, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: False or float, optional, (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    screening_percentile : float in the interval [0, 100]; Optional (
    default 10)
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'.

    standardize : bool, optional (default False):
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
        Number of jobs to use for One-vs-All classification.

    cv : int, a cv generator instance, or None (default 10)
        The input specifying which cross-validation generator to use.
        It can be an integer, in which case it is the number of folds in a
        KFold, None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    debias: bool, optional (default False)
        If set, then the estimated weigghts maps will be debiased.

    Attributes
    ----------
    `alpha_` : float
         Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

    `masker_`: instance of NiftiMasker
        The nifti masker used to mask the data.

    `mask_img_`: Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `intercept_` : array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_` : 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    SUPPORTED_PENALTIES = ["smooth-lasso", "tvl1"]

    def __init__(self, penalty="smooth-lasso", is_classif=False, alpha=None,
                 alphas=None, l1_ratio=.5, mask=None, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None, t_r=None,
                 max_iter=1000, tol=1e-4, memory=Memory(None), copy_data=True,
                 standardize=False, normalize=False, alpha_min=1e-6, verbose=0,
                 n_jobs=1, n_alphas=10, eps=1e-3, cv=10, fit_intercept=True,
                 screening_percentile=10., debias=False):
        super(SpaceNet, self).__init__()
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.penalty = penalty
        self.is_classif = is_classif
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
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.cv = cv
        self.screening_percentile = screening_percentile
        self.debias = debias

        # sanity check on params
        self.check_params()

    def _binarize_y(self, y):
        """Helper function invoked just before fitting a classifier."""
        y = np.array(y)

        # encode target classes as -1 and 1
        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y

    def _set_coef_and_intercept(self, w):
        self.w_ = np.array(w)
        if self.w_.ndim == 1:
            self.w_ = self.w_[np.newaxis, :]
        self.coef_ = self.w_[:, :-1]
        self.intercept_ = self.w_[:, -1]

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
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """

        X = self.masker_.transform(X)

        # handle regression (least-squared loss)
        if not self.is_classif:
            return LinearModel.predict(self, X)

        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def check_params(self):
        """Makes sure parameters are sane."""
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
        if self.penalty not in self.SUPPORTED_PENALTIES:
            raise ValueError(
                "'penalty' parameter must be one of %s, or %s; got %s" % (
                    ",".join(self.SUPPORTED_PENALTIES[:-1]),
                    self.SUPPORTED_PENALTIES[-1], self.penalty))

    def fit(self, X, y):
        """Fit the learner.

        Parameters
        ----------
        X: list of filenames or NiImages of length n_samples, or 2D array of
           shape (n_samples, n_features)
            Brain images on which the which a structured weights map is to be
            learned. This is the independent variable (e.g gray-matter maps
            from VBM analysis, etc.)

        y: array or list of length n_samples
            The dependent variable (age, sex, QI, etc.)

        Notes
        -----
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
        y = np.array(y).copy().ravel()

        # set backend solver
        if self.penalty == "smooth-lasso":
            if self.is_classif:
                solver = smooth_lasso_logistic
            else:
                solver = smooth_lasso_squared_loss
        else:
            if self.is_classif:
                solver = partial(tvl1_solver, loss="logistic")
            else:
                solver = partial(tvl1_solver, loss="mse")

        # always a good idea to centralize / normalize data in regression
        ymean = 0.
        if self.standardize:
            X, y, Xmean, ymean, Xstd = center_data(
                X, y, copy=True, normalize=self.normalize,
                fit_intercept=self.fit_intercept)

        # make / sanitize alpha grid
        if self.alpha is not None:
            alphas = [self.alpha]
        elif self.alphas is None:
            alphas = _space_net_alpha_grid(
                X, y, l1_ratio=self.l1_ratio, eps=self.eps,
                n_alphas=self.n_alphas, standardize=self.standardize,
                normalize=self.normalize, alpha_min=self.alpha_min,
                fit_intercept=self.fit_intercept, logistic=self.is_classif)
        else:
            alphas = np.array(self.alphas)

        # always sort alphas from largest to smallest
        alphas = np.sort(alphas)[::-1]

        if len(alphas) > 1:
            cv = list(check_cv(self.cv, X=X, y=y, classifier=self.is_classif))
        else:
            cv = [(range(n_samples), [])]  # single fold
        self.n_folds_ = len(cv)

        # misc (different for classifier and regressor)
        if self.is_classif:
            y = self._binarize_y(y)
        if self.is_classif and self.n_classes_ > 2:
            n_problems = self.n_classes_
        else:
            n_problems = 1
            y = y.ravel()
        self.scores_ = [[] for _ in xrange(n_problems)]
        w = np.zeros((n_problems, X.shape[1] + int(self.is_classif > 0)))

        # function handle for generating OVR labels
        _ovr_y = lambda c: y[:, c] if self.is_classif and (self.n_classes_ > 2
                                                           ) else y

        # main loop: loop on classes and folds
        solver_params = dict(tol=self.tol, verbose=self.verbose,
                             max_iter=self.max_iter, rescale_alpha=True)
        for test_scores, best_w, c in Parallel(n_jobs=self.n_jobs)(
            delayed(self.memory_.cache(path_scores))(
                solver, X, _ovr_y(c), self.mask_, alphas, self.l1_ratio,
                train, test, solver_params, is_classif=self.is_classif, key=c,
                debias=self.debias, ymean=ymean,
                screening_percentile=self.screening_percentile
                ) for c in xrange(n_problems) for (train, test) in cv):
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

        # bagging: average best weights maps over folds
        w /= self.n_folds_

        # set coefs and intercepts
        if self.is_classif:
            self._set_coef_and_intercept(w)
        else:
            self.coef_ = w
            if self.standardize:
                self._set_intercept(Xmean, ymean, Xstd)
            else:
                self.intercept_ = 0.

        # special treatment for non classif (i.e regression) model
        if not self.is_classif:
            self.coef_ = self.coef_[0]
            self.scores_ = self.scores_[0]

        # unmask weights map as a niimg
        self.coef_img_ = self.masker_.inverse_transform(self.coef_)

        # report time elapsed
        if self.verbose:
            print "Time Elapsed: %g seconds."  % (time.time() - tic)

        return self
