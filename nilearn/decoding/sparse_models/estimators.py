"""
sklearn-compatible estimators for TV-l1, S-LASSO, etc. models.

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Virgile Fritsch,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from functools import partial
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.linear_model.base import LinearModel
from ..._utils.fixes import center_data, LinearClassifierMixin, LabelBinarizer
from .common import (compute_mse_lipschitz_constant,
                     compute_logistic_lipschitz_constant)
from .smooth_lasso import smooth_lasso_logistic, smooth_lasso_squared_loss
from .tv import tvl1_solver


class _BaseEstimator(object):
    """
    Parameters
    ----------
    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    Attributes
    ----------
    `alpha_`: float
         Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
         Intercept (a.k.a. bias) added to the decision function.
         It is available only when parameter intercept is set to True.

    `scores_`: 2d array of shape (n_alphas, n_folds)
        Scores (misclassification) for each alpha, and on each fold

    """

    solver = None

    def __init__(self, alpha=1., l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, fit_intercept=True, standardize=False,
                 memory=Memory(None), copy_data=True, normalize=False,
                 verbose=0, callback=None, backtracking=False):
        if not 0. <= l1_ratio <= 1.:
            raise ValueError(("l1_ratio parameter must be in the interval "
                              "[0, 1]; got %g" % l1_ratio))
        if self.solver is None:
            raise RuntimeError(
                "Class attribute `solver` not set in class %s" % (
                    self.__class__.__name__))
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.mask = mask
        self.fit_intercept = fit_intercept
        self.memory = memory
        self.max_iter = max_iter
        self.tol = tol
        self.copy_data = copy_data
        self.verbose = verbose
        self.callback = callback
        self.backtracking = backtracking
        self.standardize = standardize
        self.normalize = normalize

    def _rescale_alpha(self, X):
        """Rescale alpha parameter (= amount of regularization) to handle
        1 / n_samples factor in the model.

        """

        self.alpha_ = self.alpha * X.shape[0]


class _BaseRegressor(_BaseEstimator, LinearModel, RegressorMixin):
    """Base regressor class for Smooth Lasso and TVl1.

    Each child must implement a compute_lipschitz_constant(X) method.

    Parameters
    ----------
    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    Attributes
    ----------
    `alpha_`: float
        Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function (weights map).

    `intercept_`: array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    """

    def __init__(self, alpha=1., l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, normalize=True, standardize=True,
                 memory=Memory(None), copy_data=True,
                 verbose=0, callback=None, backtracking=True,
                 fit_intercept=True):
        super(_BaseRegressor, self).__init__(
            verbose=verbose, copy_data=copy_data, fit_intercept=fit_intercept,
            tol=tol, l1_ratio=l1_ratio, memory=memory, normalize=normalize,
            standardize=standardize)
        self.alpha = alpha
        self.mask = mask
        self.callback = callback
        self.max_iter = max_iter
        self.backtracking = backtracking

    def fit(self, X, y):
        """With dataset (X, y), generates the model that minimizes the
        penalized empirical risk.

        solver is  a callable which can be invoked like so:

            out = solver(X, y, self.alpha_, self.l1_ratio,
                         mask=self.mask, lipschitz_constant=lipschitz_constant,
                         tol=self.tol, max_iter=self.max_iter,
                         verbose=self.verbose, rescale_alpha=False,
                         callback=self.callback)

        For example take a look at tv.tvl1_solver.


        """

        solver = eval(self.solver)

        self._rescale_alpha(X)

        # Preprocessing: standardize (center + normalize)
        if self.standardize:
            X, y, Xmean, ymean, Xstd = self.memory.cache(center_data)(
                X, y, fit_intercept=self.fit_intercept,
                normalize=self.normalize, copy=self.copy_data)

        # Lipschitz constant of gradient loss term
        # Lipschitz constant of gradient f1_grad
        lipschitz_constant = None
        if hasattr(self, "compute_lipschitz_constant"):
            lipschitz_constant = getattr(
                self, "compute_lipschitz_constant")(X)

        # solve the optimization problem
        self.w_, self.objective_, _ = self.memory.cache(solver)(
            X, y, self.alpha_, self.l1_ratio, mask=self.mask,
            lipschitz_constant=lipschitz_constant, tol=self.tol,
            max_iter=self.max_iter, verbose=self.verbose, rescale_alpha=False,
            callback=self.callback, backtracking=self.backtracking)
        self.coef_ = self.w_

        # set intercept (was not fitted in solver)
        if self.standardize:
            self._set_intercept(Xmean, ymean, Xstd)
        else:
            self.intercept_ = 0.

        return self


class _BaseClassifier(_BaseEstimator, BaseEstimator, LinearClassifierMixin):
    """Base Classifier class for SmoothLasso and TVl1.

    Each child must implement a compute_lipschitz_constant(X) method.

    Parameters
    ----------
    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    See _BaseEstimator for the documentation of the the other parameters.

    """

    def __init__(self, alpha=1., l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, memory=Memory(None), copy_data=True,
                 verbose=0, callback=None, backtracking=True,
                 fit_intercept=True, n_jobs=1):
        super(_BaseClassifier, self).__init__(
            verbose=verbose, copy_data=copy_data, fit_intercept=fit_intercept,
            tol=tol, l1_ratio=l1_ratio, memory=memory)
        self.alpha = alpha
        self.mask = mask
        self.callback = callback
        self.max_iter = max_iter
        self.backtracking = backtracking
        self.n_jobs = n_jobs

    def _pre_fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self._rescale_alpha(X)

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

    def fit(self, X, y):
        """With dataset (X, y), generates the model that minimizes the
        logistic penalized empirical risk.

        """

        solver = eval(self.solver)

        X, y = self._pre_fit(X, y)

        # Lipschitz constant of gradient f1_grad
        lipschitz_constant = None
        if hasattr(self, "compute_lipschitz_constant"):
            lipschitz_constant = getattr(
                self, "compute_lipschitz_constant")(X)

        # OVR (One-Versus-Rest) loop
        if self.classes_.size == 2:
            # in the 2 classes case, we don't have to train two
            # classifiers, so the loop is done just once
            w, objective_, _ = self.memory.cache(solver)(
                X, y[:, 0], self.alpha_, self.l1_ratio,
                tol=self.tol, max_iter=self.max_iter,
                mask=self.mask, lipschitz_constant=lipschitz_constant,
                verbose=self.verbose, callback=self.callback,
                backtracking=self.backtracking, rescale_alpha=False)
            self.w_ = w[np.newaxis, :]
            self.objective_ = [objective_]
        else:
            self.w_ = np.ndarray((self.n_classes_, self.n_features_ + 1))
            self.objective_ = {}
            self.time_ = []
            for c, out in enumerate(Parallel(n_jobs=self.n_jobs)(
                    delayed(self.memory.cache(solver))
                    (X, y[:, c], self.alpha_, self.l1_ratio,
                     mask=self.mask, tol=self.tol,
                     max_iter=self.max_iter, rescale_alpha=False,
                     lipschitz_constant=lipschitz_constant,
                     verbose=self.verbose, callback=self.callback,
                     backtracking=self.backtracking)
                    for c in xrange(self.classes_.size))):
                self.w_[c], self.objective_[c], _ = out
            self.objective_ = self.objective_.values()

        # gather coefs and intercepts
        self._set_coef_and_intercept(self.w_)

        return self


class SmoothLassoRegressor(_BaseRegressor):
    """
    Smooth-Lasso regression model with L1 + L2 regularization.

    w = argmin  n_samples^(-1) * || y - X w ||^2 + alpha * l1_ratio ||w||_1
           w      + alpha * (1 - l1_ratio) * ||Gw||^2_2

    where G is the gradient

    n is the number of data points used to learn
    w is the loadings vector

    Parameters
    ----------
    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float
        Constant that mixes L1 and G2 penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    Attributes
    ----------
    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    """

    solver = 'smooth_lasso_squared_loss'


class SmoothLassoClassifier(_BaseClassifier):
    """
    Smooth-Lasso logistic regression model with L1 + L2 regularization.

    w = argmin - (1 / n_samples) * log(sigmoid(y * w.T * X)) +
          w      alpha * (l1_ratio ||w||_1 (1-l1_ratio) * .5 * <Gw, Gw>)
    where G is the spatial gradient operator

    Parameters
    ----------
    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float
        Constant that mixes L1 and G2 penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    Attributes
    ----------
    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    """

    solver = "smooth_lasso_logistic"


class TVl1Regressor(_BaseRegressor):
    """TV-l1 penalized squared-loss regression.

    This object doesn't know how to do cross-validation (the latter is
    solely the user's business).

    The underlying optimization problem is the following:

        w = argmin_w (1 / n_samples) * .5 * (||y - Xw||_2)^2
                + alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_TV)

    Parameters
    ----------
    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    Attributes
    ----------
    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_`: array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    """

    solver = "partial(tvl1_solver, loss='mse')"

    def compute_lipschitz_constant(self, X):
        return 1.05 * compute_mse_lipschitz_constant(X)


class TVl1Classifier(_BaseClassifier):
    """TV-l1 penalized logisitic regression.

    This object doesn't know how to do cross-validation (the latter is
    solely the user's business).

    The underlying optimization problem is the following:

        w = argmin_w -(1 / n_samples) * log(sigmoid(y * w.T * X))
                 +  alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_TV)

    Parameters
    ----------
    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose: int, optional (default 0)
        Verbosity level.

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs: int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    Attributes
    ----------
    `alpha_`: float
        Best alpha found by cross-validation

    `coef_`: array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function (weights map).

    `intercept_`: array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    """

    solver = "partial(tvl1_solver, loss='logistic')"

    def compute_lipschitz_constant(self, X):
        return 1.1 * compute_logistic_lipschitz_constant(X)
