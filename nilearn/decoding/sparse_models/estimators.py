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

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.externals.joblib import Memory
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from ..._utils.fixes import LabelBinarizer


class _BaseEstimator(object):
    """
    Parameters
    ----------
    alpha : float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback : callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

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

    def __init__(self, alpha=1., l1_ratio=.5, mask=None, max_iter=1000,
                 tol=1e-4, fit_intercept=True, standardize=False,
                 memory=Memory(None), copy_data=True, normalize=False,
                 verbose=0, callback=None, backtracking=False):
        if not 0. <= l1_ratio <= 1.:
            raise ValueError(("l1_ratio parameter must be in the interval "
                              "[0, 1]; got %g" % l1_ratio))
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


class _BaseRegressor(_BaseEstimator, LinearModel, RegressorMixin):
    """Base regressor class for Smooth Lasso and TVl1.

    Each child must implement a compute_lipschitz_constant(X) method.

    Parameters
    ----------
    alpha : float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV, etc., penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.
        Defaults to 0.5.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    verbose : int, optional (default 0)
        Verbosity level.

    backtracking : bool
        If True, the solver does backtracking in the step size for the proximal
        operator.

    callback : callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    n_jobs : int, optional (default 1)
        Number of jobs to use for One-vs-All classification.

    Attributes
    ----------
    `alpha_` : float
        Best alpha found by cross-validation

    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function (weights map).

    `intercept_` : array, shape = [n_classes-1]
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


class _BaseClassifier(_BaseEstimator, BaseEstimator, LinearClassifierMixin):
    """Base Classifier class for SmoothLasso and TVl1.

    Each child must implement a compute_lipschitz_constant(X) method.

    Parameters
    ----------
    n_jobs : int, optional (default 1)
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

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

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
