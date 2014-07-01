"""
Ninja tricks (early stopping, etc.) to make CV a better place to live...

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gael Varoquaux,
#         Michael Eickenberg,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from functools import partial
import numpy as np
from scipy import stats
from ..._utils.fixes import center_data
from sklearn.feature_selection import (
    f_regression, f_classif, SelectPercentile)

np_version = []
for x in np.__version__.split('.'):
    try:
        np_version.append(int(x))
    except ValueError:
        # x may be of the form dev-1ea1592
        np_version.append(x)
np_version = tuple(np_version)

# Newer NumPy has a ravel that needs less copying.
if np_version < (1, 7, 1):
    _ravel = np.ravel
else:
    _ravel = partial(np.ravel, order='K')


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


class _BaseFeatureSelector(object):
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
    """ Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X: 2d array, shape (n_samples, n_features)
        Training data (design matrix).

    y: ndarray, shape (n_samples,)
        Target / response vector.

    l1_ratio: float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.

    eps: float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas: int, optional
        Number of alphas along the regularization path.

    fit_intercept: bool
        Fit or not an intercept.

    normalize: boolean, optional, default False
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
