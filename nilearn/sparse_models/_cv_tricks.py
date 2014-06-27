"""
Ninja tricks (early stopping, etc.) to make CV a better place to live...

"""

from functools import partial
from math import sqrt
import numpy as np
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
    """ Out-of-bag early stopping

        A callable that returns True when the test error starts
        rising. We use a regression loss, because it is more sensitive to
        improvements than a classification loss
    """

    a = 1

    def __init__(self, X_test, y_test, X_std, verbose=False):
        n_test_samples = len(y_test)
        self.X_test = np.reshape(X_test, (n_test_samples, -1))
        self.scaled_y_test = y_test - y_test.mean()
        y_norm = sqrt(_squared_norm(self.scaled_y_test))
        if y_norm != 0 and np.isfinite(y_norm):
            self.scaled_y_test /= y_norm
        self.X_std = np.ravel(X_std)
        self.test_errors = list()
        self.verbose = verbose

    def __call__(self, variables):
        i = variables['counter']
        if i == 0:
            # Reset the test_errors list
            self.test_errors = list()
        w = variables['w']
        w = np.ravel(w)
        # Correlation to output
        y_pred = np.dot(self.X_test, w)
        y_pred -= y_pred.mean()
        if np.any(y_pred != 0):
            y_pred /= sqrt(_squared_norm(y_pred))
        error = .5 * (1 - np.dot(self.scaled_y_test, y_pred))
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

    def unmask(self, w):
        """Unmasks the input vector `w`, according to the mask learned by the
        ANOVA.

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

    def unmask(self, w):
        """Unmasks the input vector `w`, according to the mask learned by the
        ANOVA.

        """

        if self.support_.sum() < len(self.support_):
            w_ = np.zeros(len(self.support_))
            w_ = np.append(w_, w[-1])
            w_[:-1][self.support_] = w[:-1]
        else:
            w_ = w

        return w_
