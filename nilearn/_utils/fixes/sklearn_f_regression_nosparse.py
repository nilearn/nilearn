import numpy as np
from scipy import stats
from sklearn.utils import check_arrays
from sklearn.utils.extmath import norm


# f_regression with correct degrees of freedom when center=False
# available is sklearn version >= 0.15
# This version does not support sparse matrices and is used to have tests
# passing for versions of sklearn < 0.12.
def f_regression_nosparse(X, y, center=True):
    """Univariate linear regression tests

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 3 steps:
    1. the regressor of interest and the data are orthogonalized
       with respect to constant regressors
    2. the cross correlation between data and regressors is computed
    3. it is converted to an F score then to a p-value

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    center : True, bool,
        If true, X and y will be centered.

    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.
    """
    X, y = check_arrays(X, y, dtype=np.float)
    y = y.ravel()
    if center:
        y = y - np.mean(y)
        X = X.copy('F')  # faster in fortran
        X -= X.mean(axis=0)

    # compute the correlation
    corr = np.dot(y, X)
    # XXX could use corr /= row_norms(X.T) here, but the test doesn't pass
    corr /= np.asarray(np.sqrt((X ** 2).sum(axis=0))).ravel()
    corr /= norm(y)

    # convert to p-value
    degrees_of_freedom = y.size - (2 if center else 1)
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)
    return F, pv
