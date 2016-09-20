import numpy as np

from sklearn.externals.joblib import delayed, Parallel
from itertools import product


def _jaccard_distance(x, y):
    """Compute Jaccard distance between two elements """
    x = np.abs(x)
    y = np.abs(y)
    nominator = np.min((x, y), axis=0).sum()
    denominator = np.max((x, y), axis=0).sum()
    if denominator != 0.:
        return nominator / denominator
    return 0


def jaccard_index(X, Y, n_jobs=1):
    """ Compute Jaccard similarity coefficient between two matrices """
    nx = X.shape[0]
    ny = Y.shape[0]

    if nx == 0 or ny == 0:
        return 0.

    s = 0.
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_jaccard_distance)(X[i], Y[j])
        for i, j in product(range(nx), range(ny))
    )
    s = np.sum(scores)
    s = s / (nx * ny / 2.)

    return s
