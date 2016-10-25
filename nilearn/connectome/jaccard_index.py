import numpy as np

from sklearn.externals.joblib import delayed, Parallel
from nilearn.image import load_img
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
    """ Compute Jaccard similarity coefficient between two matrices

    Parameters
    ----------
    X, Y : numpy array or niftii images
        The atlases to be compared

    Returns
    -------
    jaccard : float
        The similarity coefficient, between 0 and 1.

    """
    # Check if entry is a niftii image and if so, convert it
    try:
        X = load_img(X).get_data()
        Y = load_img(Y).get_data()
    except TypeError:
        # If variables are not numpy arrays either, raise an error
        if not (type(X).__module__ == np.__name__ and
                type(Y).__module__ == np.__name__):
            raise TypeError
        pass

    nx = X.shape[0]
    ny = Y.shape[0]

    if nx == 0 or ny == 0:
        return 0.

    jaccard = 0.
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_jaccard_distance)(X[i], Y[j])
        for i, j in product(range(nx), range(ny))
    )
    jaccard = np.sum(scores)
    jaccard = jaccard / (nx * ny / 2.)

    return jaccard
