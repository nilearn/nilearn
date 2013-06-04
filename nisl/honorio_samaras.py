
import numpy as np


def honorio(rho, covariances, n_samples, n_iter=10):
    """
    Parameters
    ==========
    rho: float
        regularization parameter

    covariances: list of numpy.ndarray
        covariance estimates. All shapes must be identical. Must be positive
        semidefinite.

    n_samples: list of int
        number of samples for each task. len(n_samples) == len(sigma)

    n_iter: int
        number of iteration
    """

    # Check that all covariances have the same size.

    # TODO: low robustness with inverse computation.
    omegas = [np.diag(1. / np.diag(covariance)) for covariance in covariances]
    n_var = covariances[0].shape[0]

    # Initial state: remove first col/row
    Ws = [omega[1:, 1:] for omega in omegas]
    ys = [omega[1:, 0] for omega in omegas]
    zs = [omega[0, 0] for omega in omegas]
    for n in xrange(n_iter):
        for v in xrange(n_var):
            pass


def update_submatrix(full, sub, sub_inv, n):
    """Update submatrix and its inverse.

    sub_inv is the inverse of the submatrix of "full" obtained by removing
    the n-th row and column.

    sub_inv is modified in-place. After execution of this function, it contains
    the inverse of the submatrix of "full" obtained by removing the n+1-th row
    and column.

    This computation is based on Sherman-Woodbury-Morrison identity.
    """

    h, v = update_vectors(full, n)

    # change row
    coln = sub_inv[:, n]  # A^(-1)*U
    V = h - sub[n, :]
    coln = coln / (1. + np.dot(V, coln))
    sub_inv -= np.outer(coln, np.dot(V, sub_inv))
    sub[n, :] = h

    # change column
    rown = sub_inv[n, :]  # V*A^(-1)
    U = v - sub[:, n]
    rown = rown / (1. + np.dot(rown, U))
    sub_inv -= np.outer(np.dot(sub_inv, U), rown)
    sub[:, n] = v   # equivalent to sub[n, :] += U


def test_update_submatrix():
    N = 5
    rand_gen = np.random.RandomState(0)
    M = rand_gen.randn(N, N)
    sub = M[1:, 1:].copy()
    sub_inv = np.linalg.inv(M[1:, 1:])
    true_sub = np.zeros(sub.shape, dtype=sub.dtype)
    true_inv = sub_inv.copy()

    for n in xrange(1, N):
        update_submatrix(M, sub, sub_inv, n - 1)

        true_sub[n:, n:] = M[n + 1:, n + 1:]
        true_sub[:n, :n] = M[:n, :n]
        true_sub[n:, :n] = M[n + 1:, :n]
        true_sub[:n, n:] = M[:n, n + 1:]
        true_inv = np.linalg.inv(true_sub)
        np.testing.assert_almost_equal(true_sub, sub)
        np.testing.assert_almost_equal(true_inv, sub_inv)


def update_vectors(full, n):
    """full is a (N, N) matrix.

    This function is a helper function for updating the submatrix equals to
    "full" with row n + 1 and column n + 1 removed. The initial state of the
    submatrix is supposed to be "full" with row and column n removed.

    This functions returns the new value of row and column n in the submatrix.
    Thus, if h, v are the return values of this function, the submatrix must
    be updated this way: sub[n, :] = h ; sub[:, n] = v
    """
    v = np.ndarray((full.shape[0] - 1,), dtype=full.dtype)
    v[:n + 1] = full[:n + 1, n]
    v[n + 1:] = full[n + 2:, n]

    h = np.ndarray((full.shape[1] - 1,), dtype=full.dtype)
    h[:n + 1] = full[n, :n + 1]
    h[n + 1:] = full[n, n + 2:]

    return h, v


if __name__ == "__main__":
    test_update_submatrix_inverse()
