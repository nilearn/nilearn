import numpy as np


def tri(mat, k=0):
    return mat[np.triu_indices(mat.shape[0], k=k)]


def untri(vec, k=0, fill=0):
    # solution of n (n + 1) / 2 = len(vec)
    n = (np.sqrt(1 + 8 * len(vec)) - 1) / 2
    n += k
    m = np.empty((n, n))
    m.fill(fill)
    m[np.triu_indices(n, k=k)] = vec
    m.T[np.triu_indices(n, k=k)] = vec
    return m


if __name__ == '__main__':
    a = np.random.random((5, 5))
    # Symetrize a
    a = a + a.T
    # Zero diagonal
    np.fill_diagonal(a, 0)
    # Take the triu
    t = a[np.triu_indices(a.shape[0], k=1)]
    b = untri(t, k=1)
    assert(np.all(a == b))
