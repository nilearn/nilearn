"""Process diffusion imaging parameters

* ``q`` is a vector in Q space
* ``b`` is a b value
* ``g`` is the unit vector along the direction of q (the gradient
  direction)

Thus:

   b = norm(q)

   g = q  / norm(q)

(``norm(q)`` is the Euclidean norm of ``q``)

The B matrix ``B`` is a symmetric positive semi-definite matrix.  If
``q_est`` is the closest q vector equivalent to the B matrix, then:

   B ~ (q_est . q_est.T) / norm(q_est)
"""

import numpy as np
import numpy.linalg as npl


def B2q(B, tol=None):
    """Estimate q vector from input B matrix `B`

    We require that the input `B` is symmetric positive definite.

    Because the solution is a square root, the sign of the returned
    vector is arbitrary.  We set the vector to have a positive x
    component by convention.

    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.
    tol : None or float
       absolute tolerance below which to consider eigenvalues of the B
       matrix to be small enough not to worry about them being negative,
       in check for positive semi-definite-ness.  None (default) results
       in a fairly tight numerical threshold proportional to the maximum
       eigenvalue

    Returns
    -------
    q : (3,) vector
       Estimated q vector from B matrix `B`
    """
    B = np.asarray(B)
    if not np.allclose(B - B.T, 0):
        raise ValueError('B matrix is not symmetric enough')
    w, v = npl.eigh(B)
    if tol is None:
        tol = np.abs(w.max()) * B.shape[0] * np.finfo(w.dtype).eps
    non_trivial = np.abs(w) > tol
    if np.any(w[non_trivial] < 0):
        raise ValueError('B not positive semi-definite')
    inds = np.argsort(w)[::-1]
    max_ind = inds[0]
    vector = v[:, max_ind]
    # because the factor is a sqrt, the sign of the vector is arbitrary.
    # We arbitrarily set it to have a positive x value.
    if vector[0] < 0:
        vector *= -1
    return vector * w[max_ind]


def nearest_pos_semi_def(B):
    """Least squares positive semi-definite tensor estimation

    Reference: Niethammer M, San Jose Estepar R, Bouix S, Shenton M,
    Westin CF.  On diffusion tensor estimation. Conf Proc IEEE Eng Med
    Biol Soc.  2006;1:2622-5. PubMed PMID: 17946125; PubMed Central
    PMCID: PMC2791793.

    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.

    Returns
    -------
    npds : (3,3) array
       Estimated nearest positive semi-definite array to matrix `B`.

    Examples
    --------
    >>> B = np.diag([1, 1, -1])
    >>> nearest_pos_semi_def(B)
    array([[0.75, 0.  , 0.  ],
           [0.  , 0.75, 0.  ],
           [0.  , 0.  , 0.  ]])
    """
    B = np.asarray(B)
    vals, vecs = npl.eigh(B)
    # indices of eigenvalues in descending order
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    cardneg = np.sum(vals < 0)
    if cardneg == 0:
        return B
    if cardneg == 3:
        return np.zeros((3, 3))
    lam1a, lam2a, lam3a = vals
    scalers = np.zeros((3,))
    if cardneg == 2:
        b112 = np.max([0, lam1a + (lam2a + lam3a) / 3.0])
        scalers[0] = b112
    elif cardneg == 1:
        lam1b = lam1a + 0.25 * lam3a
        lam2b = lam2a + 0.25 * lam3a
        if lam1b >= 0 and lam2b >= 0:
            scalers[:2] = lam1b, lam2b
        else:  # one of the lam1b, lam2b is < 0
            if lam2b < 0:
                b111 = np.max([0, lam1a + (lam2a + lam3a) / 3.0])
                scalers[0] = b111
            if lam1b < 0:
                b221 = np.max([0, lam2a + (lam1a + lam3a) / 3.0])
                scalers[1] = b221
    # resort the scalers to match the original vecs
    scalers = scalers[np.argsort(inds)]
    return np.dot(vecs, np.dot(np.diag(scalers), vecs.T))


def q2bg(q_vector, tol=1e-5):
    """Return b value and q unit vector from q vector `q_vector`

    Parameters
    ----------
    q_vector : (3,) array-like
        q vector
    tol : float, optional
        q vector L2 norm below which `q_vector` considered to be `b_value` of
        zero, and therefore `g_vector` also considered to zero.

    Returns
    -------
    b_value : float
        L2 Norm of `q_vector` or 0 if L2 norm < `tol`
    g_vector : shape (3,) ndarray
        `q_vector` / `b_value` or 0 if L2 norma < `tol`

    Examples
    --------
    >>> q2bg([1, 0, 0])
    (1.0, array([1., 0., 0.]))
    >>> q2bg([0, 10, 0])
    (10.0, array([0., 1., 0.]))
    >>> q2bg([0, 0, 0])
    (0.0, array([0., 0., 0.]))
    """
    q_vec = np.asarray(q_vector)
    norm = np.sqrt(np.sum(q_vec * q_vec))
    if norm < tol:
        return (0.0, np.zeros((3,)))
    return norm, q_vec / norm
