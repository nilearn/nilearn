"""
TODOC
"""
import warnings
import numpy as np
from scipy import linalg


def normalize_matrix_on_axis(m, axis=0):
    """ Normalize a 2D matrix on an axis.

    Parameters
    ----------
    m : numpy 2D array,
      The matrix to normalize.
    axis : integer in {0, 1}, optional
      A valid axis to normalize across.

    Returns
    -------
    ret : numpy array, shape = m.shape
      The normalized matrix

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     normalize_matrix_on_axis)
    >>> X = np.array([[0, 4], [1, 0]])
    >>> normalize_matrix_on_axis(X)
    array([[ 0.,  1.],
           [ 1.,  0.]])
    >>> normalize_matrix_on_axis(X, axis=1)
    array([[ 0.,  1.],
           [ 1.,  0.]])

    """
    if m.ndim > 2:
        raise ValueError('This function only accepts 2D arrays. '
                         'An array of shape %r was passed.' % m.shape)

    if axis == 0:
        # array transposition preserves the contiguity flag of that array
        ret = (m.T / np.sqrt(np.sum(m ** 2, axis=0))[:, np.newaxis]).T
    elif axis == 1:
        ret = normalize_matrix_on_axis(m.T).T
    else:
        raise ValueError('axis(=%d) out of bounds' % axis)
    return ret


def orthonormalize_matrix(m, tol=1.e-12):
    """ Orthonormalize a matrix.

    Uses a Singular Value Decomposition.

    Parameters
    ----------
    m : numpy array,
      The matrix to orthonormalize.

    Returns
    -------
    ret : numpy array, shape = m.shape
      The orthonormalized matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     orthonormalize_matrix)
    >>> X = np.array([[1, 0], [0, 1], [1, 1]])
    >>> orthonormalize_matrix(X)
    array([[ -4.08248290e-01,   7.07106781e-01],
           [ -4.08248290e-01,  -7.07106781e-01],
           [ -8.16496581e-01,  -1.11022302e-16]])
    >>> X = np.array([[0, 1], [4, 0]])
    >>> orthonormalize_matrix(X)
    array([[ 0., -1.],
           [-1.,  0.]])

    """
    U, s, _ = linalg.svd(m, full_matrices=False)
    n_eig = np.count_nonzero(s > tol)
    return np.ascontiguousarray(U[:, :n_eig])


def f_score(vars1, vars2, covars=None, lost_dof=0,
             normalized_design=True):
    """Compute F-score associated with the regression of vars2 against vars1

    Covariates are taken into account (if not None).
    The normalized_design case corresponds to the following assumptions:
    - vars1 and vars2 are normalized
    - covars are orthonormalized
    - vars1 and covars are orthogonal (np.dot(vars1.T, covars) == 0)

    Parameters
    ----------
    vars1: array-like, shape=(n_samples, n_var1)
      Explanatory variates
    vars2: array-like, shape=(n_samples, n_var2)
      Targets variates. F-ordered for efficient computation.
    covars, array-like, shape=(n_samples, n_covars) or None
      Confounding variates.
    lost_dof: int, >= 0
      Lost degrees of freedom
    normalized_design: bool,
      Specify whether the variates have been normalized and orthogonalized
      with respect to each other. In such a case, the computation is simpler
      and a lot more efficient.

    Returns
    -------
    score: numpy.ndarray, shape=(n_var2, n_var1)
      F-scores associated with the tests of each explanatory variate against
      each target variate (in the presence of covars).

    """
    if not normalized_design:  # not efficient, added for code exhaustivity
        # normalize variates
        vars1_normalized = normalize_matrix_on_axis(vars1)
        vars2_normalized = normalize_matrix_on_axis(vars2)
        if covars is not None:
            # orthonormalize covariates
            covars_orthonormed = orthonormalize_matrix(covars)
            updated_lost_dof = covars_orthonormed.shape[1]
            # orthogonalize vars1 with respect to covars
            beta_vars1_covars = np.dot(
                vars1_normalized.T, covars_orthonormed)
            vars1_resid_covars = vars1_normalized.T - np.dot(
                beta_vars1_covars, covars_orthonormed.T)
            vars1_normalized = normalize_matrix_on_axis(
                vars1_resid_covars, axis=1).T
        else:
            covars_orthonormed = None
            updated_lost_dof = 0
        return f_score(vars1_normalized, vars2_normalized, covars_orthonormed,
                        updated_lost_dof, normalized_design=True)
    else:  # efficient, should be used everytime with permuted OLS
        dof = vars2.shape[0] - 1 - lost_dof
        beta_vars2_vars1 = np.dot(vars2.T, vars1)
        b2 = beta_vars2_vars1 ** 2
        if covars is None:
            rss = (1 - b2)
        else:
            beta_vars2_covars = np.dot(vars2.T, covars)
            a2 = np.sum(beta_vars2_covars ** 2, 1)
            rss = (1 - a2[:, np.newaxis] - b2)
        score = b2 / rss
        score *= dof
        return np.asfortranarray(score)


def orthogonalize_design(tested_vars, target_vars, confounding_vars=None):
    """

    """
    if confounding_vars is not None:
        # step 1: extract effect of covars from target vars
        covars_orthonormed = orthonormalize_matrix(confounding_vars)
        if not covars_orthonormed.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Confounding variates not C_CONTIGUOUS.')
            covars_orthonormed = np.ascontiguousarray(covars_orthonormed)
        targetvars_normalized = normalize_matrix_on_axis(
            target_vars).T  # faster with F-ordered target_vars_chunk
        if not targetvars_normalized.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Target variates not C_CONTIGUOUS.')
            targetvars_normalized = np.ascontiguousarray(targetvars_normalized)
        beta_targetvars_covars = np.dot(targetvars_normalized,
                                        covars_orthonormed)
        targetvars_resid_covars = targetvars_normalized - np.dot(
            beta_targetvars_covars, covars_orthonormed.T)
        targetvars_resid_covars = normalize_matrix_on_axis(
            targetvars_resid_covars, axis=1)
        lost_dof = covars_orthonormed.shape[1]
        # step 2: extract effect of covars from tested vars
        testedvars_normalized = normalize_matrix_on_axis(tested_vars.T, axis=1)
        beta_testedvars_covars = np.dot(testedvars_normalized,
                                        covars_orthonormed)
        testedvars_resid_covars = testedvars_normalized - np.dot(
            beta_testedvars_covars, covars_orthonormed.T)
        testedvars_resid_covars = normalize_matrix_on_axis(
            testedvars_resid_covars, axis=1).T.copy()
    else:
        targetvars_resid_covars = normalize_matrix_on_axis(target_vars).T
        testedvars_resid_covars = normalize_matrix_on_axis(tested_vars).copy()
        covars_orthonormed = None
        lost_dof = 0
    # check arrays contiguousity (for the sake of code efficiency)
    if not targetvars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Target variates not C_CONTIGUOUS.')
        targetvars_resid_covars = np.ascontiguousarray(targetvars_resid_covars)
    if not testedvars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Tested variates not C_CONTIGUOUS.')
        testedvars_resid_covars = np.ascontiguousarray(testedvars_resid_covars)

    orthogonalized_design = (testedvars_resid_covars,
                             targetvars_resid_covars.T,
                             covars_orthonormed, lost_dof)
    return orthogonalized_design
