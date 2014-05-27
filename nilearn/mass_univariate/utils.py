"""
Utility functions for matrices and designs transformation.

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
    >>> from nilearn.mass_univariate.utils import normalize_matrix_on_axis
    >>> X = np.array([[0, 4], [1, 0]])
    >>> normalize_matrix_on_axis(X)
    array([[ 0.,  1.],
           [ 1.,  0.]])
    >>> normalize_matrix_on_axis(X, axis=1)
    array([[ 0.,  1.],
           [ 1.,  0.]])

    """
    if m.ndim != 2:
        raise ValueError('This function only accepts 2D arrays. '
                         'An array of shape %r was passed.' % (m.shape,))

    if axis == 0:
        sum_of_squares = np.sum(m ** 2, axis=0)
        n_zeros_column = np.sum(sum_of_squares == 0)
        if n_zeros_column > 0:
            raise ValueError('The matrix cannot be normalized because it has'
                             '%d zeros column%s.'
                             % (n_zeros_column,
                                "s" if n_zeros_column > 1 else ""))
        # The following line can create Nans/Infs, but this is supposed to
        # have been handled at a higher level.
        # array transposition preserves the contiguity flag of that array
        ret = (m.T / np.sqrt(np.sum(m ** 2, axis=0))[:, np.newaxis]).T
    elif axis == 1:
        # array transposition preserves the contiguity flag of that array
        ret = normalize_matrix_on_axis(m.T).T
    else:
        raise ValueError('axis(=%d) out of bounds' % axis)
    return ret


def orthonormalize_matrix(m, tol=1.e-12):
    """ Orthonormalize a matrix.

    Uses a Singular Value Decomposition.
    If the input matrix is rank-deficient, then its shape is cropped.

    Parameters
    ----------
    m : array-like,
      The matrix to orthonormalize.

    Returns
    -------
    ret : np.ndarray, shape = m.shape
      The orthonormalized matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.utils import orthonormalize_matrix
    >>> X = np.array([[1, 2], [0, 1], [1, 1]])
    >>> orthonormalize_matrix(X)
    array([[-0.81049889, -0.0987837 ],
           [-0.31970025, -0.75130448],
           [-0.49079864,  0.65252078]])
    >>> X = np.array([[0, 1], [4, 0]])
    >>> orthonormalize_matrix(X)
    array([[ 0., -1.],
           [-1.,  0.]])

    """
    U, s, _ = linalg.svd(m, full_matrices=False)
    n_eig = np.count_nonzero(s > tol)
    return np.ascontiguousarray(U[:, :n_eig])


def orthogonalize_design(tested_vars, target_vars, confounding_vars=None):
    """Orthogonalize a design (tested, target, confounding variates)

    - tested_vars and target_vars are normalized
    - covars_orthonormalized are orthonormalized
    - tested_vars and covars_orthonormalized are made orthogonal
      (np.dot(tested_vars.T, covars) == 0)

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_tested_vars)
      Explanatory variates, fitted and tested independently from each others.

    target_vars: array-like, shape=(n_samples, n_target_vars)
      Target variates to be explained by explanatory and confounding variates.

    confounding_vars: array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
      If None (default), no confounding variate is added to the model.

    Returns
    -------
    tested_vars_resid_covars: np.ndarray, shape=(n_samples, n_tested_vars)
      Normalized tested variates, from which the effect of the covariates
      has been removed.

    target_vars_resid_covars: np.ndarray, shape=(n_samples, n_target_vars)
      Normalized target variates, from which the effect of the covariates
      has been removed.

    covars_orthonormed: np.ndarray, shape=(n_samples, n_covars)
      Confounding variates (covariates), orthonormalized.

    lost_dof: int,
      Degress of freedom that are lost during the model estimation.
      Note that the tested variates are to be fitted independently so
      their number does not impact the value of `lost_dof`.

    """
    if confounding_vars is not None:
        # step 1: extract effect of covars from target vars
        covars_orthonormed = orthonormalize_matrix(confounding_vars)
        if not covars_orthonormed.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Confounding variates not C_CONTIGUOUS.')
            covars_orthonormed = np.ascontiguousarray(covars_orthonormed)
        target_vars_normalized = normalize_matrix_on_axis(
            target_vars).T  # faster with F-ordered target_vars_chunk
        if not target_vars_normalized.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Target variates not C_CONTIGUOUS.')
            target_vars_normalized = np.ascontiguousarray(
                target_vars_normalized)
        beta_target_vars_covars = np.dot(target_vars_normalized,
                                        covars_orthonormed)
        target_vars_resid_covars = target_vars_normalized - np.dot(
            beta_target_vars_covars, covars_orthonormed.T)
        target_vars_resid_covars = normalize_matrix_on_axis(
            target_vars_resid_covars, axis=1)
        lost_dof = covars_orthonormed.shape[1]
        # step 2: extract effect of covars from tested vars
        tested_vars_normalized = normalize_matrix_on_axis(tested_vars.T,
                                                          axis=1)
        beta_tested_vars_covars = np.dot(tested_vars_normalized,
                                         covars_orthonormed)
        tested_vars_resid_covars = tested_vars_normalized - np.dot(
            beta_tested_vars_covars, covars_orthonormed.T)
        tested_vars_resid_covars = normalize_matrix_on_axis(
            tested_vars_resid_covars, axis=1).T.copy()
    else:
        target_vars_resid_covars = normalize_matrix_on_axis(target_vars).T
        tested_vars_resid_covars = normalize_matrix_on_axis(tested_vars).copy()
        covars_orthonormed = None
        lost_dof = 0
    # check arrays contiguousity (for the sake of code efficiency)
    if not target_vars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Target variates not C_CONTIGUOUS.')
        target_vars_resid_covars = np.ascontiguousarray(
            target_vars_resid_covars)
    if not tested_vars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Tested variates not C_CONTIGUOUS.')
        tested_vars_resid_covars = np.ascontiguousarray(
            tested_vars_resid_covars)

    orthogonalized_design = (tested_vars_resid_covars,
                             target_vars_resid_covars.T,
                             covars_orthonormed, lost_dof)
    return orthogonalized_design


def t_score_with_covars_and_normalized_design(tested_vars, target_vars,
                                               covars_orthonormalized=None):
    """t-score in the regression of tested variates against target variates

    Covariates are taken into account (if not None).
    The normalized_design case corresponds to the following assumptions:
    - tested_vars and target_vars are normalized
    - covars_orthonormalized are orthonormalized
    - tested_vars and covars_orthonormalized are orthogonal
      (np.dot(tested_vars.T, covars) == 0)

    Parameters
    ----------
    tested_vars : array-like, shape=(n_samples, n_tested_vars)
      Explanatory variates.

    target_vars : array-like, shape=(n_samples, n_target_vars)
      Targets variates. F-ordered is better for efficient computation.

    covars_orthonormalized : array-like, shape=(n_samples, n_covars) or None
      Confounding variates.

    Returns
    -------
    score : numpy.ndarray, shape=(n_target_vars, n_tested_vars)
      t-scores associated with the tests of each explanatory variate against
      each target variate (in the presence of covars).

    """
    if covars_orthonormalized is None:
        lost_dof = 0
    else:
        lost_dof = covars_orthonormalized.shape[1]
    # Tested variates are fitted independently,
    # so lost_dof is unrelated to n_tested_vars.
    dof = target_vars.shape[0] - lost_dof
    beta_target_vars_tested_vars = np.dot(target_vars.T, tested_vars)
    if covars_orthonormalized is None:
        rss = (1 - beta_target_vars_tested_vars ** 2)
    else:
        beta_target_vars_covars = np.dot(target_vars.T, covars_orthonormalized)
        a2 = np.sum(beta_target_vars_covars ** 2, 1)
        rss = (1 - a2[:, np.newaxis] - beta_target_vars_tested_vars ** 2)
    return beta_target_vars_tested_vars * np.sqrt((dof - 1.) / rss)
