"""
Massively Univariate Linear Model estimated with OLS and permutation test.

"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
#         Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import warnings
import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state, gen_even_slices
import sklearn.externals.joblib as joblib


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


def _f_score(vars1, vars2, covars=None, lost_dof=0,
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
        return _f_score(vars1_normalized, vars2_normalized, covars_orthonormed,
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
        return score


def _permuted_ols_on_chunk(tested_vars, target_vars, confounding_vars=None,
                           n_perm_chunk=10000, lost_dof=0,
                           intercept_test=True, random_state=None):
    """Massively univariate group analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variates.
    target_vars: array-like, shape=(n_samples, n_targets)
      fMRI data. F-ordered for efficient computations.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Clinical data (covariates).
    n_perm_chunk: int,
      Number of permutations to be performed.
    intercept_test: boolean,
      Change the permutation scheme (swap signs for intercept,
      switch labels otherwise). See [1]
    random_state: int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.

    Returns
    -------
    h0_fmax_part: array-like, shape=(n_perm_chunk, )
      Distribution of the (max) F-statistic under the null hypothesis
      (limited to this permutation chunk).

    References
    ----------
    [1] Fisher, R. A. (1935). The design of experiments.

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    n_samples, n_regressors = tested_vars.shape

    # do the permutations
    h0_fmax_part = np.empty((n_perm_chunk, n_regressors))
    for i in xrange(n_perm_chunk):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            target_vars = (target_vars
                           * rng.randint(2, size=(1, n_samples)) * 2 - 1)
        else:
            # shuffle data
            # Regarding computation costs, we choose to shuffle testvars
            # and covars rather than fmri_signal.
            # Also, it is important to shuffle testedvars and covars
            # jointly to simplify f_score computation (null dot product).
            shuffle_idx = rng.permutation(n_samples)
            #rng.shuffle(shuffle_idx)
            tested_vars = tested_vars[shuffle_idx]
            if confounding_vars is not None:
                confounding_vars = confounding_vars[shuffle_idx]

        # OLS regression on randomized data
        perm_scores = _f_score(tested_vars, target_vars, confounding_vars,
                               lost_dof, normalized_design=True)
        h0_fmax_part[i] = np.amax(perm_scores, 0)

    return h0_fmax_part


def _convert_to_pvalues(h0_distribution, scores_to_convert):
    """Convert a statistic into p-values using its empirical distribution

    Parameters
    ----------
    h0_distribution: array-like, shape=(n_realizations, )
      The empirical distribution of the statistic to be converted into
      p-values.
    scores_to_convert: array-like, shape=(n_scores, )
      The values of the statistic to be converted into p-values according to
      their rank in h0_distribution.

    Returns
    -------
    pvals: numpy.array, shape=(n_scores, )
      Negative log10-pvalues corresponding to the scores confronted to their
      distribution (h0_distribution).

    """
    pvals = ((h0_distribution.size + 1
              - np.searchsorted(np.sort(h0_distribution), scores_to_convert))
             / float(h0_distribution.size + 1))
    return - np.log10(pvals)


def permuted_ols(tested_vars, target_vars, confounding_vars=None,
                 model_intercept=True, n_perm=10000,
                 random_state=None, n_jobs=1):
    """Massively univariate group analysis with permuted OLS.

    Tested variates are independently fitted to target variates descriptors
    (e.g. brain imaging signal) according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variates may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variates and the target variates. A max-type
    procedure is used to obtain family-wise corrected p-values.

    Permutations are performed on parallel computing units. Each of them
    performs a fraction of permutations on the whole dataset. Thus, the max
    F-score amongst data descriptors can be computed directly, which avoids
    storing all the computed F-scores.

    The variates should be given C-contiguous. target_vars are fortran-ordered
    automatically to speed-up computations.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variates, fitted and tested independently from each others.
    target_vars: array-like, shape=(n_samples, n_descriptors)
      fMRI data, trying to be explained by explanatory and confounding
      variates.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
      If None, no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)
    model_intercept: bool,
      If True, a constant column is added to the confounding variates
      unless the tested variate is already the intercept.
    n_perm: int,
      Number of permutations to perform.
      Permutations are costly but the more are performed, the more precision
      one gets in the p-values estimation.
    random_state: int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.
    n_jobs: int,
      Number of parallel workers.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      must be used.

    Returns
    -------
    pvals: array-like, shape=(n_regressors, n_descriptors)
      Negative log10 p-values associated with the significance test of the
      n_regressors explanatory variates against the n_descriptors target
      variates. Family-wise corrected p-values.
    score_orig_data: numpy.ndarray, shape=(n_regressors, n_descriptors)
      F-statistic associated with the significance test of the n_regressors
      explanatory variates against the n_descriptors target variates.
      The ranks of the scores into the h0 distribution correspond to the
      p-values.
    h0_fmax: array-like, shape=(n_perm, )
      Distribution of the (max) F-statistic under the null hypothesis
      (obtained from the permutations). Array is sorted.

    References
    ----------
    [1] Anderson, M. J., & Robinson, J. (2001).
        Permutation tests for linear models.
        Australian & New Zealand Journal of Statistics, 43(1), 75-88.

    """
    # check n_jobs (number of CPUs)
    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError("'n_jobs == 0' is not a valid choice. "
                         "Please provide a positive number of CPUs, or -1 "
                         "for all CPUs, or a negative number (-i) for "
                         "'all but (i-1)' CPUs (joblib conventions).")
    elif n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() - int(n_jobs) + 1)
    else:
        n_jobs = min(n_jobs, joblib.cpu_count())
    # make target_vars F-ordered to speed-up computation
    if target_vars.ndim != 2:
        raise ValueError("'target_vars' should be a 2D array. "
                         "An array with %d dimension%s was passed"
                         % (target_vars.ndim,
                            "s" if target_vars.ndim > 1 else ""))
    target_vars = np.asfortranarray(target_vars)  # efficient for chunking
    n_descriptors = target_vars.shape[1]

    # check explanatory variates dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T
    n_samples, n_regressors = tested_vars.shape

    # check if explanatory variates is intercept (constant) or not
    if (n_regressors == 1 and np.unique(tested_vars).size == 1):
        intercept_test = True
    else:
        intercept_test = False

    # optionally add intercept
    if model_intercept and not intercept_test:
        if confounding_vars is not None:
            confounding_vars = np.hstack(
                (confounding_vars, np.ones((n_samples, 1))))
        else:
            confounding_vars = np.ones((n_samples, 1))

    ### OLS regression on original data
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
    # step 3: original regression (= regression on residuals + adjust F score)
    # compute F score for original data
    scores_original_data = _f_score(
        testedvars_resid_covars, targetvars_resid_covars.T, covars_orthonormed,
        lost_dof, normalized_design=True)

    ### Permutations
    # parallel computing units perform a reduced number of permutations each
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)
        n_perm_chunks[-1] += n_perm % n_jobs
    elif n_perm > 0:
        n_perm_chunks = np.ones(n_perm, dtype=int)
    else:  # 0 or negative number of permutations => original data scores only
        return np.asarray([]), scores_original_data,  np.asarray([])

    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_permuted_ols_on_chunk)
          (testedvars_resid_covars, targetvars_resid_covars.T,
           covars_orthonormed, n_perm_chunk=n_perm_chunk, lost_dof=lost_dof,
           intercept_test=intercept_test, random_state=0)
          for n_perm_chunk in n_perm_chunks)
    # reduce resultsx
    h0_fmax = np.sort(np.ravel(np.concatenate(ret)))
    # convert scores into negative log10 p-values
    # TODO: to speed this up, we could threshold scores_original_data
    n_scores = n_descriptors * n_regressors
    ravelized_scores = np.ravel(scores_original_data)
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_convert_to_pvalues)
          (h0_fmax, ravelized_scores[chunk])
          for chunk in gen_even_slices(
            n_scores + 1, min(n_scores, n_jobs)))
    pvals = np.concatenate(ret).reshape((n_regressors, n_descriptors))

    return pvals, scores_original_data, h0_fmax
