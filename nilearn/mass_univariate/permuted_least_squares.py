"""
Massively Univariate Linear Model estimated with OLS and permutation test.

"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
#         Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import warnings
import numpy as np
from sklearn.utils import check_random_state
import sklearn.externals.joblib as joblib

from nilearn._utils import check_n_jobs
from .utils import (
    orthogonalize_design, t_score_with_covars_and_normalized_design)


def _permuted_ols_on_chunk(scores_original_data, tested_vars, target_vars,
                           confounding_vars=None, n_perm_chunk=10000,
                           intercept_test=True, two_sided_test=True,
                           random_state=None):
    """Massively univariate group analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    scores_original_data : array-like, shape=(n_descriptors, n_regressors)
      t-scores obtained for the original (non-permuted) data.

    tested_vars : array-like, shape=(n_samples, n_regressors)
      Explanatory variates.

    target_vars : array-like, shape=(n_samples, n_targets)
      fMRI data. F-ordered for efficient computations.

    confounding_vars : array-like, shape=(n_samples, n_covars)
      Clinical data (covariates).

    n_perm_chunk : int,
      Number of permutations to be performed.

    intercept_test : boolean,
      Change the permutation scheme (swap signs for intercept,
      switch labels otherwise). See [1]

    two_sided_test : boolean,
      If True, performs an unsigned t-test. Both positive and negative
      effects are considered; the null hypothesis is that the effect is zero.
      If False, only positive effects are considered as relevant. The null
      hypothesis is that the effect is zero or negative.

    random_state : int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.

    Returns
    -------
    h0_fmax_part : array-like, shape=(n_perm_chunk, )
      Distribution of the (max) t-statistic under the null hypothesis
      (limited to this permutation chunk).

    References
    ----------
    [1] Fisher, R. A. (1935). The design of experiments.

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    n_samples, n_regressors = tested_vars.shape
    n_descriptors = target_vars.shape[1]

    # run the permutations
    h0_fmax_part = np.empty((n_perm_chunk, n_regressors))
    scores_as_ranks_part = np.zeros((n_regressors, n_descriptors))
    for i in xrange(n_perm_chunk):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            target_vars = (target_vars
                           * (rng.randint(2, size=(n_samples, 1)) * 2 - 1))
        else:
            # shuffle data
            # Regarding computation costs, we choose to shuffle testvars
            # and covars rather than fmri_signal.
            # Also, it is important to shuffle tested_vars and covars
            # jointly to simplify t-scores computation (null dot product).
            shuffle_idx = rng.permutation(n_samples)
            tested_vars = tested_vars[shuffle_idx]
            if confounding_vars is not None:
                confounding_vars = confounding_vars[shuffle_idx]

        # OLS regression on randomized data
        perm_scores = np.asfortranarray(
            t_score_with_covars_and_normalized_design(tested_vars,
                                                      target_vars,
                                                      confounding_vars))
        if two_sided_test:
            perm_scores = np.fabs(perm_scores)
        h0_fmax_part[i] = np.amax(perm_scores, 0)
        # find the rank of the original scores in h0_part
        # (when n_descriptors or n_perm are large, it can be quite long to
        #  find the rank of the original scores into the whole H0 distribution.
        #  Here, it is performed in parallel by the workers involded in the
        #  permutation computation)
        scores_as_ranks_part += (h0_fmax_part[i].reshape((-1, 1))
                                 < scores_original_data.T)

    return scores_as_ranks_part, h0_fmax_part.T


def permuted_ols(tested_vars, target_vars, confounding_vars=None,
                 model_intercept=True, n_perm=10000, two_sided_test=True,
                 random_state=None, n_jobs=1, verbose=0):
    """Massively univariate group analysis with permuted OLS.

    Tested variates are independently fitted to target variates descriptors
    (e.g. brain imaging signal) according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variates may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variates and the target variates [1, 2]. A max-type
    procedure is used to obtain family-wise corrected p-values.

    The specific permutation scheme implemented here is the one of
    Freedman & Lane [3]. Its has been demonstrated in [1] that this scheme
    conveys more sensitivity than alternative schemes. This holds for
    neuroimaging applications, as discussed in details in [2].

    Permutations are performed on parallel computing units. Each of them
    performs a fraction of permutations on the whole dataset. Thus, the max
    t-score amongst data descriptors can be computed directly, which avoids
    storing all the computed t-scores.

    The variates should be given C-contiguous. target_vars are fortran-ordered
    automatically to speed-up computations.

    Parameters
    ----------
    tested_vars : array-like, shape=(n_samples, n_regressors)
      Explanatory variates, fitted and tested independently from each others.

    target_vars : array-like, shape=(n_samples, n_descriptors)
      fMRI data, trying to be explained by explanatory and confounding
      variates.

    confounding_vars : array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
      If None, no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)

    model_intercept : bool,
      If True, a constant column is added to the confounding variates
      unless the tested variate is already the intercept.

    n_perm : int,
      Number of permutations to perform.
      Permutations are costly but the more are performed, the more precision
      one gets in the p-values estimation.

    two_sided_test : boolean,
      If True, performs an unsigned t-test. Both positive and negative
      effects are considered; the null hypothesis is that the effect is zero.
      If False, only positive effects are considered as relevant. The null
      hypothesis is that the effect is zero or negative.

    random_state : int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.

    n_jobs : int,
      Number of parallel workers.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      will be used.

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    pvals : array-like, shape=(n_regressors, n_descriptors)
      Negative log10 p-values associated with the significance test of the
      n_regressors explanatory variates against the n_descriptors target
      variates. Family-wise corrected p-values.

    score_orig_data : numpy.ndarray, shape=(n_regressors, n_descriptors)
      t-statistic associated with the significance test of the n_regressors
      explanatory variates against the n_descriptors target variates.
      The ranks of the scores into the h0 distribution correspond to the
      p-values.

    h0_fmax : array-like, shape=(n_perm, )
      Distribution of the (max) t-statistic under the null hypothesis
      (obtained from the permutations). Array is sorted.

    References
    ----------
    [1] Anderson, M. J. & Robinson, J. (2001).
        Permutation tests for linear models.
        Australian & New Zealand Journal of Statistics, 43(1), 75-88.
    [2] Winkler, A. M. et al. (2014).
        Permutation inference for the general linear model.
        Neuroimage.
    [3] Freedman, D. & Lane, D. (1983).
        A nonstochastic interpretation of reported significance levels.
        J. Bus. Econ. Stats., 1(4), 292-298

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # check n_jobs (number of CPUs)
    n_jobs = check_n_jobs(n_jobs)

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

    # orthogonalize design to speed up subsequent permutations
    orthogonalized_design = orthogonalize_design(tested_vars, target_vars,
                                                 confounding_vars)
    tested_vars_resid_covars = orthogonalized_design[0]
    target_vars_resid_covars = orthogonalized_design[1]
    covars_orthonormalized = orthogonalized_design[2]

    # OLS regression (t-scores) on original data
    scores_original_data = t_score_with_covars_and_normalized_design(
        tested_vars_resid_covars, target_vars_resid_covars,
        covars_orthonormalized)

    if two_sided_test:
        sign_scores_original_data = np.sign(scores_original_data)
        scores_original_data = np.fabs(scores_original_data)

    # Permutations
    # parallel computing units perform a reduced number of permutations each
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)
        n_perm_chunks[-1] += n_perm % n_jobs
    elif n_perm > 0:
        warnings.warn('The specified number of permutations is %d and '
                      'the number of jobs to be performed in parallel has '
                      'set to %s. This is incompatible so only %d jobs will '
                      'be running. You may want to perform more permutations '
                      'in order to take the most of the available computing '
                      'ressources.' % (n_perm, n_jobs, n_perm))
        n_perm_chunks = np.ones(n_perm, dtype=int)
    else:  # 0 or negative number of permutations => original data scores only
        if two_sided_test:
            scores_original_data = (scores_original_data
                                    * sign_scores_original_data)
        return np.asarray([]), scores_original_data,  np.asarray([])
    # actual permutations, seeded from a random integer between 0 and maximum
    # value represented by np.int32 (to have a large entropy).
    ret = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_permuted_ols_on_chunk)(
            scores_original_data, tested_vars_resid_covars,
            target_vars_resid_covars.T, covars_orthonormalized,
            n_perm_chunk=n_perm_chunk, intercept_test=intercept_test,
            two_sided_test=two_sided_test,
            random_state=rng.random_integers(np.iinfo(np.int32).max))
        for n_perm_chunk in n_perm_chunks)
    # reduce results
    scores_as_ranks_parts, h0_fmax_parts = zip(*ret)
    h0_fmax = np.hstack((h0_fmax_parts))
    scores_as_ranks = np.zeros((n_regressors, n_descriptors))
    for scores_as_ranks_part in scores_as_ranks_parts:
        scores_as_ranks += scores_as_ranks_part
    # convert ranks into p-values
    pvals = (n_perm + 1 - scores_as_ranks) / float(1 + n_perm)

    # put back sign on scores if it was removed in the case of a two-sided test
    # (useful to distinguish between positive and negative effects)
    if two_sided_test:
        scores_original_data = scores_original_data * sign_scores_original_data

    return - np.log10(pvals), scores_original_data.T, h0_fmax[0]
