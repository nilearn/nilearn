"""
Massively Univariate Linear Model estimated with OLS and permutation test.

"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
# refactorized by Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import numpy as np
from scipy import sparse
import sklearn.externals.joblib as joblib
from permuted_least_squares_aux import MULMSparseArray, _permuted_OLS_on_chunk


def permuted_OLS(tested_vars, imaging_vars, confounding_vars,
                 n_perm=10000, seed=0, n_jobs=0):
    """Massively univariate group analysis with permuted OLS.

    Tested variables are independently fitted to brain imaging signal
    descriptors according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variables may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variables and the imaging variables. A max-type
    procedure is used to obtain family-wise corrected p-values.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variables, fitted and tested independently from each others.
    imaging_vars: array-like, shape=(n_descriptors, n_samples)
      fMRI data, trying to be explained by explanatory and confounding
      variables.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Confounding variables (covariables), fitted but not tested.
    n_perm: int,
      Number of permutations to perform. Default is 10000.
      Permutations are costly but the more are performed, the more precision
      we get in the pvalues estimation.
    seed: int,
      Seed for random number generator, to have the same permutations
      in each computing units.
    n_jobs: int,
      Number of parallel workers.
      if 0 or negative numbers provided, all CPUs are used.

    Returns
    -------
    pvals: array-like, shape=(n_regressors, n_descriptors)
      Negative log10 p-values associated to the significance test of the
      n_regressors explanatory variables against the n_descriptors target
      variables. Family-wise corrected p-values.
    score_orig_data: MULMSparseArray object,
      Statistic associated to the significance test of the n_regressors
      explanatory variables against the n_descriptors target variables.
      The ranks of the scores into the h0 distribution correspond to the
      p-values.
    h0: array-like, shape=(n_perm, )
      Distribution of the test statistic under the null hypothesis
      (obtained from the permutations).
    params: dict,
      Parameters of the permuted model:
      - lost_dof: lost degrees of freedom
      - n_perm: number of permutations
      - n_subj: number of observations
      - threshold: threshold used to sparsify the results and reduce the
                   size of the permutation scores in memory.

    """
    if n_jobs < 1:
        n_jobs = joblib.cpu_count()
    # TODO: add various checks
    # check explanatory variables dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T

    # check if explanatory variables is intercept (constant) or not
    if (tested_vars.shape[1] == 1 and np.unique(tested_vars).size == 1):
        intercept_test = True
    else:
        intercept_test = False

    # split target variables into chunks for parallel processing
    n_descriptors = imaging_vars.shape[0]
    n_regressors = tested_vars.shape[1]
    sizes = np.linspace(
        0, n_descriptors, max(2, min(n_descriptors, n_jobs + 1))).astype(int)
    # run computation on chunks
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_permuted_OLS_on_chunk)
        (tested_vars, imaging_vars[start:sizes[i + 1]], confounding_vars,
         n_perm, seed=seed, target_vars_chunk_position=start,
         intercept_test=intercept_test)
        for i, start in enumerate(sizes[:-1]))
    # reduce results
    all_chunks_results, params = zip(*ret)
    final_results = MULMSparseArray(n_perm + 1)
    final_results.merge(all_chunks_results)
    # get h0
    h0 = np.zeros(n_perm)
    cum_sizes = final_results._sizes.cumsum().astype(int)
    for i in range(n_perm):
        h0[i] = final_results.get_data()[
            cum_sizes[i]:cum_sizes[i + 1]]['score'].max()
    # convert scores into p-values
    score_orig_data = final_results.get_data()[:final_results._sizes[0]]
    pvals = ((n_perm - np.searchsorted(h0, score_orig_data['score']))
             / float(n_perm))
    np.seterr(divide='ignore')  # ignore division-by-zero warning in log10
    pvals_mat = sparse.coo_matrix(
        (- np.log10(pvals),
         (score_orig_data['testvar_id'],
          score_orig_data['imgvar_id'])),
        shape=(n_regressors, n_descriptors), dtype=np.float).todense()
    return pvals_mat, score_orig_data, h0, params[0]
