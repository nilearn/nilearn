"""
Massively Univariate Linear Model estimated with OLS and permutation test.

"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
# refactorized by Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import warnings
import numpy as np
from scipy import sparse, linalg, stats
from sklearn.utils import gen_even_slices, check_random_state
import sklearn.externals.joblib as joblib


def normalize_matrix_on_axis(m, axis=0):
    """ Normalize a 2D matrix on an axis.

    Parameters
    ----------
    m : numpy 2D array,
        The matrix to normalize
    axis : integer in {0, 1}, optional
        A valid axis to normalize accross (0 by default)

    Returns
    -------
    ret : numpy array, shape = m.shape
        The normalize matrix

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
        raise Exception('Only for 2D array.')

    if axis == 0:
        ret = m / np.sqrt(np.sum(m ** 2, axis=0))
    elif axis == 1:
        ret = normalize_matrix_on_axis(m.T).T
    else:
        raise Exception('Invalid axis in normalization.')
    return np.ascontiguousarray(ret)


def orthonormalize_matrix(m, tol=1.e-12):
    """ Orthonormalize a matrix.

    Parameters
    ----------
    m : numpy array,
        The matrix to orthonormalize

    Returns
    -------
    ret : numpy array, shape = m.shape
        The orthonormalize matrix

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     orthonormalize_matrix)
    >>> X = np.array([[0, 1], [0, 1]])
    >>> orthonormalize_matrix(X)
    array([[ 0.70710678,  0.        ],
           [ 0.70710678,  0.        ]])
    >>> X = np.array([[0, 1], [4, 0]])
    >>> orthonormalize_matrix(X)
    array([[ 0., -1.],
           [-1.,  0.]])

    """
    U, s, _ = linalg.svd(m, 0)
    n_eig = s[abs(s) > tol].size
    tmp = np.dot(U, np.diag(s))[:, :n_eig]
    n_null_eig = s.size - n_eig
    tmp = normalize_matrix_on_axis(tmp)
    if n_null_eig > 0:
        tmp = np.ascontiguousarray(
            np.hstack((tmp, np.zeros((tmp.shape[0], n_null_eig)))))
    return tmp


class GrowableSparseArray(object):
    """Data structure to contain data from numerous estimations.

    Examples of application are all resampling schemes
    (bootstrap, permutations, ...)

    GrowableSparseArray can be seen as a three-dimensional array that contains
    scores associated with three position indices corresponding to
    (i) an iteration (or estimation), (ii) a test variate and
    (iii) a target variate.
    Memory is pre-allocated to store a large number of scores. The structure
    can be indexed efficiently according to three dimensions to add new
    scores at the right position fast.
    The allocated space can be extended if needed, but we want to avoid this
    because it is costly. User should carefully initialize the structure.
    Only scores above a predetermined threshold are actually stored, others
    are ignored.

    Attributes
    ----------
    n_elts: int
      The total number of scores actually stored into the data structure
    n_iter: int
      Number of trials (using as many iterators)
    max_elts: int
      Maximum number of scores that can be stored into the structure
    data: array-like, own-designed dtype
      The actual scores corresponding to all the estimations.
      dtype is built so that every score is associated with three position
      GrowableSparseArray can be seen as a three-dimensional array so every
      score is associated with three position indices corresponding to
      (i) an iteration (or an estimator) ('iter_id'),
      (ii) a test variate ('x_id') and
      (iii) a target variate ('y_id').
    sizes: array-like, shape=(n_iter, )
      The number of scores stored for each estimation.
      Useful to select a range of values from iteration ids.
    threshold: float,
      Sparsity threshold used to discard scores that are to low to have a
      chance to correspond to a maximum value amongst all the scores of
      a given iteration.

    """
    def __init__(self, n_iter=10000, n_elts=0, max_elts=None,
                 threshold=np.inf):
        self.n_elts = n_elts
        self.n_iter = n_iter
        self.max_elts = max(max_elts, n_elts)
        self.data = np.empty(
            self.max_elts,
            dtype=[('iter_id', np.int32), ('x_id', np.int32),
                   ('y_id', np.int32), ('score', np.float32)])
        self.sizes = np.zeros((n_iter))
        self.threshold = threshold

    def get_data(self):
        return self.data[:self.n_elts]

    def merge(self, others):
        """Copy one or several GrowableSparseArray into the current structure.

        Parameters
        ----------
        l: list of GrowableSparseArray or GrowableSparseArray
          The structures to be merged into the current structure.

        """
        if isinstance(others, GrowableSparseArray):
            return self.merge([others])
        if not isinstance(others, list) and not isinstance(others, tuple):
            raise Exception(
                '\'others\' is not a list/tuple of GrowableSparseArray '
                'or a GrowableSparseArray.')
        for msarray in others:
            if not isinstance(msarray, GrowableSparseArray):
                raise Exception('msarray is not a GrowableSparseArray.')

        self.sizes = np.array([self.sizes] +
                               [msa.sizes for msa in others]).sum(axis=0)
        self.data = np.concatenate([self.get_data()] +
                                    [msa.get_data() for msa in others])
        self.n_elts = self.sizes.sum()
        self.max_elts = self.n_elts
        self.data = np.sort(self.data, order=['iter_id', 'x_id', 'y_id'])

        return

    def append_iter_data(self, iter_id, iter_data, y_offset=0):
        """Add the data of one estimation (iteration) into the structure.

        This is done in a memory-efficient way, by taking into account
        pre-allocated space.

        Parameters
        ----------
        iter_id: int,
          ID of the estimation we are inserting into the structure
        iter_data: array-like, shape=(n_targets_chunk, n_regressors)
          Scores corresponding to the iteration chunk to be inserted into
          the data structure.
        y_offset: int,
          Position of the target variates chunk relative to the original
          dataset.

        """
        # we only store float32 to save space
        iter_data = iter_data.astype('float32')
        # we sparsify the matrix wrt. threshold using coordinates list
        y_idx, x_idx = (iter_data >= self.threshold).nonzero()
        score_size = len(x_idx)
        new_n_elts = score_size + self.n_elts
        if (new_n_elts > self.max_elts or
            self.sizes[iter_id + 1:].sum() > 0):  # insertion (costly)
            new_data = np.empty(score_size,
                        dtype=[('iter_id', np.int32), ('x_id', np.int32),
                               ('y_id', np.int32), ('score', np.float32)])
            new_data['x_id'][:] = x_idx
            new_data['y_id'][:] = y_idx + y_offset
            new_data['score'][:] = iter_data[y_idx, x_idx]
            new_data['iter_id'][:] = iter_id
            msarray = GrowableSparseArray(self.n_iter)
            msarray.data = new_data
            msarray.sizes = np.zeros((msarray.n_iter))
            msarray.sizes[iter_id] = score_size
            msarray.n_elts = score_size
            msarray.max_elts = score_size
            self.merge(msarray)
        else:  # it fits --> updates (efficient)
            self.data['x_id'][self.n_elts:new_n_elts] = x_idx
            self.data['y_id'][self.n_elts:new_n_elts] = y_idx + y_offset
            self.data['score'][self.n_elts:new_n_elts] = (
                iter_data[y_idx, x_idx])
            self.data['iter_id'][self.n_elts:new_n_elts] = iter_id
            self.sizes[iter_id] += score_size
            self.n_elts = new_n_elts
        return


def f_score(vars1, vars2, covars, lost_dof):
    """Compute F-score associated with the regression of vars2 against vars1

    Covariates are taken into account

    Parameters
    ----------
    vars1: array-like, shape=(n_samples, n_var1)
      Explanatory variates
    vars2: array-like, shape=(n_var2, n_samples)
      Targets variates
    covars, array-like, shape=(n_samples, n_covars)
      Confounding variates
    lost_dof: int,
      Lost degrees of freedom

    Returns
    -------
    score: array-like, shape=(n_var2, n_var1)
      F-scores associated with the tests of each explanatory variate against
      each target variate (in the presence of covars).

    """
    if not vars1.flags['C_CONTIGUOUS']:
        warnings.warn('explanatory variates not C_CONTIGUOUS.')
        vars1 = np.ascontiguousarray(vars1)
    if not vars2.flags['C_CONTIGUOUS']:
        warnings.warn('target variates not C_CONTIGUOUS.')
        vars2 = np.ascontiguousarray(vars2)
    if not covars.flags['C_CONTIGUOUS']:
        warnings.warn('confounding variates not C_CONTIGUOUS.')
        covars = np.ascontiguousarray(covars)
    beta_vars2_vars1 = np.dot(vars2, vars1)
    beta_vars2_covars = np.dot(vars2, covars)
    dof = vars2.shape[1] - 1 - lost_dof
    b2 = beta_vars2_vars1 ** 2
    a2 = np.sum(beta_vars2_covars ** 2, 1)
    rss = (1 - a2[:, np.newaxis] - b2)
    score = b2 / rss
    score *= dof
    return score


def permuted_ols_on_chunk(tested_vars, target_vars_chunk,
                           confounding_vars, n_perm, sparsity_threshold=1e-04,
                           target_vars_chunk_position=0,
                           intercept_test=True, random_state=0):
    """Massively univariate group analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variates.
    target_vars_chunk: array-like, shape=(n_targets, n_samples)
      fMRI data.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Clinical data (covariates).
    n_perm: int,
      Number of permutations
    sparsity_threshold: float,
      Threshold under which the permutation scores are not stored
      (because they have no chance to correspond to the max)
    target_vars_offset:
      offset corresponding to the target variates chunk position
    intercept_test: boolean,
      Change the permutation scheme (swap signs for intercept,
      switch labels otherwise). See [1]
    random_state: int,
      Seed for random number generator, to have the same permutations
      in each computing units.

    Returns
    -------
    msarray: GrowableSparseArray,
      Permutation scores corresponding to the current target variates chunk
      (passed as an argument of the function call).
    params: dict,
      Parameters of the permuted model:
      - lost_dof: lost degrees of freedom
      - n_perm: number of permutations
      - n_subj: number of observations
      - threshold: threshold used to sparsify the results and reduce the
                   size of the permutation scores in memory.

    References
    ----------
    [1] Fisher, R. A. (1935). The design of experiments.

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     permuted_ols_on_chunk)
    >>> X = np.ones((4, 1))
    >>> Y = np.array([[1, 2, 2, 1]])
    >>> Z = np.zeros((4, 1))
    >>> res, params = permuted_ols_on_chunk(
    ...     X, Y, Z, n_perm=1, sparsity_threshold=1.)
    >>> res.get_data()[0]
    (0, 0, 0, 18.0)
    >>> res.get_data()[1]
    (1, 0, 0, 0.2222222238779068)
    >>> params
    {'lost_dof': 1, 'threshold': 0.0, 'n_perm': 1, 'n_subj': 4}

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    n_samples, n_regressors = tested_vars.shape
    n_descriptors_chunk = target_vars_chunk.shape[0]

    # OLS regression on original data
    # step 1: extract effect of covars from target vars
    covars_orthonormed = orthonormalize_matrix(confounding_vars)
    targetvars_chunk_normalized = normalize_matrix_on_axis(
        target_vars_chunk, axis=1)
    beta_targetvars_covars = np.dot(
        targetvars_chunk_normalized, covars_orthonormed)
    targetvars_resid_covars = targetvars_chunk_normalized - np.dot(
        beta_targetvars_covars, covars_orthonormed.T)
    targetvars_resid_covars = normalize_matrix_on_axis(
        targetvars_resid_covars, axis=1)
    lost_dof = covars_orthonormed.shape[1]
    # step 2: extract effect of covars from tested vars
    testedvars_normalized = normalize_matrix_on_axis(tested_vars.T, axis=1)
    beta_testedvars_covars = np.dot(testedvars_normalized, covars_orthonormed)
    testedvars_resid_covars = testedvars_normalized - np.dot(
        beta_testedvars_covars, covars_orthonormed.T)
    testedvars_resid_covars = normalize_matrix_on_axis(
        testedvars_resid_covars, axis=1).T.copy()
    # step 3: original regression (= regression on residuals + adjust F score)
    # compute F score for original data
    score_original_data = f_score(
        testedvars_resid_covars, targetvars_resid_covars, covars_orthonormed,
        lost_dof)

    # We use a threshold to sparsify the permutations results since not all
    # the scores have the chance to be retained as the max value at the end
    # we keep scores < threshold
    threshold = stats.f.isf(sparsity_threshold, 1, n_samples - lost_dof - 1)
    # We use a special data structure to store the results of the permutations
    # max_elts is used to preallocate memory
    max_elts = int(n_regressors * n_descriptors_chunk
                   * np.sqrt(sparsity_threshold) * n_perm)
    msarray = GrowableSparseArray(
        n_perm + 1, max_elts=max_elts, threshold=threshold)
    # add original data results as permutation 0
    msarray.append_iter_data(
        0, score_original_data, y_offset=target_vars_chunk_position)

    # do the permutations
    for i in xrange(1, n_perm + 1):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            targetvars_resid_covars = (
                targetvars_resid_covars
                * (rng.randint(2, size=(1, n_samples)) * 2 - 1))
        else:
            # shuffle data (regarding computation costs, we choose to shuffle
            # testvars and covars rather than fmri_signal)
            shuffle_idx = rng.permutation(n_samples)
            #rng.shuffle(shuffle_idx)
            testedvars_resid_covars = testedvars_resid_covars[shuffle_idx]
            covars_orthonormed = covars_orthonormed[shuffle_idx]

        # OLS regression on randomized data
        cur_res = f_score(
            testedvars_resid_covars, targetvars_resid_covars,
            covars_orthonormed, lost_dof)
        msarray.append_iter_data(
            i, cur_res, y_offset=target_vars_chunk_position)

    params = {'lost_dof': lost_dof, 'threshold': threshold,
              'n_perm': n_perm, 'n_subj': n_samples}
    return msarray, params


def permuted_ols(tested_vars, imaging_vars, confounding_vars, n_perm=10000,
                 sparsity_threshold=1e-04, random_state=0, n_jobs=0):
    """Massively univariate group analysis with permuted OLS.

    Tested variates are independently fitted to brain imaging signal
    descriptors according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variates may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variates and the imaging variates. A max-type
    procedure is used to obtain family-wise corrected p-values.

    The variates should be given C-contiguous.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variates, fitted and tested independently from each others.
    imaging_vars: array-like, shape=(n_descriptors, n_samples)
      fMRI data, trying to be explained by explanatory and confounding
      variates.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
    n_perm: int,
      Number of permutations to perform. Default is 10000.
      Permutations are costly but the more are performed, the more precision
      we get in the pvalues estimation.
    sparsity_threshold: float,
      Threshold under which the permutation scores are not stored
      (because they have no chance to correspond to the max)
    random_state: int,
      Seed for random number generator, to have the same permutations
      in each computing units.
    n_jobs: int,
      Number of parallel workers.
      if 0 or negative numbers provided, all CPUs are used.

    Returns
    -------
    pvals: array-like, shape=(n_regressors, n_descriptors)
      Negative log10 p-values associated with the significance test of the
      n_regressors explanatory variates against the n_descriptors target
      variates. Family-wise corrected p-values.
    score_orig_data: GrowableSparseArray object,
      Statistic associated with the significance test of the n_regressors
      explanatory variates against the n_descriptors target variates.
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

    References
    ----------
    [1] Anderson, M. J., & Robinson, J. (2001).
        Permutation tests for linear models.
        Australian & New Zealand Journal of Statistics, 43(1), 75-88.

    """
    if n_jobs < 1:
        n_jobs = joblib.cpu_count()
    # TODO: add various checks
    # check explanatory variates dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T

    # check if explanatory variates is intercept (constant) or not
    if (tested_vars.shape[1] == 1 and np.unique(tested_vars).size == 1):
        intercept_test = True
    else:
        intercept_test = False

    # split target variates into chunks for parallel processing
    n_descriptors = imaging_vars.shape[0]
    n_regressors = tested_vars.shape[1]
    # run computation on chunks
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(permuted_ols_on_chunk)
        (tested_vars, imaging_vars[chunk], confounding_vars, n_perm,
         sparsity_threshold=sparsity_threshold, random_state=random_state,
         target_vars_chunk_position=chunk.start, intercept_test=intercept_test)
        for chunk in gen_even_slices(
            n_descriptors, max(2, min(n_descriptors, n_jobs))))
    # reduce results
    all_chunks_results, params = zip(*ret)
    final_results = GrowableSparseArray(n_perm + 1)
    final_results.merge(all_chunks_results)
    # get h0
    h0 = np.zeros(n_perm)
    cum_sizes = final_results.sizes.cumsum().astype(int)
    for i in range(n_perm):
        tmp = final_results.get_data()[cum_sizes[i]:cum_sizes[i + 1]]['score']
        if tmp.size > 0:
            h0[i] = tmp.max()
        else:
            h0[i] = - np.inf
    if np.isinf(h0).sum() > 0.75 * n_perm:
        warnings.warn(
            "Sparsity threshold may be too low, yielding false negative.")
    # convert scores into p-values
    score_orig_data = final_results.get_data()[:final_results.sizes[0]]
    pvals = (n_perm + 1 - np.searchsorted(
                np.sort(h0), score_orig_data['score'])) / float(n_perm + 1)
    np.seterr(divide='ignore')  # ignore division-by-zero warning in log10
    pvals_mat = sparse.coo_matrix(
        (- np.log10(pvals),
         (score_orig_data['x_id'],
          score_orig_data['y_id'])),
        shape=(n_regressors, n_descriptors), dtype=np.float).todense()
    return pvals_mat, score_orig_data, h0, params[0]
