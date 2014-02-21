"""
Massively Univariate Linear Model estimated with OLS and permutation test.

"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
# refactorized by Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import sys
import warnings
import numpy as np
from scipy import linalg, stats
from sklearn.utils import gen_even_slices, check_random_state
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
    The two above features (memory allocation and data thresholding) are not
    implemented in scipy.sparse objects or numpy array structures.

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
                 threshold=-np.inf):
        self.n_elts = n_elts
        self.n_iter = n_iter
        self.max_elts = max(max_elts, n_elts)
        self.data = np.empty(
            self.max_elts,
            dtype=[('iter_id', np.int32), ('x_id', np.int32),
                   ('y_id', np.int32), ('score', np.float32)])
        self.sizes = np.zeros(n_iter, dtype=int)
        self.threshold = threshold

    def get_data(self):
        return self.data[:self.n_elts]

    def merge(self, others):
        """Copy one or several GrowableSparseArray into the current structure.

        Parameters
        ----------
        others: list of GrowableSparseArray or GrowableSparseArray
          The structures to be merged into the current structure.

        """
        if isinstance(others, GrowableSparseArray):
            return self.merge([others])
        if not isinstance(others, list) and not isinstance(others, tuple):
            raise TypeError(
                '\'others\' is not a list/tuple of GrowableSparseArray '
                'or a GrowableSparseArray.')
        for gs_array in others:
            if not isinstance(gs_array, GrowableSparseArray):
                raise TypeError('List element is not a GrowableSparseArray.')
            if gs_array.n_iter != self.n_iter:
                raise ValueError('Cannot merge a structure with %d iterations '
                                'into a structure with %d iterations.'
                                % (gs_array.n_iter, self.n_iter))

        acc_sizes = [self.sizes]
        acc_data = [self.get_data()]
        for gs_array in others:
            # threshold the data to respect self.threshold
            if gs_array.threshold < self.threshold:
                gs_array_data_thresholded = (
                    gs_array.get_data()[gs_array.get_data()['score']
                                       >= self.threshold])
                acc_sizes.append([gs_array_data_thresholded.size])
                acc_data.append(gs_array_data_thresholded)
            elif gs_array.threshold > self.threshold:
                warnings.warn('Merging a GrowableSparseArray into another '
                              'with a lower threshold: parent array may '
                              'contain less scores than its threshold '
                              'suggests.')
                acc_sizes.append(gs_array.sizes)
                acc_data.append(gs_array.get_data())
            else:
                acc_sizes.append(gs_array.sizes)
                acc_data.append(gs_array.get_data())

        self.sizes = np.array(acc_sizes).sum(axis=0)
        self.data = np.concatenate(acc_data)
        self.n_elts = self.sizes.sum()
        self.max_elts = self.n_elts
        self.data = np.sort(self.data, order=['iter_id', 'x_id', 'y_id'])

    def append(self, iter_id, iter_data, y_offset=0):
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
            gs_array = GrowableSparseArray(
                self.n_iter, threshold=self.threshold)
            gs_array.data = new_data
            gs_array.sizes = np.zeros((gs_array.n_iter), dtype=int)
            gs_array.sizes[iter_id] = score_size
            gs_array.n_elts = score_size
            gs_array.max_elts = score_size
            self.merge(gs_array)
        else:  # it fits --> updates (efficient)
            self.data['x_id'][self.n_elts:new_n_elts] = x_idx
            self.data['y_id'][self.n_elts:new_n_elts] = y_idx + y_offset
            self.data['score'][self.n_elts:new_n_elts] = (
                iter_data[y_idx, x_idx])
            self.data['iter_id'][self.n_elts:new_n_elts] = iter_id
            self.sizes[iter_id] += score_size
            self.n_elts = new_n_elts


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


def _permuted_ols_on_chunk(seed, tested_vars, target_vars_chunk,
                           confounding_vars=None, n_perm=10000,
                           sparsity_threshold=1e-04,
                           target_vars_chunk_position=0,
                           intercept_test=True):
    """Massively univariate group analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    seed: int,
      Seed for random number generator, to have the same permutations
      in each computing units. It must be the same in all parallel calls
      to _permuted_ols_on_chunk.
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variates.
    target_vars_chunk: array-like, shape=(n_samples, n_targets)
      fMRI data. F-ordered for efficient computations.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Clinical data (covariates).
    n_perm: int,
      Number of permutations.
    sparsity_threshold: float,
      Threshold under which the permutation scores are not stored
      (because they have no chance to correspond to the max)
    target_vars_chunk_position: int,
      offset corresponding to the target variates chunk position
    intercept_test: boolean,
      Change the permutation scheme (swap signs for intercept,
      switch labels otherwise). See [1]

    Returns
    -------
    gs_array: GrowableSparseArray,
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
    ...     _permuted_ols_on_chunk)
    >>> X = np.ones((4, 1))
    >>> Y = np.array([[1, 2, 2, 1]]).T
    >>> res, params = _permuted_ols_on_chunk(
    ...     0, X, Y, n_perm=1, sparsity_threshold=1.)
    >>> res.get_data()[0]
    (0, 0, 0, 27.0)
    >>> res.get_data()[1]
    (1, 0, 0, 0.3333333432674408)
    >>> params
    {'lost_dof': 0, 'threshold': 0.0, 'n_perm': 1, 'n_subj': 4}

    """
    # initialize the seed of the random generator
    rng = check_random_state(seed)

    n_samples, n_regressors = tested_vars.shape
    n_descriptors_chunk = target_vars_chunk.shape[1]

    # OLS regression on original data
    if confounding_vars is not None:
        # step 1: extract effect of covars from target vars
        covars_orthonormed = orthonormalize_matrix(confounding_vars)
        if not covars_orthonormed.flags['C_CONTIGUOUS']:
            warnings.warn('Confounding variates not C_CONTIGUOUS.')
            covars_orthonormed = np.ascontiguousarray(covars_orthonormed)
        targetvars_chunk_normalized = normalize_matrix_on_axis(
            target_vars_chunk).T  # faster with F-ordered target_vars_chunk
        if not targetvars_chunk_normalized.flags['C_CONTIGUOUS']:
            warnings.warn('Target variates not C_CONTIGUOUS.')
            targetvars_chunk_normalized = np.ascontiguousarray(
                targetvars_chunk_normalized)
        beta_targetvars_covars = np.dot(
            targetvars_chunk_normalized, covars_orthonormed)
        targetvars_resid_covars = targetvars_chunk_normalized - np.dot(
            beta_targetvars_covars, covars_orthonormed.T)
        targetvars_resid_covars = normalize_matrix_on_axis(
            targetvars_resid_covars, axis=1)
        lost_dof = covars_orthonormed.shape[1]
        # step 2: extract effect of covars from tested vars
        testedvars_normalized = normalize_matrix_on_axis(tested_vars.T, axis=1)
        beta_testedvars_covars = np.dot(
            testedvars_normalized, covars_orthonormed)
        testedvars_resid_covars = testedvars_normalized - np.dot(
            beta_testedvars_covars, covars_orthonormed.T)
        testedvars_resid_covars = normalize_matrix_on_axis(
            testedvars_resid_covars, axis=1).T.copy()
        if not testedvars_resid_covars.flags['C_CONTIGUOUS']:
            warnings.warn('Tested variates not C_CONTIGUOUS.')
            testedvars_resid_covars = np.ascontiguousarray(
                testedvars_resid_covars)
    else:
        targetvars_resid_covars = normalize_matrix_on_axis(
            target_vars_chunk).T
        testedvars_resid_covars = normalize_matrix_on_axis(
            tested_vars).copy()
        if not targetvars_resid_covars.flags['C_CONTIGUOUS']:
            warnings.warn('Target variates not C_CONTIGUOUS.')
            targetvars_resid_covars = np.ascontiguousarray(
                targetvars_resid_covars)
        if not testedvars_resid_covars.flags['C_CONTIGUOUS']:
            warnings.warn('Tested variates not C_CONTIGUOUS.')
            testedvars_resid_covars = np.ascontiguousarray(
                testedvars_resid_covars)
        covars_orthonormed = None
        lost_dof = 0
    # step 3: original regression (= regression on residuals + adjust F score)
    # compute F score for original data
    score_original_data = _f_score(
        testedvars_resid_covars, targetvars_resid_covars.T, covars_orthonormed,
        lost_dof, normalized_design=True)

    # We use a threshold to sparsify the permutations results since not all
    # the scores have the chance to be retained as the max value at the end
    # we keep scores < threshold
    threshold = stats.f.isf(sparsity_threshold, 1, n_samples - lost_dof - 1)
    # We use a special data structure to store the results of the permutations
    # max_elts is used to preallocate memory
    max_elts = int(n_regressors * n_descriptors_chunk
                   * np.sqrt(sparsity_threshold) * n_perm)
    gs_array = GrowableSparseArray(
        n_perm + 1, max_elts=max_elts, threshold=threshold)
    # add original data results as permutation 0
    gs_array.append(
        0, score_original_data, y_offset=target_vars_chunk_position)

    # do the permutations
    for i in xrange(1, n_perm + 1):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            targetvars_resid_covars = (
                targetvars_resid_covars
                * (rng.randint(2, size=(1, n_samples)) * 2 - 1))
        else:
            # shuffle data
            # Regarding computation costs, we choose to shuffle testvars
            # and covars rather than fmri_signal)
            # Also, it is important to keep testedvars and covars in the
            # same order to simplify f_score computation (null dot product)
            shuffle_idx = rng.permutation(n_samples)
            #rng.shuffle(shuffle_idx)
            testedvars_resid_covars = testedvars_resid_covars[shuffle_idx]
            if covars_orthonormed is not None:
                covars_orthonormed = covars_orthonormed[shuffle_idx]

        # OLS regression on randomized data
        if confounding_vars is not None:
            assert(covars_orthonormed.flags['C_CONTIGUOUS'])
        cur_res = _f_score(
            testedvars_resid_covars, targetvars_resid_covars.T,
            covars_orthonormed, lost_dof, normalized_design=True)
        gs_array.append(
            i, cur_res, y_offset=target_vars_chunk_position)

    params = {'lost_dof': lost_dof, 'threshold': threshold,
              'n_perm': n_perm, 'n_subj': n_samples}
    return gs_array, params


def permuted_ols(tested_vars, target_vars, confounding_vars=None,
                 model_intercept=True, n_perm=10000, sparsity_threshold=None,
                 random_state=None, n_jobs=1):
    """Massively univariate group analysis with permuted OLS.

    Tested variates are independently fitted to target variates descriptors
    (e.g. brain imaging signal) according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variates may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variates and the target variates. A max-type
    procedure is used to obtain family-wise corrected p-values.

    The variates should be given C-contiguous.

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
      we get in the pvalues estimation.
    sparsity_threshold: float,
      Threshold under which the permutation scores are not stored
      (because they have no chance to correspond to the max).
      If None is provided, it is automatically set at best from the problem
      dimensions. However, it may be useful to manually set it for specific
      needs.
    random_state: int,
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
    # initialize the seed of the random generator
    rng = check_random_state(random_state)
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
    # check explanatory variates dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T

    n_descriptors = target_vars.shape[1]
    n_samples, n_regressors = tested_vars.shape

    # automatically set sparsity_threshold if not provided
    if sparsity_threshold is None:
        sparsity_threshold = 1. / np.sqrt(n_perm * n_descriptors)

    # check if explanatory variates is intercept (constant) or not
    if (tested_vars.shape[1] == 1 and np.unique(tested_vars).size == 1):
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

    # split target variates into chunks for parallel processing
    seed = rng.randint(sys.maxint)  # seed must be the same in every job
    # caveat: it is dangerous to rely on the seed as a guaranty the random
    # numbers will be the same. On a cluster, two computation units may run
    # the code under different environments and possibly have different
    # random sequences.
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_permuted_ols_on_chunk)
          (seed, tested_vars, target_vars[:, chunk], confounding_vars, n_perm,
           sparsity_threshold=sparsity_threshold,
           target_vars_chunk_position=chunk.start,
           intercept_test=intercept_test)
          for chunk in gen_even_slices(
            n_descriptors + 1, min(n_descriptors, n_jobs)))
    # reduce results
    all_chunks_results, params = zip(*ret)
    final_results = GrowableSparseArray(
        n_perm + 1,
        threshold=all_chunks_results[0].threshold)  # same threshold everywhere
    final_results.merge(all_chunks_results)
    # get h0
    h0 = np.zeros(n_perm)
    cum_sizes = final_results.sizes.cumsum()
    for i in range(n_perm):
        tmp = final_results.get_data()[cum_sizes[i]:cum_sizes[i + 1]]['score']
        if tmp.size > 0:
            h0[i] = tmp.max()
        else:
            h0[i] = - np.inf
    if np.isinf(h0).sum() > 0.8 * n_perm:
        warnings.warn(
            "Sparsity threshold may be too low, yielding false negative.")
    # convert scores into p-values
    score_orig_data = final_results.get_data()[:final_results.sizes[0]]
    pvals = ((n_perm + 1
              - np.searchsorted(np.sort(h0), score_orig_data['score']))
             / float(n_perm + 1))
    pvals_mat = np.zeros(shape=(n_regressors, n_descriptors), dtype=np.float)
    pvals_mat[score_orig_data['x_id'], score_orig_data['y_id']] = - np.log10(
        pvals)
    return pvals_mat, score_orig_data, h0, params[0]
