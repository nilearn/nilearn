"""
Massively Univariate Linear Model estimated with OLS and permutation test.
Separate module from the main routine since joblib requires parallelized
functions to be in a different file.

"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
# refactorized by Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import numpy as np
from scipy import linalg, stats


def normalize_matrix_on_axis(m, axis=0, copy=True):
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
    """
    if axis == 0:
        ret = m / np.sqrt(np.sum(m ** 2, axis=axis))
    elif axis == 1:
        ret = normalize_matrix_on_axis(m.T).T
    else:
        raise Exception('Only for 2D array.')
    if copy:
        ret = ret.copy()
    return ret


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
    """
    U, s, _ = linalg.svd(m, 0)
    n_eig = s[abs(s) > tol].size
    tmp = np.dot(U, np.diag(s))[:, :n_eig]
    n_null_eig = tmp.shape[1] - n_eig
    if n_null_eig > 0:
        ret = np.hstack((
            normalize_matrix_on_axis(tmp),
            np.zeros((tmp.shape[0], n_null_eig))))
    else:
        ret = normalize_matrix_on_axis(tmp)
    return ret


class MULMSparseArray(object):
    """Data structure to contain permutations data.

    Memory is pre-allocated to store the large number of data produced by
    the permutation scheme. The structure is indexed efficiently to add
    new scores fast. Would more space be needed, the allocated space would
    need to be extended, which is costly.
    Only sparse results are stored because we look forward using a max-type
    correction on the scores to ensure family-wise error control on false
    detections. Therefore, scores are relevant only if they have a chance to
    correspond to the maximum value accross all tests on all targets for the
    same permutation (the standard assumption in neuroimagin is to consider
    that all the image descriptors --i.e. the targets variables-- have the
    same distribution under the null hypothesis).

    Attributes
    ----------
    _n_elts: int
      The total number of scores actually stored into the data structure
    _n_perm: int
      Number of permutations performed in the permutation scheme
    _max_elts: int
      Maximum number of scores that can be contained into the structure
    _data: array-like, own-designed dtype
      The actual scores corresponding to all the tests performed under
      permutation.
    _sizes: array-like, shape=(n_perm, )
      The number of scores stored for each permutation.
      Useful to select a range of values from permutation ids.
    _threshold: float,
      Sparsity threshold used to discard scores that are to low to have a
      chance to correspond to a maximum value amongst all the scores of
      a given permutation.

    """
    def __init__(self, n_perm=10000, n_elts=0, max_elts=None,
                 threshold=np.inf):
        self._n_elts = n_elts
        self._n_perm = n_perm
        self._max_elts = max(max_elts, n_elts)
        self._data = np.empty(
            self._max_elts,
            dtype=[('perm_id', np.int32), ('testvar_id', np.int32),
                   ('imgvar_id', np.int32), ('score', np.float32)])
        self._sizes = np.zeros((n_perm))
        self._threshold = threshold

    def get_data(self):
        return self._data[:self._n_elts]

    def merge(self, l):
        """Copy one or several MULMSparseArray into the current structure.

        Parameters
        ----------
        l: list of MULMSparseArray or MULMSparseArray
          The structures to be merged into the current structure.

        """
        if isinstance(l, MULMSparseArray):
            return self.merge([l])
        if not isinstance(l, list) and not isinstance(l, tuple):
            raise Exception('l is not a list/tuple of MULMSparseArray '
                            'or a MULMSparseArray.')
        for msarray in l:
            if not isinstance(msarray, MULMSparseArray):
                raise Exception('msarray is not a MULMSparseArray.')

        self._sizes = np.array([self._sizes] +
                               [msa._sizes for msa in l]).sum(axis=0)
        self._data = np.concatenate([self.get_data()] +
                                    [msa.get_data() for msa in l])
        self._n_elts = self._sizes.sum()
        self._max_elts = self._n_elts
        self._data = np.sort(self._data,
                             order=['perm_id', 'testvar_id', 'imgvar_id'])

        return

    def append_perm_data(self, perm_id=None, perm_data=None,
                         target_vars_offset=0):
        """Add the data of one permutation into the structure.

        This is done in a memory-efficient way, by taking into account
        pre-allocated space.

        Parameters
        ----------
        perm_id: int,
          ID of the permutation we are inserting into the structure
        perm_data: array-like, shape=(n_targets_chunk, n_regressors)
          Scores corresponding to the permutation chunk to be inserted into
          the data structure.
        target_vars_offset: int,
          Position of the target variables chunk relative to the original
          dataset.

        """
        if perm_id is None:
            raise Exception('Missing permutation id')
        if perm_data is None:
            raise Exception('No permutation data provided')
        # we only store float32 to save space
        perm_data = perm_data.astype('float32')
        # we sparsify the matrix wrt. threshold using coordinates list
        imgvar_idx, testvar_idx = (perm_data >= self._threshold).nonzero()
        score_size = len(testvar_idx)
        new_n_elts = score_size + self._n_elts
        if (new_n_elts > self._max_elts or
            self._sizes[perm_id + 1:].sum() > 0):  # insertion (costly)
            new_data = np.empty(score_size,
                        dtype=[('perm_id', np.int32), ('testvar_id', np.int32),
                               ('imgvar_id', np.int32), ('score', np.float32)])
            new_data['testvar_id'][:] = testvar_idx
            new_data['imgvar_id'][:] = imgvar_idx + target_vars_offset
            new_data['score'][:] = perm_data[imgvar_idx, testvar_idx]
            new_data['perm_id'][:] = perm_id
            msarray = MULMSparseArray(self._n_perm)
            msarray._data = new_data
            msarray._sizes = np.zeros((msarray._n_perm))
            msarray._sizes[perm_id] = score_size
            msarray._n_elts = score_size
            msarray._max_elts = score_size
            self.merge(msarray)
        else:  # it fits --> updates (efficient)
            self._data['testvar_id'][self._n_elts:new_n_elts] = testvar_idx
            self._data['imgvar_id'][self._n_elts:new_n_elts] = (
                imgvar_idx + target_vars_offset)
            self._data['score'][self._n_elts:new_n_elts] = (
                perm_data[imgvar_idx, testvar_idx])
            self._data['perm_id'][self._n_elts:new_n_elts] = perm_id
            self._sizes[perm_id] += score_size
            self._n_elts = new_n_elts
        return


def f_score(vars1, vars2, covars, lost_dof):
    """Compute the F-score associated to the regression of vars2 against vars1

    Covariables are taken into account

    Parameters
    ----------
    vars1: array-like, shape=(n_samples, n_var1)
      Explanatory variables
    vars2: array-like, shape=(n_var2, n_samples)
      Targets variables
    covars, array-like, shape=(n_samples, n_covars)
      Confounding variables
    lost_dof: int,
      Lost degrees of freedom

    Returns
    -------
    score: array-like, shape=(n_var2, n_var1)
      F-scores associated to the tests of each explanatory variable against
      each target variable (in the presence of covars).

    """
    if not vars1.flags['C_CONTIGUOUS']:
        raise Exception('explanatory variables not C_CONTIGUOUS.')
    if not vars2.flags['C_CONTIGUOUS']:
        raise Exception('target variables not C_CONTIGUOUS.')
    if not covars.flags['C_CONTIGUOUS']:
        raise Exception('confounding variables not C_CONTIGUOUS.')
    beta_vars2_vars1 = np.dot(vars2, vars1)
    beta_vars2_covars = np.dot(vars2, covars)
    dof = vars2.shape[1] - 1 - lost_dof
    b_size = vars1.shape[1]
    b2 = beta_vars2_vars1 ** 2
    a2 = np.sum(beta_vars2_covars ** 2, 1)
    a2 = a2.repeat(b_size).reshape((-1, b_size))
    rss = (1 - a2 - b2)
    score = b2 / rss
    score *= dof
    return score


def _permuted_OLS_on_chunk(tested_vars, target_vars_chunk,
                           confounding_vars, n_perm,
                           target_vars_chunk_position=0, seed=0,
                           intercept_test=True):
    """Massively univariate group analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Explanatory variables.
    target_vars_chunk: array-like, shape=(n_targets, n_samples)
      fMRI data.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Clinical data (covariables).
    n_perm: int,
      Number of permutations
    target_vars_offset:
      offset corresponding to the target variables chunk position
    seed: int,
      Seed for random number generator, to have the same permutations
      in each computing units.
    intercept_test: boolean,
      Change the permutation scheme (swap signs for intercept,
      switch labels otherwise).

    """
    # initialize the seed of the random generator
    rng = np.random.RandomState(seed)

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
    sparsity_threshold = 1e-04
    sparsity_threshold = 0.5
    threshold = stats.f.isf(sparsity_threshold, 1, n_samples - lost_dof - 1)
    # We use a special data structure to store the results of the permutations
    # max_elts is used to preallocate memory
    max_elts = int(
        n_regressors * n_descriptors_chunk
        * (1 + (1.1 * n_perm * sparsity_threshold)))
    msarray = MULMSparseArray(
        n_perm + 1, max_elts=max_elts, threshold=threshold)
    # add original data results as permutation 0
    msarray.append_perm_data(perm_id=0, perm_data=score_original_data,
                             target_vars_offset=target_vars_chunk_position)

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
        msarray.append_perm_data(
            perm_id=i, perm_data=cur_res,
            target_vars_offset=target_vars_chunk_position)

    params = {'lost_dof': lost_dof, 'threshold': threshold,
              'n_perm': n_perm, 'n_subj': n_samples}
    return msarray, params
