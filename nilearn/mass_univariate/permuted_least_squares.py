"""
Massively Univariate Linear Model estimated with OLS and permutation test.
"""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
#         Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import sys
import time
import warnings

import joblib
import nibabel as nib
import numpy as np
from scipy import linalg, ndimage, stats
from sklearn.utils import check_random_state


def _null_to_p(test_value, null_array, tail='two', symmetric=False):
    """Return p-value for test value(s) against null array.

    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    symmetric : bool
        When tail="two", indicates how to compute p-values.
        When False (default), both one-tailed p-values are computed,
        and the two-tailed p is double the minimum one-tailed p.
        When True, it is assumed that the null distribution is zero-centered
        and symmetric, and the two-tailed p-value is computed as
        P(abs(test_value) >= abs(null_array)).

    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the
        null distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).

    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.

    When the null distribution is known to be symmetric and centered on zero,
    and two-tailed p-values are desired, use symmetric=True, as it is
    approximately twice as efficient computationally, and has lower variance.
    """
    if tail not in {'two', 'upper', 'lower'}:
        raise ValueError(
            'Argument "tail" must be one of ["two", "upper", "lower"]'
        )

    return_first = isinstance(test_value, (float, int))
    test_value = np.atleast_1d(test_value)
    null_array = np.array(null_array)

    # For efficiency's sake, if there are more than 1000 values, pass only the
    # unique values through percentileofscore(), and then reconstruct.
    if len(test_value) > 1000:
        reconstruct = True
        test_value, uniq_idx = np.unique(test_value, return_inverse=True)
    else:
        reconstruct = False

    def compute_p(t, null):
        null = np.sort(null)
        idx = np.searchsorted(null, t, side='left').astype(float)
        return 1 - idx / len(null)

    if tail == 'two':
        if symmetric:
            p = compute_p(np.abs(test_value), np.abs(null_array))
        else:
            p_l = compute_p(test_value, null_array)
            p_r = compute_p(test_value * -1, null_array * -1)
            p = 2 * np.minimum(p_l, p_r)
    elif tail == 'lower':
        p = compute_p(test_value * -1, null_array * -1)
    else:
        p = compute_p(test_value, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result[0] if return_first else result


def _calculate_cluster_measures(arr4d, threshold, conn, two_sided_test=False):
    """Calculate maximum cluster mass and size for an array.

    This method assesses both positive and negative clusters.

    Parameters
    ----------
    arr4d : :obj:`numpy.ndarray`
        Unthresholded 4D array of 3D t-statistic maps.
    threshold : :obj:`float`
        Uncorrected t-statistic threshold for defining clusters.
    conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.

    Returns
    -------
    max_size, max_mass : :obj:`float`
        Maximum cluster size and mass from the matrix.
    """
    n_regressors = arr4d.shape[3]

    max_sizes = np.zeros(n_regressors, int)
    max_masses = np.zeros(n_regressors, float)

    for i_regressor in range(n_regressors):
        arr3d = arr4d[..., i_regressor]

        if two_sided_test:
            arr3d[np.abs(arr3d) <= threshold] = 0
        else:
            arr3d[arr3d <= threshold] = 0

        labeled_arr3d, _ = ndimage.measurements.label(arr3d > 0, conn)

        if two_sided_test:
            # Label positive and negative clusters separately
            n_positive_clusters = np.max(labeled_arr3d)
            temp_labeled_arr3d, _ = ndimage.measurements.label(arr3d < 0, conn)
            temp_labeled_arr3d[temp_labeled_arr3d > 0] += n_positive_clusters
            labeled_arr3d = labeled_arr3d + temp_labeled_arr3d
            del temp_labeled_arr3d

        clust_vals, clust_sizes = np.unique(labeled_arr3d, return_counts=True)
        assert clust_vals[0] == 0

        clust_vals = clust_vals[1:]  # First cluster is zeros in matrix
        clust_sizes = clust_sizes[1:]

        # Cluster mass-based inference
        max_mass = 0
        for unique_val in clust_vals:
            ss_vals = np.abs(arr3d[labeled_arr3d == unique_val]) - threshold
            max_mass = np.maximum(max_mass, np.sum(ss_vals))

        # Cluster size-based inference
        if clust_sizes.size:
            max_size = np.max(clust_sizes)
        else:
            max_size = 0

        max_sizes[i_regressor], max_masses[i_regressor] = max_size, max_mass

    return max_sizes, max_masses


def _normalize_matrix_on_axis(m, axis=0):
    """ Normalize a 2D matrix on an axis.

    Parameters
    ----------
    m : numpy 2D array,
      The matrix to normalize.

    axis : integer in {0, 1}, optional
      A valid axis to normalize across.
      Default=0.

    Returns
    -------
    ret : numpy array, shape = m.shape
      The normalized matrix

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     _normalize_matrix_on_axis)
    >>> X = np.array([[0, 4], [1, 0]])
    >>> _normalize_matrix_on_axis(X)
    array([[0., 1.],
           [1., 0.]])
    >>> _normalize_matrix_on_axis(X, axis=1)
    array([[0., 1.],
           [1., 0.]])

    """
    if m.ndim > 2:
        raise ValueError('This function only accepts 2D arrays. '
                         'An array of shape %r was passed.' % m.shape)

    if axis == 0:
        # array transposition preserves the contiguity flag of that array
        ret = (m.T / np.sqrt(np.sum(m ** 2, axis=0))[:, np.newaxis]).T
    elif axis == 1:
        ret = _normalize_matrix_on_axis(m.T).T
    else:
        raise ValueError('axis(=%d) out of bounds' % axis)
    return ret


def _orthonormalize_matrix(m, tol=1.e-12):
    """ Orthonormalize a matrix.

    Uses a Singular Value Decomposition.
    If the input matrix is rank-deficient, then its shape is cropped.

    Parameters
    ----------
    m : numpy array,
      The matrix to orthonormalize.

    tol: float, optional
      Tolerance parameter for nullity. Default=1e-12.

    Returns
    -------
    ret : numpy array, shape = m.shape
      The orthonormalized matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     _orthonormalize_matrix)
    >>> X = np.array([[1, 2], [0, 1], [1, 1]])
    >>> _orthonormalize_matrix(X)
    array([[-0.81049889, -0.0987837 ],
           [-0.31970025, -0.75130448],
           [-0.49079864,  0.65252078]])
    >>> X = np.array([[0, 1], [4, 0]])
    >>> _orthonormalize_matrix(X)
    array([[ 0., -1.],
           [-1.,  0.]])

    """
    U, s, _ = linalg.svd(m, full_matrices=False)
    n_eig = np.count_nonzero(s > tol)
    return np.ascontiguousarray(U[:, :n_eig])


def _t_score_with_covars_and_normalized_design(tested_vars, target_vars,
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

    covars_orthonormalized : array-like, shape=(n_samples, n_covars) or None, optional
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
    beta_targetvars_testedvars = np.dot(target_vars.T, tested_vars)
    if covars_orthonormalized is None:
        rss = (1 - beta_targetvars_testedvars ** 2)
    else:
        beta_targetvars_covars = np.dot(target_vars.T, covars_orthonormalized)
        a2 = np.sum(beta_targetvars_covars ** 2, 1)
        rss = (1 - a2[:, np.newaxis] - beta_targetvars_testedvars ** 2)
    return beta_targetvars_testedvars * np.sqrt((dof - 1.) / rss)


def _permuted_ols_on_chunk(
    scores_original_data,
    tested_vars,
    target_vars,
    thread_id,
    threshold,
    confounding_vars=None,
    masker=None,
    n_perm=10000,
    n_perm_chunk=10000,
    intercept_test=True,
    two_sided_test=True,
    random_state=None,
    verbose=0,
):
    """Perform massively univariate analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    scores_original_data : array-like, shape=(n_descriptors, n_regressors)
        t-scores obtained for the original (non-permuted) data.

    tested_vars : array-like, shape=(n_samples, n_regressors)
        Explanatory variates.

    target_vars : array-like, shape=(n_samples, n_targets)
        fMRI data. F-ordered for efficient computations.

    thread_id : int
        process id, used for display.

    threshold : :obj:`float`
        Cluster-forming threshold in t-scale.
        This is only used for cluster-level inference.

    confounding_vars : array-like, shape=(n_samples, n_covars), optional
        Clinical data (covariates).

    masker

    n_perm : int, optional
        Total number of permutations to perform, only used for
        display in this function. Default=10000.

    n_perm_chunk : int, optional
        Number of permutations to be performed. Default=10000.

    intercept_test : boolean, optional
        Change the permutation scheme (swap signs for intercept,
        switch labels otherwise). See [1]_.
        Default=True.

    two_sided_test : boolean, optional
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.
        Default=True

    random_state : int or None, optional
        Seed for random number generator, to have the same permutations
        in each computing units.

    verbose : int, optional
        Defines the verbosity level. Default=0.

    Returns
    -------
    scores_as_ranks_part : array-like, shape=(n_regressors, n_descriptors)
        The ranks of the original scores in h0_fmax_part.
        When ``n_descriptors`` or ``n_perm`` are large, it can be quite long to
        find the rank of the original scores into the whole H0 distribution.
        Here, it is performed in parallel by the workers involved in the
        permutation computation.

    h0_fmax_part : array-like, shape=(n_perm_chunk, n_regressors)
        Distribution of the (max) t-statistic under the null hypothesis
        (limited to this permutation chunk).

    References
    ----------
    .. [1] Fisher, R. A. (1935). The design of experiments.

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    n_samples, n_regressors = tested_vars.shape
    n_descriptors = target_vars.shape[1]

    # run the permutations
    t0 = time.time()
    h0_vfwe_part = np.empty((n_regressors, n_perm_chunk))
    h0_csfwe_part = np.empty((n_regressors, n_perm_chunk))
    h0_cmfwe_part = np.empty((n_regressors, n_perm_chunk))
    vfwe_scores_as_ranks_part = np.zeros((n_regressors, n_descriptors))

    for i_perm in range(n_perm_chunk):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            target_vars = (
                target_vars * (rng.randint(2, size=(n_samples, 1)) * 2 - 1)
            )
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
            _t_score_with_covars_and_normalized_design(
                tested_vars, target_vars, confounding_vars
            )
        )
        if two_sided_test:
            perm_scores = np.fabs(perm_scores)

        h0_vfwe_part[:, i_perm] = np.nanmax(perm_scores, axis=0)

        # TODO: Eliminate need for transpose
        arr4d = masker.inverse_transform(perm_scores.T).get_fdata()
        conn = ndimage.generate_binary_structure(3, 1)
        (
            h0_csfwe_part[:, i_perm],
            h0_cmfwe_part[:, i_perm],
        ) = _calculate_cluster_measures(
            arr4d,
            threshold,
            conn,
            two_sided_test=two_sided_test,
        )

        # find the rank of the original scores in h0_vfwe_part
        # (when n_descriptors or n_perm are large, it can be quite long to
        #  find the rank of the original scores into the whole H0 distribution.
        #  Here, it is performed in parallel by the workers involved in the
        #  permutation computation)
        # NOTE: This is not done for the cluster-level methods.
        vfwe_scores_as_ranks_part += (
            h0_vfwe_part[:, i_perm].reshape((-1, 1)) < scores_original_data.T
        )

        if verbose > 0:
            step = 11 - min(verbose, 10)
            if i_perm % step == 0:
                # If there is only one job, progress information is fixed
                if n_perm == n_perm_chunk:
                    crlf = "\r"
                else:
                    crlf = "\n"

                percent = float(i_perm) / n_perm_chunk
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                remaining = (100. - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i_perm}/{n_perm_chunk} "
                    f"permutations ({percent:0.2f}%, {remaining} seconds "
                    f"remaining){crlf}"
                )

    return (
        vfwe_scores_as_ranks_part,
        h0_vfwe_part,
        h0_csfwe_part,
        h0_cmfwe_part,
    )


def permuted_ols(
    tested_vars,
    target_vars,
    confounding_vars=None,
    model_intercept=True,
    masker=None,
    threshold=0.001,
    n_perm=10000,
    two_sided_test=True,
    random_state=None,
    n_jobs=1,
    verbose=0,
):
    """Massively univariate group analysis with permuted OLS.

    Tested variates are independently fitted to target variates descriptors
    (e.g. brain imaging signal) according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variates may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variates and the target variates [1]_, [2]_.
    A max-type procedure is used to obtain family-wise corrected p-values.

    The specific permutation scheme implemented here is the one of
    [3]_. Its has been demonstrated in [1]_ that this
    scheme conveys more sensitivity than alternative schemes. This holds
    for neuroimaging applications, as discussed in details in [2]_.

    Permutations are performed on parallel computing units. Each of them
    performs a fraction of permutations on the whole dataset. Thus, the max
    t-score amongst data descriptors can be computed directly, which avoids
    storing all the computed t-scores.

    The variates should be given C-contiguous. ``target_vars`` are
    fortran-ordered automatically to speed-up computations.

    Parameters
    ----------
    tested_vars : array-like, shape=(n_samples, n_regressors)
        Explanatory variates, fitted and tested independently from each others.

    target_vars : array-like, shape=(n_samples, n_descriptors)
        fMRI data to analyze according to the explanatory and confounding
        variates.

        In a group-level analysis, the samples will typically be voxels
        (for volumetric data) or vertices (for surface data), while the
        descriptors will generally be images, such as run-wise z-statistic
        maps.

    confounding_vars : array-like, shape=(n_samples, n_covars), optional
        Confounding variates (covariates), fitted but not tested.
        If None, no confounding variate is added to the model
        (except maybe a constant column according to the value of
        ``model_intercept``).

    model_intercept : :obj:`bool`, optional
        If True, a constant column is added to the confounding variates
        unless the tested variate is already the intercept.
        Default=True.

    masker

    threshold : :obj:`float`, optional
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        Default=0.001.

    n_perm : :obj:`int`, optional
        Number of permutations to perform.
        Permutations are costly but the more are performed, the more precision
        one gets in the p-values estimation.
        If ``n_perm`` is set to 0, then no p-values will be estimated.
        Default=10000.

    two_sided_test : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative. Default=True.

    random_state : :obj:`int` or None, optional
        Seed for random number generator, to have the same permutations
        in each computing units.

    n_jobs : :obj:`int`, optional
        Number of parallel workers.
        If 0 is provided, all CPUs are used.
        A negative number indicates that all the CPUs except (abs(n_jobs) - 1)
        ones will be used. Default=1.

    verbose : :obj:`int`, optional
        verbosity level (0 means no message). Default=0.

    Returns
    -------
    neg_log10_pvals : array-like, shape=(n_regressors, n_descriptors)
        Negative log10 p-values associated with the significance test of the
        ``n_regressors`` explanatory variates against the ``n_descriptors``
        target variates. Family-wise corrected p-values.
        This will be an empty array of ``n_perms`` is 0.

    score_orig_data : numpy.ndarray, shape=(n_regressors, n_descriptors)
        T-statistics associated with the significance test of the
        ``n_regressors`` explanatory variates against the ``n_descriptors``
        target variates.
        The ranks of the scores into the h0 distribution correspond to the
        p-values.

    h0_fmax : array-like, shape=(n_regressors, n_perm)
        Distribution of the (max) t-statistic under the null hypothesis
        (obtained from the permutations). Array is sorted.

    References
    ----------
    .. [1] Anderson, M. J. & Robinson, J. (2001). Permutation tests for
       linear models. Australian & New Zealand Journal of Statistics, 43(1),
       75-88.

    .. [2] Winkler, A. M. et al. (2014). Permutation inference for the general
       linear model. Neuroimage.

    .. [3] Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of
       reported significance levels. J. Bus. Econ. Stats., 1(4), 292-298

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # check n_jobs (number of CPUs)
    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError(
            "'n_jobs == 0' is not a valid choice. "
            "Please provide a positive number of CPUs, or -1 for all CPUs, "
            "or a negative number (-i) for 'all but (i-1)' CPUs "
            "(joblib conventions)."
        )
    elif n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() - int(n_jobs) + 1)
    else:
        n_jobs = min(n_jobs, joblib.cpu_count())

    # make target_vars F-ordered to speed-up computation
    if target_vars.ndim != 2:
        raise ValueError(
            "'target_vars' should be a 2D array. "
            f"An array with {target_vars.ndim} dimension(s) was passed."
        )

    target_vars = np.asfortranarray(target_vars)  # efficient for chunking
    n_descriptors = target_vars.shape[1]
    if np.any(np.all(target_vars == 0, axis=0)):
        warnings.warn(
            "Some descriptors in 'target_vars' have zeros across all samples. "
            "These descriptors will be ignored during null distribution "
            "generation."
        )

    # check explanatory variates' dimensions
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
        covars_orthonormalized = _orthonormalize_matrix(confounding_vars)
        if not covars_orthonormalized.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Confounding variates not C_CONTIGUOUS.')
            covars_orthonormalized = np.ascontiguousarray(
                covars_orthonormalized)

        targetvars_normalized = _normalize_matrix_on_axis(
            target_vars).T  # faster with F-ordered target_vars_chunk
        if not targetvars_normalized.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Target variates not C_CONTIGUOUS.')
            targetvars_normalized = np.ascontiguousarray(targetvars_normalized)

        beta_targetvars_covars = np.dot(targetvars_normalized,
                                        covars_orthonormalized)
        targetvars_resid_covars = targetvars_normalized - np.dot(
            beta_targetvars_covars, covars_orthonormalized.T)
        targetvars_resid_covars = _normalize_matrix_on_axis(
            targetvars_resid_covars, axis=1)

        # step 2: extract effect of covars from tested vars
        testedvars_normalized = _normalize_matrix_on_axis(
            tested_vars.T, axis=1
        )
        beta_testedvars_covars = np.dot(testedvars_normalized,
                                        covars_orthonormalized)
        testedvars_resid_covars = testedvars_normalized - np.dot(
            beta_testedvars_covars, covars_orthonormalized.T)
        testedvars_resid_covars = _normalize_matrix_on_axis(
            testedvars_resid_covars, axis=1).T.copy()

        n_covars = confounding_vars.shape[1]

    else:
        targetvars_resid_covars = _normalize_matrix_on_axis(target_vars).T
        testedvars_resid_covars = _normalize_matrix_on_axis(tested_vars).copy()
        covars_orthonormalized = None
        n_covars = 0

    # check arrays contiguousity (for the sake of code efficiency)
    if not targetvars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Target variates not C_CONTIGUOUS.')
        targetvars_resid_covars = np.ascontiguousarray(targetvars_resid_covars)

    if not testedvars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Tested variates not C_CONTIGUOUS.')
        testedvars_resid_covars = np.ascontiguousarray(testedvars_resid_covars)

    # step 3: original regression (= regression on residuals + adjust t-score)
    # compute t score map of each tested var for original data
    # scores_original_data is in samples-by-regressors shape
    scores_original_data = _t_score_with_covars_and_normalized_design(
        testedvars_resid_covars,
        targetvars_resid_covars.T,
        covars_orthonormalized,
    )

    # determine t-statistic threshold
    # NOTE: This needs to be adjusted for two-sided tests
    dof = n_samples - (n_regressors + n_covars)
    threshold_t = stats.t.isf(threshold, df=dof)

    if two_sided_test:
        # TODO: Retain original signs in permutations, to measure cluster
        # sizes/masses separately for positive and negative clusters.

        # Ensure that all t-statistics are positive for permutation tests,
        # so that ranks reflect significance
        sign_scores_original_data = np.sign(scores_original_data)
        scores_original_data = np.fabs(scores_original_data)

    ### Permutations
    # parallel computing units perform a reduced number of permutations each
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)
        n_perm_chunks[-1] += n_perm % n_jobs

    elif n_perm > 0:
        warnings.warn(
            f'The specified number of permutations is {n_perm} and the number '
            f'of jobs to be performed in parallel has set to {n_jobs}. '
            f'This is incompatible so only {n_perm} jobs will be running. '
            'You may want to perform more permutations in order to take the '
            'most of the available computing resources.'
        )
        n_perm_chunks = np.ones(n_perm, dtype=int)

    else:  # 0 or negative number of permutations => original data scores only
        if two_sided_test:
            scores_original_data *= sign_scores_original_data

        return np.asarray([]), scores_original_data.T, np.asarray([])

    # actual permutations, seeded from a random integer between 0 and maximum
    # value represented by np.int32 (to have a large entropy).
    ret = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_permuted_ols_on_chunk)(
            scores_original_data,
            testedvars_resid_covars,
            targetvars_resid_covars.T,
            thread_id=thread_id + 1,
            threshold=threshold_t,
            confounding_vars=covars_orthonormalized,
            masker=masker,
            n_perm=n_perm,
            n_perm_chunk=n_perm_chunk,
            intercept_test=intercept_test,
            two_sided_test=two_sided_test,
            random_state=rng.randint(1, np.iinfo(np.int32).max - 1),
            verbose=verbose,
        )
        for thread_id, n_perm_chunk in enumerate(n_perm_chunks))

    # reduce results
    (
        vfwe_scores_as_ranks_parts,
        vfwe_h0_parts,
        csfwe_h0_parts,
        cmfwe_h0_parts,
    ) = zip(*ret)

    # Voxel-level FWE
    vfwe_h0 = np.hstack((vfwe_h0_parts))
    vfwe_scores_as_ranks = np.zeros((n_regressors, n_descriptors))
    for vfwe_scores_as_ranks_part in vfwe_scores_as_ranks_parts:
        vfwe_scores_as_ranks += vfwe_scores_as_ranks_part

    vfwe_pvals = (n_perm + 1 - vfwe_scores_as_ranks) / float(1 + n_perm)

    # Cluster-size and cluster-mass FWE
    csfwe_h0 = np.hstack((csfwe_h0_parts))
    cmfwe_h0 = np.hstack((cmfwe_h0_parts))

    csfwe_pvals = np.zeros_like(vfwe_pvals)
    cmfwe_pvals = np.zeros_like(vfwe_pvals)

    # TODO: Eliminate need to transpose
    scores_original_data_4d = masker.inverse_transform(
        scores_original_data.T
    ).get_fdata()
    conn = ndimage.generate_binary_structure(3, 1)

    for i_regressor in range(n_regressors):
        scores_original_data_3d = scores_original_data_4d[..., i_regressor]

        # Label the clusters
        labeled_arr3d, _ = ndimage.measurements.label(
            scores_original_data_3d > threshold_t,
            conn,
        )

        if two_sided_test:
            # Label positive and negative clusters separately
            n_positive_clusters = np.max(labeled_arr3d)
            temp_labeled_arr3d, _ = ndimage.measurements.label(
                scores_original_data_3d < -threshold_t,
                conn,
            )
            temp_labeled_arr3d[
                temp_labeled_arr3d > threshold_t
            ] += n_positive_clusters
            labeled_arr3d = labeled_arr3d + temp_labeled_arr3d
            del temp_labeled_arr3d

        cluster_labels, idx, cluster_sizes = np.unique(
            labeled_arr3d,
            return_inverse=True,
            return_counts=True,
        )
        assert cluster_labels[0] == 0  # the background

        # Cluster mass-based inference
        cluster_masses = np.zeros(cluster_labels.shape)
        for j_val in cluster_labels[1:]:  # skip background
            cluster_mass = np.sum(
                scores_original_data_3d[labeled_arr3d == j_val] - threshold_t
            )
            cluster_masses[j_val] = cluster_mass

        p_cmfwe_vals = _null_to_p(
            cluster_masses,
            cmfwe_h0[i_regressor, :],
            'upper',
        )
        p_cmfwe_map = p_cmfwe_vals[np.reshape(idx, labeled_arr3d.shape)]

        # Convert 3D to image, then to 1D
        cmfwe_pvals[i_regressor, :] = np.squeeze(
            masker.transform(
                nib.Nifti1Image(
                    p_cmfwe_map,
                    masker.mask_img_.affine,
                    masker.mask_img_.header,
                )
            )
        )

        # Cluster size-based inference
        cluster_sizes[0] = 0  # replace background's "cluster size" with zeros
        p_csfwe_vals = _null_to_p(
            cluster_sizes,
            csfwe_h0[i_regressor, :],
            'upper',
        )
        p_csfwe_map = p_csfwe_vals[np.reshape(idx, labeled_arr3d.shape)]
        csfwe_pvals[i_regressor, :] = np.squeeze(
            masker.transform(
                nib.Nifti1Image(
                    p_csfwe_map,
                    masker.mask_img_.affine,
                    masker.mask_img_.header,
                )
            )
        )

    # put back sign on scores if it was removed in the case of a two-sided test
    # (useful to distinguish between positive and negative effects)
    if two_sided_test:
        scores_original_data = scores_original_data * sign_scores_original_data

    return (
        -np.log10(vfwe_pvals),
        -np.log10(csfwe_pvals),
        -np.log10(cmfwe_pvals),
        scores_original_data.T,
        vfwe_h0[0],
        csfwe_h0[0],
        cmfwe_h0[0],
    )
