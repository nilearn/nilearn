"""Utility functions for the permuted least squares method."""

import numpy as np
from scipy import linalg
from scipy.ndimage import label


def calculate_tfce(
    arr4d,
    bin_struct,
    E=0.5,
    H=2,
    dh="auto",
    two_sided_test=True,
):
    """Calculate threshold-free cluster enhancement values for scores maps.

    The :term:`TFCE` calculation is mostly implemented as described in [1]_,
    with minor modifications to produce similar results to fslmaths, as well
    as to support two-sided testing.

    Parameters
    ----------
    arr4d : :obj:`numpy.ndarray` of shape (X, Y, Z, R)
        Unthresholded 4D array of 3D t-statistic maps.
        R = regressor.
    bin_struct : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.
    E : :obj:`float`, default=0.5
        Extent weight.
    H : :obj:`float`, default=2
        Height weight.
    dh : 'auto' or :obj:`float`, default='auto'
        Step size for TFCE calculation.
        If set to 'auto', use 100 steps, as is done in fslmaths.
        A good alternative is 0.1 for z and t maps, as in [1]_.
    two_sided_test : :obj:`bool`, default=False
        Whether to assess both positive and negative clusters (True) or just
        positive ones (False).

    Returns
    -------
    tfce_arr : :obj:`numpy.ndarray`, shape=(n_descriptors, n_regressors)
        :term:`TFCE` values.

    Notes
    -----
    In [1]_, each threshold's partial TFCE score is multiplied by dh,
    which makes directly comparing TFCE values across different thresholds
    possible.
    However, in fslmaths, this is not done.
    In the interest of maximizing similarity between nilearn and established
    tools, we chose to follow fslmaths' approach.

    Additionally, we have modified the method to support two-sided testing.
    In fslmaths, only positive clusters are considered.

    References
    ----------
    .. [1] Smith, S. M., & Nichols, T. E. (2009).
       Threshold-free cluster enhancement: addressing problems of smoothing,
       threshold dependence and localisation in cluster inference.
       Neuroimage, 44(1), 83-98.
    """
    tfce_4d = np.zeros_like(arr4d)

    # For each passed t map
    for i_regressor in range(arr4d.shape[3]):
        arr3d = arr4d[..., i_regressor]

        # Get signs / threshs
        if two_sided_test:
            signs = [-1, 1]
            max_score = np.max(np.abs(arr3d))
        else:
            signs = [1]
            max_score = np.max(arr3d)

        step = max_score / 100 if dh == "auto" else dh

        # Set based on determined step size
        score_threshs = np.arange(step, max_score + step, step)

        # If we apply the sign first...
        for sign in signs:
            # Init a temp copy of arr3d with the current sign applied,
            # which can then be reused by incrementally setting more
            # voxel's to background, by taking advantage that each score_thresh
            # is incrementally larger
            temp_arr3d = arr3d * sign

            # Prep step
            for score_thresh in score_threshs:
                temp_arr3d[temp_arr3d < score_thresh] = 0

                # Label into clusters - importantly (for the next step)
                # this returns clusters labeled ordinally
                # from 1 to n_clusters+1,
                # which allows us to use bincount to count
                # frequencies directly.
                labeled_arr3d, _ = label(temp_arr3d, bin_struct)

                # Next, we want to replace each label with its cluster
                # extent, that is, the size of the cluster it is part of
                # To do this, we will first compute a flattened version of
                # only the non-zero cluster labels.
                labeled_arr3d_flat = labeled_arr3d.flatten()
                non_zero_inds = np.where(labeled_arr3d_flat != 0)[0]
                labeled_non_zero = labeled_arr3d_flat[non_zero_inds]

                # Count the size of each unique cluster, via its label.
                # The reason why we pass only the non-zero labels to bincount
                # is because it includes a bin for zeros, and in our labels
                # zero represents the background,
                # which we want to have a TFCE value of 0.
                cluster_counts = np.bincount(labeled_non_zero)

                # Next, we convert each unique cluster count to its TFCE value.
                # Where each cluster's tfce value is based
                # on both its cluster extent and z-value
                # (via the current score_thresh)
                # NOTE: We do not multiply by dh, based on fslmaths'
                # implementation. This differs from the original paper.
                cluster_tfces = sign * (cluster_counts**E) * (score_thresh**H)

                # Before we can add these values to tfce_4d, we need to
                # map cluster-wise tfce values back to a voxel-wise array,
                # including any zero / background voxels.
                tfce_step_values = np.zeros(labeled_arr3d_flat.shape)
                tfce_step_values[non_zero_inds] = cluster_tfces[
                    labeled_non_zero
                ]

                # Now, we just need to reshape these values back to 3D
                # and they can be incremented to tfce_4d.
                tfce_4d[..., i_regressor] += tfce_step_values.reshape(
                    temp_arr3d.shape
                )

    return tfce_4d


def null_to_p(test_values, null_array, alternative="two-sided"):
    """Return p-value for test value(s) against null array.

    Parameters
    ----------
    test_values : :obj:`int`, :obj:`float`, or array_like of shape (n_samples,)
        Value(s) for which to determine p-value.
    null_array : array_like of shape (n_iters,)
        Null distribution against which test_values is compared.
    alternative : {'two-sided', 'larger', 'smaller'}, default='two-sided'
        Whether to compare value against null distribution in a two-sided
        or one-sided ('larger' or 'smaller') manner. If 'larger', then higher
        values for the test_values are more significant. If 'smaller', then
        lower values for the test_values are more significant.

    Returns
    -------
    p_values : :obj:`float` or array_like of shape (n_samples,)
        P-value(s) associated with the test value when compared against the
        null distribution. Return type matches input type (i.e., a float if
        test_values is a single float, and an array if test_values is an
        array).

    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.

    This function assumes that the null distribution for two-sided tests is
    symmetric around zero.
    """
    if alternative not in {"two-sided", "larger", "smaller"}:
        raise ValueError(
            'Argument "alternative" must be one of '
            '["two-sided", "larger", "smaller"]'
        )

    return_first = isinstance(test_values, (float, int))
    test_values = np.atleast_1d(test_values)
    null_array = np.array(null_array)

    # For efficiency's sake, if there are more than 1000 values, pass only the
    # unique values through percentileofscore(), and then reconstruct.
    if len(test_values) > 1000:
        reconstruct = True
        test_values, uniq_idx = np.unique(test_values, return_inverse=True)
    else:
        reconstruct = False

    def compute_p(t, null):
        null = np.sort(null)
        idx = np.searchsorted(null, t, side="left").astype(float)
        return 1 - idx / len(null)

    if alternative == "two-sided":
        # Assumes null distribution is symmetric around 0
        p = compute_p(np.abs(test_values), np.abs(null_array))
    elif alternative == "smaller":
        p = compute_p(test_values * -1, null_array * -1)
    else:
        p = compute_p(test_values, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result[0] if return_first else result


def calculate_cluster_measures(
    arr4d,
    threshold,
    bin_struct,
    two_sided_test=False,
):
    """Calculate maximum cluster mass and size for an array.

    Parameters
    ----------
    arr4d : :obj:`numpy.ndarray` of shape (X, Y, Z, R)
        Unthresholded 4D array of 3D t-statistic maps.
        R = regressor.
    threshold : :obj:`float`
        Uncorrected t-statistic threshold for defining clusters.
    bin_struct : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.
    two_sided_test : :obj:`bool`, default=False
        Whether to assess both positive and negative clusters (True) or just
        positive ones (False).

    Returns
    -------
    max_size, max_mass : :obj:`numpy.ndarray` of shape (n_regressors,)
        Maximum cluster size and mass from the matrix, for each regressor.
    """
    n_regressors = arr4d.shape[3]

    max_sizes = np.zeros(n_regressors, int)
    max_masses = np.zeros(n_regressors, float)

    for i_regressor in range(n_regressors):
        arr3d = arr4d[..., i_regressor].copy()

        if two_sided_test:
            arr3d[np.abs(arr3d) <= threshold] = 0
        else:
            arr3d[arr3d <= threshold] = 0

        labeled_arr3d, _ = label(arr3d > 0, bin_struct)

        if two_sided_test:
            # Label positive and negative clusters separately
            n_positive_clusters = np.max(labeled_arr3d)
            temp_labeled_arr3d, _ = label(
                arr3d < 0,
                bin_struct,
            )
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
        max_size = 0
        if clust_sizes.size:
            max_size = np.max(clust_sizes)

        max_sizes[i_regressor], max_masses[i_regressor] = max_size, max_mass

    return max_sizes, max_masses


def normalize_matrix_on_axis(m, axis=0):
    """Normalize a 2D matrix on an axis.

    Parameters
    ----------
    m : numpy 2D array,
        The matrix to normalize.

    axis : integer in {0, 1}, default=0
        A valid axis to normalize across.

    Returns
    -------
    ret : numpy array, shape = m.shape
        The normalized matrix

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     normalize_matrix_on_axis,
    ... )
    >>> X = np.array([[0, 4], [1, 0]])
    >>> normalize_matrix_on_axis(X)
    array([[0., 1.],
           [1., 0.]])
    >>> normalize_matrix_on_axis(X, axis=1)
    array([[0., 1.],
           [1., 0.]])

    """
    if m.ndim > 2:
        raise ValueError(
            "This function only accepts 2D arrays. "
            f"An array of shape {m.shape:r} was passed."
        )

    if axis == 0:
        # array transposition preserves the contiguity flag of that array
        ret = (m.T / np.sqrt(np.sum(m**2, axis=0))[:, np.newaxis]).T
    elif axis == 1:
        ret = normalize_matrix_on_axis(m.T).T
    else:
        raise ValueError(f"axis(={int(axis)}) out of bounds")
    return ret


def orthonormalize_matrix(m, tol=1.0e-12):
    """Orthonormalize a matrix.

    Uses a Singular Value Decomposition.
    If the input matrix is rank-deficient, then its shape is cropped.

    Parameters
    ----------
    m : numpy array,
        The matrix to orthonormalize.

    tol : float, default=1e-12
        Tolerance parameter for nullity.

    Returns
    -------
    ret : numpy array, shape = m.shape
        The orthonormalized matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.mass_univariate.permuted_least_squares import (
    ...     orthonormalize_matrix,
    ... )
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


def t_score_with_covars_and_normalized_design(
    tested_vars, target_vars, covars_orthonormalized=None
):
    """t-score in the regression of tested variates against target variates.

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

    covars_orthonormalized : array-like, shape=(n_samples, n_covars) or None, \
            optional
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
        rss = 1 - beta_targetvars_testedvars**2
    else:
        beta_targetvars_covars = np.dot(target_vars.T, covars_orthonormalized)
        a2 = np.sum(beta_targetvars_covars**2, 1)
        rss = 1 - a2[:, np.newaxis] - beta_targetvars_testedvars**2
    return beta_targetvars_testedvars * np.sqrt((dof - 1.0) / rss)
