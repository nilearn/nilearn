"""Utility functions for the permuted_least_squares module."""
import numpy as np
from scipy import ndimage


def _null_to_p(test_values, null_array, alternative='two-sided'):
    """Return p-value for test value(s) against null array.

    Parameters
    ----------
    test_values : :obj:`int`, :obj:`float`, or array_like of shape (n_samples,)
        Value(s) for which to determine p-value.
    null_array : array_like of shape (n_iters,)
        Null distribution against which test_values is compared.
    alternative : {'two-sided', 'larger', 'smaller'}, optional
        Whether to compare value against null distribution in a two-sided
        or one-sided ('larger' or 'smaller') manner. If 'larger', then higher
        values for the test_values are more significant. If 'smaller', then
        lower values for the test_values are more significant.
        Default is 'two-sided'.

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
    if alternative not in {'two-sided', 'larger', 'smaller'}:
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
        idx = np.searchsorted(null, t, side='left').astype(float)
        return 1 - idx / len(null)

    if alternative == 'two-sided':
        # Assumes null distribution is symmetric around 0
        p = compute_p(np.abs(test_values), np.abs(null_array))
    elif alternative == 'smaller':
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


def _calculate_cluster_measures(arr4d, threshold, conn, two_sided_test=False):
    """Calculate maximum cluster mass and size for an array.

    Parameters
    ----------
    arr4d : :obj:`numpy.ndarray` of shape (X, Y, Z, R)
        Unthresholded 4D array of 3D t-statistic maps.
        R = regressor.
    threshold : :obj:`float`
        Uncorrected t-statistic threshold for defining clusters.
    conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.
    two_sided_test : :obj:`bool`, optional
        Whether to assess both positive and negative clusters (True) or just
        positive ones (False).
        Default is False.

    Returns
    -------
    max_size, max_mass : :obj:`numpy.ndarray` of shape (n_regressors,)
        Maximum cluster size and mass from the matrix, for each regressor.
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
