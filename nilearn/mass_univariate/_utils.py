"""Utility functions for the permuted_least_squares module."""
import numpy as np
from scipy import ndimage


def _null_to_p(test_value, null_array, tail='two', symmetric=False):
    """Return p-value for test value(s) against null array.

    Parameters
    ----------
    test_value : array_like of shape (n_samples,)
        Values for which to determine p-value.
    null_array : array_like of shape (n_iters,)
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    symmetric : :obj:`bool`, optional
        When tail="two", indicates how to compute p-values.
        When False (default), both one-tailed p-values are computed,
        and the two-tailed p is double the minimum one-tailed p.
        When True, it is assumed that the null distribution is zero-centered
        and symmetric, and the two-tailed p-value is computed as
        P(abs(test_value) >= abs(null_array)).
        Default=False.

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
    arr4d : :obj:`numpy.ndarray` of shape (X, Y, Z, T)
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
