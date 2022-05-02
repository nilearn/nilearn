"""Utility functions for the permuted least squares method."""
import numpy as np
from scipy import ndimage


def _calculate_tfce(
    arr4d,
    bin_struct,
    E=0.5,
    H=2,
    dh='auto',
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
    E : :obj:`float`, optional
        Extent weight. Default is 0.5.
    H : :obj:`float`, optional
        Height weight. Default is 2.
    dh : 'auto' or :obj:`float`, optional
        Step size for TFCE calculation.
        If set to 'auto', use 100 steps, as is done in fslmaths.
        A good alternative is 0.1 for z and t maps, as in [1]_.
        Default is 'auto'.
    two_sided_test : :obj:`bool`, optional
        Whether to assess both positive and negative clusters (True) or just
        positive ones (False).
        Default is False.

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

    for i_regressor in range(arr4d.shape[3]):
        arr3d = arr4d[..., i_regressor]
        if two_sided_test:
            signs = [-1, 1]
        else:
            arr3d[arr3d < 0] = 0
            signs = [1]

        # Get the maximum statistic in the map
        max_score = np.max(np.abs(arr3d))

        if dh == 'auto':
            step = max_score / 100
        else:
            step = dh

        for score_thresh in np.arange(step, max_score + step, step):
            for sign in signs:
                temp_arr3d = arr3d * sign

                # Threshold map at *h*
                temp_arr3d[temp_arr3d < score_thresh] = 0

                # Derive clusters
                labeled_arr3d, n_clusters = ndimage.measurements.label(
                    temp_arr3d,
                    bin_struct,
                )

                # Label each cluster with its extent
                # Each voxel's cluster extent at threshold *h* is thus *e(h)*
                cluster_map = np.zeros(temp_arr3d.shape, int)
                for cluster_val in range(1, n_clusters + 1):
                    bool_map = labeled_arr3d == cluster_val
                    cluster_map[bool_map] = np.sum(bool_map)

                # Calculate each voxel's tfce value based on its cluster extent
                # and z-value
                # NOTE: We do not multiply by dh, based on fslmaths'
                # implementation. This differs from the original paper.
                tfce_step_values = (cluster_map**E) * (score_thresh**H)
                tfce_4d[..., i_regressor] += sign * tfce_step_values

    return tfce_4d


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


def _calculate_cluster_measures(
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
        arr3d = arr4d[..., i_regressor].copy()

        if two_sided_test:
            arr3d[np.abs(arr3d) <= threshold] = 0
        else:
            arr3d[arr3d <= threshold] = 0

        labeled_arr3d, _ = ndimage.measurements.label(arr3d > 0, bin_struct)

        if two_sided_test:
            # Label positive and negative clusters separately
            n_positive_clusters = np.max(labeled_arr3d)
            temp_labeled_arr3d, _ = ndimage.measurements.label(
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
        if clust_sizes.size:
            max_size = np.max(clust_sizes)
        else:
            max_size = 0

        max_sizes[i_regressor], max_masses[i_regressor] = max_size, max_mass

    return max_sizes, max_masses
