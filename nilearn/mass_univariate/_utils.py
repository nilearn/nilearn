"""Utility functions for the permuted least squares method."""
import nibabel as nib
import numpy as np
from scipy import ndimage
from nilearn.masking import apply_mask, unmask


def _calculate_tfce(scores_array, masker, E=0.5, H=2, dh=0.1):
    """Calculate threshold-free cluster enhancement values for scores maps.

    The :term:`TFCE` calculation is implemented as described in [1]_.

    Parameters
    ----------
    scores_array : :obj:`numpy.ndarray`, shape=(n_regressors, n_descriptors)
        Scores (t-statistics) for a set of regressors.
    masker
    E : :obj:`float`, optional
        Extent weight. Default is 0.5.
    H : :obj:`float`, optional
        Height weight. Default is 2.
    dh : :obj:`float`, optional
        Step size for TFCE calculation. Default is 0.1.

    Returns
    -------
    tfce_arr : :obj:`numpy.ndarray`, shape=(n_regressors, n_descriptors)
        :term:`TFCE` values.

    References
    ----------
    .. [1] Smith, S. M., & Nichols, T. E. (2009).
       Threshold-free cluster enhancement: addressing problems of smoothing,
       threshold dependence and localisation in cluster inference.
       Neuroimage, 44(1), 83-98.
    """
    # Define connectivity matrix for cluster labeling
    conn = ndimage.generate_binary_structure(3, 1)

    scores_4d_img = unmask(scores_array.T, masker.mask_img_)
    scores_4d = scores_4d_img.get_fdata()
    tfce_4d = np.zeros_like(scores_4d)

    for i_regressor in range(scores_4d.shape[3]):
        scores_3d = scores_4d[..., i_regressor]

        # Get the step right before the maximum z-statistic in the map
        max_z = np.floor(np.max(scores_3d) / dh) * dh

        for score_thresh in np.arange(dh, max_z + dh, dh):
            # Threshold map at *h*
            scores_3d[scores_3d < score_thresh] = 0

            # Derive clusters
            labeled_arr3d, n_clusters = ndimage.measurements.label(
                scores_3d,
                conn,
            )

            # Label each cluster with its extent
            # Each voxel's cluster extent at threshold *h* is thus *e(h)*
            cluster_map = np.zeros(scores_3d.shape, int)
            for cluster_val in range(1, n_clusters + 1):
                bool_map = labeled_arr3d == cluster_val
                cluster_map[bool_map] = np.sum(bool_map)

            # Calculate each voxel's tfce value based on its cluster extent
            # and z-value
            tfce_step_values = (cluster_map**E) * (score_thresh**H)
            tfce_4d[..., i_regressor] += tfce_step_values

    tfce_arr = apply_mask(
        nib.Nifti1Image(
            tfce_4d,
            masker.mask_img_.affine,
            masker.mask_img_.header,
        ),
        masker.mask_img_,
    )

    return tfce_arr.T
