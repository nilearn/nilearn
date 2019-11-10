""" Utilities to describe the result of cluster-level analysis of statistical
maps.

Author: Bertrand Thirion, 2015
"""
import numpy as np

from nilearn.input_data import NiftiMasker
from scipy.ndimage import label
from scipy.stats import norm
from nistats.utils import get_data


def fdr_threshold(z_vals, alpha):
    """ return the Benjamini-Hochberg FDR threshold for the input z_vals
    
    Parameters
    ----------
    z_vals: array,
            a set of z-variates from which the FDR is computed
    alpha: float,
           desired FDR control
    
    Returns
    -------
    threshold: float,
               FDR-controling threshold from the Benjamini-Hochberg procedure
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    pos = p_vals < alpha * np.linspace(
        .5 / n_samples, 1 - .5 / n_samples, n_samples)
    if pos.any():
        return (z_vals_[pos][-1] - 1.e-12)
    else:
        return np.infty


def map_threshold(stat_img=None, mask_img=None, alpha=.001, threshold=3.,
                  height_control='fpr', cluster_threshold=0):
    """ Compute the required threshold level and return the thresholded map

    Parameters
    ----------
    stat_img : Niimg-like object or None, optional
       statistical image (presumably in z scale)
       whenever height_control is 'fpr' or None,
       stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni', an error is raised if stat_img is None.

    mask_img : Niimg-like object, optional,
        mask image

    alpha: float, optional
        number controling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    threshold: float, optional
       desired threshold in z-scale.
       This is used only if height_control is None

    height_control: string, or None optional
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'\|None

    cluster_threshold : float, optional
        cluster size threshold. In the returned thresholded map,
        sets of connected voxels (`clusters`) with size smaller
        than this number will be removed.

    Returns
    -------
    thresholded_map : Nifti1Image,
        the stat_map thresholded at the prescribed voxel- and cluster-level

    threshold: float,
        the voxel-level threshold used actually

    Note
    ----
    If the input image is not z-scaled (i.e. some z-transformed statistic)
    the computed threshold is not rigorous and likely meaningless
    """
    # Check that height_control is correctly specified
    if height_control not in ['fpr', 'fdr', 'bonferroni', None]:
        raise ValueError(
            "height control should be one of ['fpr', 'fdr', 'bonferroni', None]")

    # if height_control is 'fpr' or None, we don't need to look at the data
    # to compute the threhsold
    if height_control == 'fpr':
        threshold = norm.isf(alpha)

    # In this case, and is stat_img is None, we return
    if stat_img is None:
        if height_control in ['fpr', None]:
            return None, threshold
        else:
            raise ValueError(
                'Map_threshold requires stat_img not to be None'
                'when the heigh_control procedure is bonferroni or fdr')
    
    # Masking
    if mask_img is None:
        masker = NiftiMasker(mask_strategy='background').fit(stat_img)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    stats = np.ravel(masker.transform(stat_img))
    n_voxels = np.size(stats)

    # Thresholding
    if height_control == 'fdr':
        threshold = fdr_threshold(stats, alpha)
    elif height_control == 'bonferroni':
        threshold = norm.isf(alpha / n_voxels)
    stats *= (stats > threshold)

    # embed it back to 3D grid
    stat_map = get_data(masker.inverse_transform(stats))

    # Extract connected components above threshold
    label_map, n_labels = label(stat_map > threshold)
    labels = label_map[get_data(masker.mask_img_) > 0]

    for label_ in range(1, n_labels + 1):
        if np.sum(labels == label_) < cluster_threshold:
            stats[labels == label_] = 0

    return masker.inverse_transform(stats), threshold
