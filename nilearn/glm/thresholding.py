"""Utilities for probabilistic error control at voxel- and
cluster-level in brain imaging: cluster-level thresholding, false
discovery rate control, false discovery proportion in clusters.

Author: Bertrand Thirion, 2015 -- 2019

"""
import warnings

import numpy as np
from scipy.ndimage import label
from scipy.stats import norm

from nilearn.input_data import NiftiMasker
from nilearn.image import get_data, math_img


def _compute_hommel_value(z_vals, alpha, verbose=False):
    """Compute the All-Resolution Inference hommel-value"""
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_samples
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_samples, 1, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(n_samples + (alpha - slope * n_samples) / slope)
    if verbose:
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            warnings.warn('"verbose" option requires the package Matplotlib.'
                          'Please install it using `pip install matplotlib`.')
        else:
            plt.figure()
            plt.plot(p_vals, 'o')
            plt.plot([n_samples - hommel_value, n_samples], [0, alpha])
            plt.plot([0, n_samples], [0, 0], 'k')
            plt.show(block=False)
    return np.minimum(hommel_value, n_samples)


def _true_positive_fraction(z_vals, hommel_value, alpha):
    """Given a bunch of z-avalues, return the true positive fraction

    Parameters
    ----------
    z_vals : array,
        A set of z-variates from which the FDR is computed.

    hommel_value: int
        The Hommel value, used in the computations.

    alpha : float
        The desired FDR control.

    Returns
    -------
    threshold : float
        Estimated true positive fraction in the set of values.

    """
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    c = np.ceil((hommel_value * p_vals) / alpha)
    unique_c, counts = np.unique(c, return_counts=True)
    criterion = 1 - unique_c + np.cumsum(counts)
    proportion_true_discoveries = np.maximum(0, criterion.max() / n_samples)
    return proportion_true_discoveries


def fdr_threshold(z_vals, alpha):
    """Return the Benjamini-Hochberg FDR threshold for the input z_vals

    Parameters
    ----------
    z_vals : array
        A set of z-variates from which the FDR is computed.

    alpha : float
        The desired FDR control.

    Returns
    -------
    threshold : float
        FDR-controling threshold from the Benjamini-Hochberg procedure.

    """
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'alpha should be between 0 and 1. {} was provided'.format(alpha))
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    pos = p_vals < alpha * np.linspace(
        .5 / n_samples, 1 - .5 / n_samples, n_samples)
    if pos.any():
        return (z_vals_[pos][-1] - 1.e-12)

    return np.infty


def cluster_level_inference(stat_img, mask_img=None,
                            threshold=3., alpha=.05, verbose=False):
    """ Report the proportion of active voxels for all clusters
    defined by the input threshold.

    This implements the method described in [1]_.

    Parameters
    ----------
    stat_img : Niimg-like object or None, optional
       statistical image (presumably in z scale)

    mask_img : Niimg-like object, optional,
        mask image

    threshold : list of floats, optional
       Cluster-forming threshold in z-scale. Default=3.0.

    alpha : float or list, optional
        Level of control on the true positive rate, aka true dsicovery
        proportion. Default=0.05.

    verbose : bool, optional
        Verbosity mode. Default=False.

    Returns
    -------
    proportion_true_discoveries_img : Nifti1Image
        The statistical map that gives the true positive.

    Notes
    -----
    This function is experimental.
    It may change in any future release of Nilearn.

    References
    ----------
    .. [1] Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ. All-Resolutions
       Inference for brain imaging. Neuroimage. 2018 Nov 1;181:786-796. doi:
       10.1016/j.neuroimage.2018.07.060

    """

    if not isinstance(threshold, list):
        threshold = [threshold]

    if mask_img is None:
        masker = NiftiMasker(mask_strategy='background').fit(stat_img)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    stats = np.ravel(masker.transform(stat_img))
    hommel_value = _compute_hommel_value(stats, alpha, verbose=verbose)

    # embed it back to 3D grid
    stat_map = get_data(masker.inverse_transform(stats))

    # Extract connected components above threshold
    proportion_true_discoveries_img = math_img('0. * img', img=stat_img)
    proportion_true_discoveries = masker.transform(
        proportion_true_discoveries_img).ravel()

    for threshold_ in sorted(threshold):
        label_map, n_labels = label(stat_map > threshold_)
        labels = label_map[get_data(masker.mask_img_) > 0]
        for label_ in range(1, n_labels + 1):
            # get the z-vals in the cluster
            cluster_vals = stats[labels == label_]
            proportion = _true_positive_fraction(cluster_vals, hommel_value,
                                                 alpha)
            proportion_true_discoveries[labels == label_] = proportion

    proportion_true_discoveries_img = masker.inverse_transform(
        proportion_true_discoveries)
    return proportion_true_discoveries_img


def threshold_stats_img(stat_img=None, mask_img=None, alpha=.001, threshold=3.,
                        height_control='fpr', cluster_threshold=0,
                        two_sided=True):
    """ Compute the required threshold level and return the thresholded map

    Parameters
    ----------
    stat_img : Niimg-like object or None, optional
       Statistical image (presumably in z scale) whenever height_control
       is 'fpr' or None, stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni', an error is raised if stat_img is None.

    mask_img : Niimg-like object, optional,
        Mask image

    alpha : float or list, optional
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.
        Default=0.001.

    threshold : float, optional
       Desired threshold in z-scale.
       This is used only if height_control is None. Default=3.0.

    height_control : string, or None optional
        False positive control meaning of cluster forming
        threshold: None|'fpr'|'fdr'|'bonferroni'
        Default='fpr'.

    cluster_threshold : float, optional
        cluster size threshold. In the returned thresholded map,
        sets of connected voxels (`clusters`) with size smaller
        than this number will be removed. Default=0.

    two_sided : Bool, optional
        Whether the thresholding should yield both positive and negative
        part of the maps.
        In that case, alpha is corrected by a factor of 2.
        Default=True.

    Returns
    -------
    thresholded_map : Nifti1Image,
        The stat_map thresholded at the prescribed voxel- and cluster-level.

    threshold : float
        The voxel-level threshold used actually.

    Notes
    -----
    If the input image is not z-scaled (i.e. some z-transformed statistic)
    the computed threshold is not rigorous and likely meaningless

    This function is experimental.
    It may change in any future release of Nilearn.

    See also
    --------
    nilearn.image.threshold_img

    """
    height_control_methods = ['fpr', 'fdr', 'bonferroni',
                              'all-resolution-inference', None]
    if height_control not in height_control_methods:
        raise ValueError(
            "height control should be one of {0}", height_control_methods)

    # if two-sided, correct alpha by a factor of 2
    alpha_ = alpha / 2 if two_sided else alpha

    # if height_control is 'fpr' or None, we don't need to look at the data
    # to compute the threshold
    if height_control == 'fpr':
        threshold = norm.isf(alpha_)

    # In this case, and if stat_img is None, we return
    if stat_img is None:
        if height_control in ['fpr', None]:
            return None, threshold
        else:
            raise ValueError('Map_threshold requires stat_img not to be None'
                             'when the height_control procedure '
                             'is "bonferroni" or "fdr"')

    if mask_img is None:
        masker = NiftiMasker(mask_strategy='background').fit(stat_img)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    stats = np.ravel(masker.transform(stat_img))
    n_voxels = np.size(stats)

    # Thresholding
    if two_sided:
        # replace stats by their absolute value after storing the sign
        sign = np.sign(stats)
        stats = np.abs(stats)

    if height_control == 'fdr':
        threshold = fdr_threshold(stats, alpha_)
    elif height_control == 'bonferroni':
        threshold = norm.isf(alpha_ / n_voxels)
    stats *= (stats > threshold)
    if two_sided:
        stats *= sign

    # embed it back to 3D grid
    stat_map = get_data(masker.inverse_transform(stats))

    # Extract connected components above threshold
    label_map, n_labels = label(np.abs(stat_map) > threshold)
    labels = label_map[get_data(masker.mask_img_) > 0]

    for label_ in range(1, n_labels + 1):
        if np.sum(labels == label_) < cluster_threshold:
            stats[labels == label_] = 0

    return masker.inverse_transform(stats), threshold
