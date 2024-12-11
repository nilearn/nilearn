"""Utilities for probabilistic error control at voxel- and \
cluster-level in brain imaging: cluster-level thresholding, false \
discovery rate control, false discovery proportion in clusters.

Author: Bertrand Thirion, 2015 -- 2019

"""

import warnings

import numpy as np
from scipy.ndimage import label
from scipy.stats import norm

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.image import get_data, math_img, threshold_img
from nilearn.maskers import NiftiMasker


def _compute_hommel_value(z_vals, alpha, verbose=0):
    """Compute the All-Resolution Inference hommel-value."""
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha should be between 0 and 1")
    z_vals_ = -np.sort(-z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_samples
    if p_vals[-1] < alpha:
        return 0
    slopes = (alpha - p_vals[:-1]) / np.arange(n_samples - 1, 0, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(alpha / slope)
    if verbose > 0:
        if not is_matplotlib_installed():
            warnings.warn(
                '"verbose" option requires the package Matplotlib.'
                "Please install it using `pip install matplotlib`."
            )
        else:
            from matplotlib import pyplot as plt

            plt.figure()
            plt.plot(np.arange(1, 1 + n_samples), p_vals, "o")
            plt.plot([n_samples - hommel_value, n_samples], [0, alpha])
            plt.plot([0, n_samples], [0, 0], "k")
            plt.show(block=False)
    return int(np.minimum(hommel_value, n_samples))


def _true_positive_fraction(z_vals, hommel_value, alpha):
    """Given a bunch of z-avalues, return the true positive fraction.

    Parameters
    ----------
    z_vals : array,
        A set of z-variates from which the FDR is computed.

    hommel_value : :obj:`int`
        The Hommel value, used in the computations.

    alpha : :obj:`float`
        The desired FDR control.

    Returns
    -------
    threshold : :obj:`float`
        Estimated true positive fraction in the set of values.

    """
    z_vals_ = -np.sort(-z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    c = np.ceil((hommel_value * p_vals) / alpha)
    unique_c, counts = np.unique(c, return_counts=True)
    criterion = 1 - unique_c + np.cumsum(counts)
    proportion_true_discoveries = np.maximum(0, criterion.max() / n_samples)
    return proportion_true_discoveries


def fdr_threshold(z_vals, alpha):
    """Return the Benjamini-Hochberg FDR threshold for the input z_vals.

    Parameters
    ----------
    z_vals : array
        A set of z-variates from which the FDR is computed.

    alpha : :obj:`float`
        The desired FDR control.

    Returns
    -------
    threshold : :obj:`float`
        FDR-controling threshold from the Benjamini-Hochberg procedure.

    """
    if alpha < 0 or alpha > 1:
        raise ValueError(
            f"alpha should be between 0 and 1. {alpha} was provided"
        )
    z_vals_ = -np.sort(-z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    pos = p_vals < alpha * np.linspace(1 / n_samples, 1, n_samples)
    return z_vals_[pos][-1] - 1.0e-12 if pos.any() else np.inf


def cluster_level_inference(
    stat_img, mask_img=None, threshold=3.0, alpha=0.05, verbose=0
):
    """Report the proportion of active voxels for all clusters \
    defined by the input threshold.

    This implements the method described in :footcite:t:`Rosenblatt2018`.

    Parameters
    ----------
    stat_img : Niimg-like object
       statistical image (presumably in z scale)

    mask_img : Niimg-like object, default=None
        mask image

    threshold : :obj:`list` of :obj:`float`, default=3.0
       Cluster-forming threshold in z-scale.

    alpha : :obj:`float` or :obj:`list`, default=0.05
        Level of control on the true positive rate, aka true discovery
        proportion.

    verbose : :obj:`int` or :obj:`bool`, default=0
        Verbosity mode.

    Returns
    -------
    proportion_true_discoveries_img : Nifti1Image
        The statistical map that gives the true positive.

    References
    ----------
    .. footbibliography::

    """
    if verbose is False:
        verbose = 0
    if verbose is True:
        verbose = 1

    if not isinstance(threshold, list):
        threshold = [threshold]

    if mask_img is None:
        masker = NiftiMasker(mask_strategy="background").fit(stat_img)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    stats = np.ravel(masker.transform(stat_img))
    hommel_value = _compute_hommel_value(stats, alpha, verbose=verbose)

    # embed it back to 3D grid
    stat_map = get_data(masker.inverse_transform(stats))

    # Extract connected components above threshold
    proportion_true_discoveries_img = math_img("0. * img", img=stat_img)
    proportion_true_discoveries = masker.transform(
        proportion_true_discoveries_img
    ).ravel()

    for threshold_ in sorted(threshold):
        label_map, n_labels = label(stat_map > threshold_)
        labels = label_map[get_data(masker.mask_img_) > 0]

        for label_ in range(1, n_labels + 1):
            # get the z-vals in the cluster
            cluster_vals = stats[labels == label_]
            proportion = _true_positive_fraction(
                cluster_vals, hommel_value, alpha
            )
            proportion_true_discoveries[labels == label_] = proportion

    proportion_true_discoveries_img = masker.inverse_transform(
        proportion_true_discoveries
    )
    return proportion_true_discoveries_img


def threshold_stats_img(
    stat_img=None,
    mask_img=None,
    alpha=0.001,
    threshold=3.0,
    height_control="fpr",
    cluster_threshold=0,
    two_sided=True,
):
    """Compute the required threshold level and return the thresholded map.

    Parameters
    ----------
    stat_img : Niimg-like object or None, default=None
       Statistical image (presumably in z scale) whenever height_control
       is 'fpr' or None, stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni', an error is raised if stat_img is None.

    mask_img : Niimg-like object, default=None
        Mask image

    alpha : :obj:`float` or :obj:`list`, default=0.001
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    threshold : :obj:`float`, default=3.0
       Desired threshold in z-scale.
       This is used only if height_control is None.

    height_control : :obj:`str`, or None optional, default='fpr'
        False positive control meaning of cluster forming
        threshold: None|'fpr'|'fdr'|'bonferroni'

    cluster_threshold : :obj:`float`, default=0
        cluster size threshold. In the returned thresholded map,
        sets of connected voxels (`clusters`) with size smaller
        than this number will be removed.

    two_sided : :obj:`bool`, default=True
        Whether the thresholding should yield both positive and negative
        part of the maps.
        In that case, alpha is corrected by a factor of 2.

    Returns
    -------
    thresholded_map : Nifti1Image,
        The stat_map thresholded at the prescribed voxel- and cluster-level.

    threshold : :obj:`float`
        The voxel-level threshold used actually.

    Notes
    -----
    If the input image is not z-scaled (i.e. some z-transformed statistic)
    the computed threshold is not rigorous and likely meaningless

    See Also
    --------
    nilearn.image.threshold_img :
        Apply an explicit voxel-level (and optionally cluster-level) threshold
        without correction.

    """
    height_control_methods = [
        "fpr",
        "fdr",
        "bonferroni",
        "all-resolution-inference",
        None,
    ]
    if height_control not in height_control_methods:
        raise ValueError(
            f"height control should be one of {height_control_methods}"
        )

    # if two-sided, correct alpha by a factor of 2
    alpha_ = alpha / 2 if two_sided else alpha

    # if height_control is 'fpr' or None, we don't need to look at the data
    # to compute the threshold
    if height_control == "fpr":
        threshold = norm.isf(alpha_)

    # In this case, and if stat_img is None, we return
    if stat_img is None:
        if height_control in ["fpr", None]:
            return None, threshold
        else:
            raise ValueError(
                "Map_threshold requires stat_img not to be None "
                "when the height_control procedure "
                'is "bonferroni" or "fdr"'
            )

    if mask_img is None:
        masker = NiftiMasker(mask_strategy="background").fit(stat_img)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    stats = np.ravel(masker.transform(stat_img))
    n_voxels = np.size(stats)

    # Thresholding
    if two_sided:
        # replace stats by their absolute value
        stats = np.abs(stats)

    if height_control == "fdr":
        threshold = fdr_threshold(stats, alpha_)
    elif height_control == "bonferroni":
        threshold = norm.isf(alpha_ / n_voxels)

    # Apply cluster-extent thresholding with new cluster-defining threshold
    stat_img = threshold_img(
        img=stat_img,
        threshold=threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=mask_img,
        copy=True,
        copy_header=True,
    )

    return stat_img, threshold
