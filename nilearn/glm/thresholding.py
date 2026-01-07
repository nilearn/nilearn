"""Utilities for probabilistic error control at voxel- and \
cluster-level in brain imaging: cluster-level thresholding, false \
discovery rate control, false discovery proportion in clusters.
"""

import warnings

import numpy as np
from scipy.ndimage import label
from scipy.stats import norm

from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import (
    check_parameter_in_allowed,
    check_params,
)
from nilearn.image import (
    check_niimg_3d,
    get_data,
    math_img,
    new_img_like,
    threshold_img,
)
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.surface.surface import SurfaceImage, check_surf_img
from nilearn.typing import ClusterThreshold, HeightControl

DEFAULT_Z_THRESHOLD = norm.isf(0.001)


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
                "Please install it using `pip install matplotlib`.",
                stacklevel=find_stack_level(),
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


@fill_doc
def cluster_level_inference(
    stat_img,
    mask_img=None,
    threshold: float | int | list[float | int] = 3.0,
    alpha=0.05,
    verbose: int = 0,
):
    """Report the proportion of active voxels for all clusters \
    defined by the input threshold.

    This implements the method described in :footcite:t:`Rosenblatt2018`.

    Parameters
    ----------
    stat_img : 3D Niimg-like object or \
        :obj:`~nilearn.surface.SurfaceImage` with a single sample.
       statistical image (presumably in z scale)

    mask_img : Niimg-like object, or :obj:`~nilearn.surface.SurfaceImage` \
        or None, default=None
        mask image

    threshold : Non-negative :obj:`float`, :obj:`int`, \
                 or :obj:`list` of \
                 non-negative :obj:`float` or :obj:`int`, default=3.0
       Cluster-forming threshold in z-scale.

    alpha : :obj:`float` or :obj:`list`, default=0.05
        Level of control on the true positive rate, aka true discovery
        proportion.

    %(verbose0)s

    Returns
    -------
    proportion_true_discoveries_img : Nifti1Image \
          or :obj:`~nilearn.surface.SurfaceImage`
        The statistical map that gives the true positive.

    References
    ----------
    .. footbibliography::

    """
    # TODO (nilearn >= 0.15.0) remove
    if threshold == 3.0:
        warnings.warn(
            "\nFrom nilearn version>=0.15, "
            "the default 'threshold' will be set to "
            f"{DEFAULT_Z_THRESHOLD}.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    original_threshold = threshold
    if not isinstance(threshold, list):
        threshold = [threshold]
    if any(x < 0 for x in threshold):
        raise ValueError(
            "'threshold' cannot be negative or "
            "contain negative values. "
            f"Got: 'threshold={original_threshold}'."
        )

    if isinstance(stat_img, SurfaceImage) or isinstance(
        mask_img, SurfaceImage
    ):
        return _cluster_level_inference_surface(
            stat_img, mask_img, threshold, alpha, verbose
        )

    return _cluster_level_inference_volume(
        stat_img, mask_img, threshold, alpha, verbose
    )


def _cluster_level_inference_surface(
    stat_img, mask_img, threshold, alpha, verbose
):
    """Run the inference on each hemisphere indendently
    by creating a temporary mask that only includes one hemisphere.
    """
    check_surf_img(stat_img)
    stat_img.data._check_n_samples(1)

    if mask_img is None:
        masker = SurfaceMasker().fit(stat_img)
        mask_img = masker.mask_img_
        del masker

    data = {
        "left": np.zeros(stat_img.data.parts["left"].shape),
        "right": np.zeros(stat_img.data.parts["right"].shape),
    }
    for hemi in ["left", "right"]:
        if hemi == "left":
            mask_left = mask_img.data.parts["left"].astype(bool)
            hemi_empty = not np.any(mask_left.ravel())
            mask_right = np.zeros(
                mask_img.data.parts["right"].shape, dtype=bool
            )
        else:
            mask_left = np.zeros(mask_img.data.parts["left"].shape, dtype=bool)
            mask_right = mask_img.data.parts["right"].astype(bool)
            hemi_empty = not np.any(mask_right.ravel())

        if hemi_empty:
            continue

        tmp_mask = new_img_like(
            stat_img, {"left": mask_left, "right": mask_right}
        )
        masker = SurfaceMasker(mask_img=tmp_mask).fit()

        stats = np.ravel(masker.transform(stat_img))
        hommel_value = _compute_hommel_value(stats, alpha, verbose=verbose)

        # embed it back to image
        stat_map = masker.inverse_transform(stats).data.parts[hemi]

        # Extract connected components above threshold
        proportion_true_discoveries_img = math_img("0. * img", img=stat_img)
        proportion_true_discoveries = masker.transform(
            proportion_true_discoveries_img
        ).ravel()

        for threshold_ in sorted(threshold):
            label_map, n_labels = label(stat_map > threshold_)
            labels = label_map[masker.mask_img_.data.parts[hemi] > 0]

            for label_ in range(1, n_labels + 1):
                # get the z-vals in the cluster
                cluster_vals = stats[labels == label_]
                proportion = _true_positive_fraction(
                    cluster_vals, hommel_value, alpha
                )
                proportion_true_discoveries[labels == label_] = proportion

        tmp_img = masker.inverse_transform(proportion_true_discoveries)
        data[hemi] = tmp_img.data.parts[hemi]

    return new_img_like(stat_img, data)


def _cluster_level_inference_volume(
    stat_img, mask_img, threshold, alpha, verbose
):
    stat_img = check_niimg_3d(stat_img)
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

    return masker.inverse_transform(proportion_true_discoveries)


@fill_doc
def threshold_stats_img(
    stat_img=None,
    mask_img=None,
    alpha=0.001,
    threshold: float | int | np.floating | np.integer | None = None,
    height_control: HeightControl = "fpr",
    cluster_threshold: ClusterThreshold = 0,
    two_sided: bool = True,
):
    """Compute the required threshold level and return the thresholded map.

    Parameters
    ----------
    stat_img : Niimg-like object, or a :obj:`~nilearn.surface.SurfaceImage` \
               or None, default=None
       Statistical image (presumably in z scale) whenever height_control
       is 'fpr' or None, stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni', an error is raised if stat_img is None.

    mask_img : Niimg-like object, default=None
        Mask image

    alpha : :obj:`float` or :obj:`list`, default=0.001
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    threshold : :obj:`float` or :obj:`int` or None, default=None
       Desired threshold in z-scale.
       This is used only if ``height_control`` is None.
       If ``threshold`` is set to None when ``height_control`` is None,
       ``threshold`` will be set to 3.0.

       .. note::

            - When ``two_sided`` is True:

              ``'threshold'`` cannot be negative.

              The given value should be within the range of minimum and maximum
              intensity of the input image.
              All intensities in the interval ``[-threshold, threshold]``
              will be set to zero.

            - When ``two_sided`` is False:

              - If the threshold is negative:

                It should be greater than the minimum intensity
                of the input data.
                All intensities greater than or equal
                to the specified threshold will be set to zero.
                All other intensities keep their original values.

              - If the threshold is positive:

                It should be less than the maximum intensity
                of the input data.
                All intensities less than or equal
                to the specified threshold will be set to zero.
                All other intensities keep their original values.

    height_control : :obj:`str`, or None, default='fpr'
        False positive control meaning of cluster forming
        threshold: None|'fpr'|'fdr'|'bonferroni'

    %(cluster_threshold)s

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
    if height_control is None:
        if threshold is None:
            threshold = 3.0

        # TODO (nilearn >= 0.15.0) remove
        if threshold == 3.0:
            warnings.warn(
                "\nFrom nilearn version>=0.15, "
                "the default 'threshold' will be set to "
                f"{DEFAULT_Z_THRESHOLD}.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    elif threshold is not None:
        threshold = float(threshold)
        warnings.warn(
            f"\n'{threshold=}' is not used with '{height_control=}'."
            "\n'threshold' is only used when 'height_control=None'. "
            "\n'threshold' was set to 'None'. ",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        threshold = None

    height_control_methods = [
        "fpr",
        "fdr",
        "bonferroni",
        None,
    ]
    check_parameter_in_allowed(
        height_control, height_control_methods, "height_control"
    )
    check_params(locals())

    if cluster_threshold < 0:
        raise ValueError(
            f"'cluster_threshold' must be > 0. Got {cluster_threshold=}"
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
                f"'stat_img' cannot be None for {height_control=}"
            )

    if mask_img is None:
        if isinstance(stat_img, SurfaceImage):
            masker = SurfaceMasker()
        else:
            masker = NiftiMasker(mask_strategy="background")
        masker.fit(stat_img)
    else:
        if isinstance(stat_img, SurfaceImage):
            masker = SurfaceMasker(mask_img=mask_img)
        else:
            masker = NiftiMasker(mask_img=mask_img)
        masker.fit()

    stats = np.ravel(masker.transform(stat_img))
    n_elements = np.size(stats)

    # Thresholding
    if two_sided:
        # replace stats by their absolute value
        stats = np.abs(stats)

    if height_control == "fdr":
        threshold = fdr_threshold(stats, alpha_)
    elif height_control == "bonferroni":
        threshold = norm.isf(alpha_ / n_elements)

    # Apply cluster-extent thresholding with new cluster-defining threshold
    stat_img = threshold_img(
        img=stat_img,
        threshold=threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=mask_img,
        copy=True,
    )

    return stat_img, threshold
