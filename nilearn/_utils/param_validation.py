"""Utilities to check for valid parameters."""

import numbers
import warnings

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression

from nilearn._utils import logger

from .niimg import _get_data

# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1827243.0


def check_threshold(threshold, data, percentile_func, name="threshold"):
    """Check if the given threshold is in correct format \
    and within the limit.

    If necessary, this function also returns score of the data calculated based
    upon the given specific percentile function.
    Note: This is only for threshold as string.

    Parameters
    ----------
    threshold : float or str
        If threshold is a float value, it should be within the range of the
        maximum intensity value of the data.
        If threshold is a percentage expressed in a string it must finish with
        a percent sign like "99.7%".

    data : ndarray
        An array of the input masked data.

    percentile_func : function {scoreatpercentile, fastabspercentile}
        Percentile function for example scipy.stats.scoreatpercentile
        to calculate the score on the data.

    name : str, default='threshold'
        A string just used for representing
        the name of the threshold for a precise
        error message.

    Returns
    -------
    threshold : number
        Returns the score of the percentile on the data or
        returns threshold as it is
        if given threshold is not a string percentile.

    """
    if isinstance(threshold, str):
        message = (
            f'If "{name}" is given as string it '
            "should be a number followed by the percent "
            'sign, e.g. "25.3%"'
        )
        if not threshold.endswith("%"):
            raise ValueError(message)

        try:
            percentile = float(threshold[:-1])
        except ValueError as exc:
            exc.args += (message,)
            raise

        threshold = percentile_func(data, percentile)
    elif isinstance(threshold, numbers.Real):
        # checks whether given float value exceeds the maximum
        # value of the image data
        value_check = abs(data).max()
        if abs(threshold) > value_check:
            warnings.warn(
                f"The given float value must not exceed {value_check}. "
                f"But, you have given threshold={threshold}.",
                category=UserWarning,
                stacklevel=3,
            )
    else:
        raise TypeError(
            f"{name} should be either a number "
            "or a string finishing with a percent sign"
        )
    return threshold


def _get_mask_extent(mask_img):
    """Compute the extent of the provided brain mask.
    The extent is the volume of the mask in mm^3 if mask_img is a Nifti1Image
    or the number of vertices if mask_img is a SurfaceImage.

    Parameters
    ----------
    mask_img : Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    Returns
    -------
    mask_extent : float
        The computed volume in mm^3 (if mask_img is a Nifti1Image) or the
        number of vertices (if mask_img is a SurfaceImage).

    """
    if hasattr(mask_img, "affine"):
        affine = mask_img.affine
        prod_vox_dims = 1.0 * np.abs(np.linalg.det(affine[:3, :3]))
        return prod_vox_dims * _get_data(mask_img).astype(bool).sum()
    else:
        # sum number of True values in both hemispheres
        return (
            mask_img.data.parts["left"].sum()
            + mask_img.data.parts["right"].sum()
        )


def adjust_screening_percentile(
    screening_percentile,
    mask_img,
    verbose=0,
    mesh_n_vertices=None,
):
    """Adjust the screening percentile according to the MNI152 template or
    the number of vertices of the provided standard brain mesh.

    Parameters
    ----------
    screening_percentile : float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is expressed
        w.r.t the volume of either a standard (MNI152) brain (if mask_img is a
        3D volume) or a the number of vertices in the standard brain mesh
        (if mask_img is a SurfaceImage). This means that the
        `screening_percentile` is corrected at runtime by premultiplying it
        with the ratio of the volume of the mask of the data and volume of the
        standard brain.

    mask_img :  Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    verbose : int, default=0
        Verbosity level.

    mesh_n_vertices : int, default=None
        Number of vertices of the reference brain mesh, eg., fsaverage5
        or fsaverage7 etc.. If provided, the screening percentile will be
        adjusted according to the number of vertices.

    Returns
    -------
    screening_percentile : float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection.

    """
    original_screening_percentile = screening_percentile
    # correct screening_percentile according to the volume of the data mask
    # or the number of vertices of the reference mesh
    mask_extent = _get_mask_extent(mask_img)
    # if mask_img is a surface mesh, reference is the number of vertices
    # in the standard mesh otherwise it is the volume of the MNI152 brain
    # template
    reference_extent = (
        MNI152_BRAIN_VOLUME if mesh_n_vertices is None else mesh_n_vertices
    )
    if mask_extent > 1.1 * reference_extent:
        warnings.warn(
            "Brain mask is bigger than the standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=3,
        )
    elif mask_extent < 0.005 * reference_extent:
        warnings.warn(
            "Brain mask is smaller than .5% of the size of the standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=3,
        )

    if screening_percentile < 100.0:
        screening_percentile = screening_percentile * (
            reference_extent / mask_extent
        )
        screening_percentile = min(screening_percentile, 100.0)
    # if screening_percentile is 100, we don't do anything

    if hasattr(mask_img, "mesh"):
        log_mask = f"Mask n_vertices = {mask_extent:g}"
    else:
        log_mask = (
            f"Mask volume = {mask_extent:g}mm^3 = {mask_extent / 1000.0:g}cm^3"
        )
    logger.log(
        log_mask,
        verbose=verbose,
        msg_level=1,
    )
    if hasattr(mask_img, "mesh"):
        log_ref = f"Reference mesh n_vertices = {reference_extent:g}"
    else:
        log_ref = f"Standard brain volume = {MNI152_BRAIN_VOLUME:g}mm^3"
    logger.log(
        log_ref,
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Original screening-percentile: {original_screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Corrected screening-percentile: {screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    return screening_percentile


def check_feature_screening(
    screening_percentile,
    mask_img,
    is_classification,
    verbose=0,
    mesh_n_vertices=None,
):
    """Check feature screening method.

    Turns floats between 1 and 100 into SelectPercentile objects.

    Parameters
    ----------
    screening_percentile : float in the interval [0, 100]
        Percentile value for :term:`ANOVA` univariate feature selection.
        A value of 100 means 'keep all features'.
        This percentile is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.

    mask_img : nibabel image object
        Input image whose :term:`voxel` dimensions are to be computed.

    is_classification : bool
        If is_classification is True, it indicates that a classification task
        is performed. Otherwise, a regression task is performed.

    verbose : int, default=0
        Verbosity level.

    mesh_n_vertices : int, default=None
        Number of vertices of the reference mesh, eg., fsaverage5 or
        fsaverage7 etc.. If provided, the screening percentile will be adjusted
        according to the number of vertices.

    Returns
    -------
    selector : SelectPercentile instance
       Used to perform the :term:`ANOVA` univariate feature selection.

    """
    f_test = f_classif if is_classification else f_regression

    if screening_percentile == 100 or screening_percentile is None:
        return None
    elif not (0.0 <= screening_percentile <= 100.0):
        raise ValueError(
            "screening_percentile should be in the interval"
            f" [0, 100], got {screening_percentile:g}"
        )
    else:
        # correct screening_percentile according to the volume or the number of
        # vertices in the data mask
        screening_percentile_ = adjust_screening_percentile(
            screening_percentile,
            mask_img,
            verbose=verbose,
            mesh_n_vertices=mesh_n_vertices,
        )

        return SelectPercentile(f_test, percentile=int(screening_percentile_))


def check_run_sample_masks(n_runs, sample_masks):
    """Check that number of sample_mask matches number of runs."""
    if not isinstance(sample_masks, (list, tuple, np.ndarray)):
        raise TypeError(
            f"sample_mask has an unhandled type: {sample_masks.__class__}"
        )

    if isinstance(sample_masks, np.ndarray):
        sample_masks = (sample_masks,)

    checked_sample_masks = [_convert_bool2index(sm) for sm in sample_masks]

    if len(checked_sample_masks) != n_runs:
        raise ValueError(
            f"Number of sample_mask ({len(checked_sample_masks)}) not "
            f"matching number of runs ({n_runs})."
        )
    return checked_sample_masks


def _convert_bool2index(sample_mask):
    """Convert boolean to index."""
    check_boolean = [
        type(i) is bool or type(i) is np.bool_ for i in sample_mask
    ]
    if all(check_boolean):
        sample_mask = np.where(sample_mask)[0]
    return sample_mask
