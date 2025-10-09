"""Utilities to check for decoders."""

import warnings

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression

from nilearn._utils import logger
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import _get_data
from nilearn.exceptions import MaskWarning
from nilearn.surface import SurfaceImage

# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1882989.0


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
    if not hasattr(mask_img, "affine"):
        # sum number of True values in both hemispheres
        return (
            mask_img.data.parts["left"].sum()
            + mask_img.data.parts["right"].sum()
        )
    affine = mask_img.affine
    prod_vox_dims = 1.0 * np.abs(np.linalg.det(affine[:3, :3]))
    return prod_vox_dims * _get_data(mask_img).astype(bool).sum()


@fill_doc
def adjust_screening_percentile(screening_percentile, mask_img, verbose=0):
    """Adjust the screening percentile according to the MNI152 template or
    the number of vertices of the provided standard brain mesh.

    Parameters
    ----------
    %(screening_percentile)s

    mask_img :  Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    %(verbose0)s

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
        mask_img.mesh.n_vertices
        if isinstance(mask_img, SurfaceImage)
        else MNI152_BRAIN_VOLUME
    )
    if mask_extent > 1.1 * reference_extent:
        unit = "mm^3"
        if hasattr(mask_img, "mesh"):
            unit = "vertices"
        warnings.warn(
            f"Brain mask ({mask_extent} {unit}) is bigger than the standard "
            f"human brain ({reference_extent} {unit})."
            "This object is probably not tuned to be used on such data.",
            stacklevel=find_stack_level(),
            category=MaskWarning,
        )
    elif mask_extent < 0.005 * reference_extent:
        warnings.warn(
            "Brain mask is smaller than .5% of the size of the standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=find_stack_level(),
            category=MaskWarning,
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


@fill_doc
def check_feature_screening(
    screening_percentile, mask_img, is_classification, verbose=0
):
    """Check feature screening method.

    Turns floats between 1 and 100 into SelectPercentile objects.

    Parameters
    ----------
    %(screening_percentile)s

    mask_img : nibabel image object
        Input image whose :term:`voxel` dimensions are to be computed.

    is_classification : bool
        If is_classification is True, it indicates that a classification task
        is performed. Otherwise, a regression task is performed.

    %(verbose0)s

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
        effective_screening_percentile = adjust_screening_percentile(
            screening_percentile,
            mask_img,
            verbose=verbose,
        )

        if effective_screening_percentile == 100:
            warnings.warn(
                f"screening_percentile set to '100' despite "
                f"requesting '{screening_percentile=}'. "
                "\nAll elements in the mask will be included. "
                "\nThis usually occurs when the mask image "
                "is too small compared to full brain mask.",
                category=UserWarning,
                stacklevel=find_stack_level(),
            )

        return SelectPercentile(
            f_test, percentile=int(effective_screening_percentile)
        )
