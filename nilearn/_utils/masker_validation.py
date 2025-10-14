"""Masker utilities available for other nilearn modules.

Must be kept out of the nilearn.maskers subpackage to avoid circular imports.
"""

from collections.abc import Iterable

from nilearn._utils.param_validation import check_is_of_allowed_type
from nilearn.surface.surface import SurfaceImage
from nilearn.typing import NiimgLike


def check_compatibility_mask_and_images(mask_img, run_imgs):
    """Check that mask type and image types are compatible.

    Images to fit should be a Niimg-Like
    if the mask is a NiftiImage, NiftiMasker or a path.
    Similarly, only SurfaceImages can be fitted
    with a SurfaceImage or a SurfaceMasker as mask.
    """
    from nilearn.maskers import NiftiMasker, SurfaceMasker

    if mask_img is None:
        return None

    if not isinstance(run_imgs, Iterable):
        run_imgs = [run_imgs]

    msg = (
        "Mask and input images must be of compatible types.\n"
        f"Got mask of type: {mask_img.__class__.__name__}, "
        f"and images of type: {[type(x) for x in run_imgs]}"
    )

    volumetric_type = (*NiimgLike, NiftiMasker)
    surface_type = (SurfaceImage, SurfaceMasker)
    all_allowed_types = (*volumetric_type, *surface_type)

    check_is_of_allowed_type(mask_img, all_allowed_types, "mask")

    if isinstance(mask_img, volumetric_type) and any(
        not isinstance(x, NiimgLike) for x in run_imgs
    ):
        raise TypeError(
            f"{msg} "
            f"where images should be NiftiImage-like instances "
            f"(Nifti1Image or str or Path)."
        )
    elif isinstance(mask_img, surface_type) and any(
        not isinstance(x, SurfaceImage) for x in run_imgs
    ):
        raise TypeError(
            f"{msg} where SurfaceImage instances would be expected."
        )
