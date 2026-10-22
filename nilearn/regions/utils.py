"""Utility functions from regions modules."""

import warnings

import numpy as np
from nibabel import Nifti1Image

from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import safe_get_data
from nilearn._utils.numpy_conversions import as_ndarray


def _non_empty_maps(maps_data: np.ndarray) -> set[int]:
    """Return the indices of the maps that contain at least one non-zero."""
    return {int(i) for i in np.flatnonzero(np.any(maps_data, axis=(0, 1, 2)))}


def apply_mask_to_maps_img(
    mask_img: Nifti1Image, maps_img: Nifti1Image
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Apply a mask image to maps image.

    Checks that any map is left and warns if some were dropped by masking.
    """
    maps_data = safe_get_data(maps_img, ensure_finite=True)

    maps_before_mask = _non_empty_maps(maps_data)
    maps_data, maps_mask, maps = trim_maps(
        maps_data,
        safe_get_data(mask_img, ensure_finite=True),
    )
    maps_mask = as_ndarray(maps_mask, dtype=bool)

    maps_after_mask = {int(map) for map in maps}
    maps_diff = maps_before_mask.difference(maps_after_mask)

    # Raising a warning if any map is removed due to the mask
    if maps_diff:
        if len(maps_after_mask) == 0:
            raise ValueError(
                "No map left after applying mask to the maps image."
            )

        warnings.warn(
            "After applying mask to the maps image, "
            "maps with the following indices were "
            f"removed: {maps_diff}. "
            f"Out of {len(maps_before_mask)} maps, the "
            "masked map image only contains "
            f"{len(maps_after_mask)} maps.",
            stacklevel=find_stack_level(),
        )
    return maps_data, maps_mask, maps


def trim_maps(maps, mask, keep_empty=False, order="F"):
    """Crop maps using a mask.

    No consistency check is performed (esp. on affine). Every required check
    must be performed before calling this function.

    Parameters
    ----------
    maps : :class:`numpy.ndarray`
        Set of maps, defining some regions.

    mask : :class:`numpy.ndarray`
        Definition of a mask. The shape must match that of a single map.

    keep_empty : :obj:`bool`, default=False
        If False, maps that lie completely outside the mask are dropped from
        the output. If True, they are kept, meaning that maps that are
        completely zero can occur in the output.

    order : "F" or "C", default="F"
        Ordering of the output maps array (trimmed_maps).

    Returns
    -------
    trimmed_maps : :class:``numpy.ndarray`
        New set of maps, computed as intersection of each input map and mask.
        Empty maps are discarded if keep_empty is False, thus the number of
        output maps is not necessarily the same as the number of input maps.
        shape: mask.shape + (output maps number,). Data ordering depends
        on the "order" parameter.

    maps_mask : :class:`numpy.ndarray`
        Union of all output maps supports. One non-zero value in this
        array guarantees that there is at least one output map that is
        non-zero at this voxel.
        shape: mask.shape. Order is always C.

    indices : :class:`numpy.ndarray`
        Indices of regions that have an non-empty intersection with the
        given mask. len(indices) == trimmed_maps.shape[-1].
    """
    maps = maps.copy()
    sums = abs(maps[as_ndarray(mask, dtype=bool), :]).sum(axis=0)

    n_regions = maps.shape[-1] if keep_empty else (sums > 0).sum()
    trimmed_maps = np.zeros(
        (*maps.shape[:3], n_regions), dtype=maps.dtype, order=order
    )
    # use int8 instead of np.bool for Nifti1Image
    maps_mask = np.zeros(mask.shape, dtype=np.int8)

    # iterate on maps
    p = 0
    mask = as_ndarray(mask, dtype=bool, order="C")
    for n, _ in enumerate(np.rollaxis(maps, -1)):
        if not keep_empty and sums[n] == 0:
            continue
        trimmed_maps[mask, p] = maps[mask, n]
        maps_mask[trimmed_maps[..., p] != 0] = 1
        p += 1

    indices = (
        np.arange(trimmed_maps.shape[-1], dtype=int)
        if keep_empty
        else np.where(sums > 0)[0]
    )

    return trimmed_maps, maps_mask, indices
