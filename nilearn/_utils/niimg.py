"""Neuroimaging file input and output."""

import collections.abc
import warnings
from copy import deepcopy
from pathlib import Path
from warnings import warn

import numpy as np
from nibabel import Nifti1Image, is_proxy, load, spatialimages

from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level
from nilearn._utils.numpy_conversions import as_ndarray, get_target_dtype


def _get_data(img: Nifti1Image) -> np.ndarray:
    # copy-pasted from
    # https://github.com/nipy/nibabel/blob/de44a10/nibabel/dataobj_images.py#L204
    #
    # get_data is removed from nibabel because:
    # see https://github.com/nipy/nibabel/wiki/BIAP8
    if img._data_cache is not None:
        return img._data_cache
    data = np.asanyarray(img._dataobj)
    img._data_cache = data
    return data


def safe_get_data(
    img: Nifti1Image, ensure_finite: bool = False, copy_data: bool = False
) -> np.ndarray:
    """Get the data in the image without having a side effect \
    on the Nifti1Image object.

    This function will create a copy of the image and load data to new image's
    cache, leaving image's state unchanged if:

    - image data is not loaded to cache,
    - ``copy_data`` is `True`,

    If ``ensure_finite`` is set to `True`, this is no more guaranteed. If data
    is not cached, it will be cached and infinite values are replaced with 0.

    Parameters
    ----------
    img : Nifti image/object
        Image to get data.

    ensure_finite : :obj:`bool`
        If True, non-finite values such as (NaNs and infs) found in the
        image will be replaced by zeros.

    copy_data : :obj:`bool`, default=False
        If true, the returned data is a copy of the img data.

    Returns
    -------
    data : numpy array
        Data returned by nilearn.image.get_data from a Nifti image.
    """
    if not img.in_memory or copy_data:
        img = deepcopy(img)

    data = _get_data(img)

    if ensure_finite:
        ensure_finite_data(data)
    return data


def has_non_finite(data: np.ndarray) -> tuple[bool, np.ndarray]:
    """Return True if data contains at least one NaN or inf value; False if
    there are no NaN and inf values.

    Besides boolean value, return the mask.
    """
    non_finite_mask = ~np.isfinite(data)
    has_not_finite = bool(non_finite_mask.any())
    return has_not_finite, non_finite_mask


def ensure_finite_data(
    data: np.ndarray, raise_warning: bool = True
) -> np.ndarray:
    """Check if data contains NaN or inf values, set infinite values
    to 0 inplace if exists and return data.
    """
    has_not_finite, non_finite_mask = has_non_finite(data)
    if has_not_finite:
        if raise_warning:
            warn(
                "Non-finite values detected. "
                "These values will be replaced with zeros.",
                stacklevel=find_stack_level(),
            )
        data[non_finite_mask] = 0
    return data


@fill_doc
def load_niimg(niimg, dtype=None):
    """Load a niimg, check if it is a nibabel SpatialImage and cast if needed.

    Parameters
    ----------
    niimg : Niimg-like object
        See :ref:`extracting_data`.
        Image to load.

    %(dtype)s

    Returns
    -------
    img : image
        A loaded image object.
    """
    from nilearn.image.image import new_img_like  # avoid circular imports

    niimg = stringify_path(niimg)
    if isinstance(niimg, str):
        # data is a filename, we load it
        niimg = load(niimg)
    elif not isinstance(niimg, spatialimages.SpatialImage):
        raise TypeError(
            "Data given cannot be loaded because it is"
            " not compatible with nibabel format:\n"
            + repr_niimgs(niimg, shorten=True)
        )

    # avoid loading data if dtype is None
    if dtype is not None:
        img_data = _get_data(niimg)
        target_dtype = get_target_dtype(img_data.dtype, dtype)

        if target_dtype is not None:
            copy_header = niimg.header is not None
            niimg = new_img_like(
                niimg, img_data.astype(target_dtype), niimg.affine
            )
            if copy_header:
                niimg.header.set_data_dtype(target_dtype)

    return niimg


def is_binary_niimg(
    niimg: Nifti1Image,
    block_size: int = 1_000_000,
    accept_non_finite: bool = True,
) -> bool:
    """Return whether a given niimg is binary or not.

    Parameters
    ----------
    niimg : Niimg-like object
        See :ref:`extracting_data`.
        Image to test.

    block_size : :obj:`int`, default = 1_000_000

    accept_non_finite : :obj:`bool`, default = True

    Returns
    -------
    is_binary : :obj:`bool`
        True if binary, False otherwise.
    """
    niimg = load_niimg(niimg)
    data = niimg.dataobj
    return is_binary_data(data, block_size, accept_non_finite)


def _binary_mask(block: np.ndarray) -> np.ndarray:
    """Create a boolean mask for values equal to 0 or 1."""
    return (block == 0) | (block == 1)


def _binary_mask_with_nonfinite(block: np.ndarray) -> np.ndarray:
    """Create a boolean mask for values equal to 0, 1, nan, or +-inf."""
    return _binary_mask(block) | ~np.isfinite(block)


def is_binary_data(data, block_size=1_000_000, accept_non_finite=True) -> bool:
    """Return whether a given proxy array or ndarray is binary or not.
    If accept_non_finite is True, NaN and inf values are ignored.
    """
    flat = np.ravel(data)

    mask_func = (
        _binary_mask_with_nonfinite if accept_non_finite else _binary_mask
    )

    for i in range(0, flat.size, block_size):
        block = flat[i : i + block_size]

        mask = mask_func(block)
        if not mask.all():
            return False

    return True


def repr_niimgs(niimgs, shorten=True):
    """Pretty printing of niimg or niimgs.

    Parameters
    ----------
    niimgs : image or collection of images
        nibabel SpatialImage or SurfaceImage to repr.

    shorten : :obj:`bool`, default=True
        If True, filenames with more than 20 characters will be
        truncated, and lists of more than 3 file names will be
        printed with only first and last element.

    Returns
    -------
    repr : :obj:`str`
        String representation of the image.
    """
    # Simple string case
    if isinstance(niimgs, (str, Path)):
        return _short_repr(niimgs, shorten=shorten)

    # SurfaceImage imports repr_niimgs, so import locally to avoid a cycle.
    from nilearn.surface.surface import SurfaceImage

    if isinstance(niimgs, SurfaceImage):
        return repr(niimgs)

    # Collection case
    if isinstance(niimgs, collections.abc.Iterable):
        # Maximum number of elements to be displayed
        # Note: should be >= 3 to make sense...
        list_max_display = 3
        if len(niimgs) > list_max_display:
            if shorten:
                tmp = ",\n         ...\n ".join(
                    repr_niimgs(niimg, shorten=shorten)
                    for niimg in [niimgs[0], niimgs[-1]]
                )
            else:
                tmp = ",\n ".join(
                    repr_niimgs(niimg, shorten=shorten) for niimg in niimgs
                )
            return f"[\n {tmp},\n]"
        else:
            tmp = [repr_niimgs(niimg, shorten=shorten) for niimg in niimgs]
            return f"[\n {', '.join(tmp)},\n]"

    # Nibabel objects have a 'get_filename'
    try:
        filename = niimgs.get_filename()
        if filename is not None:
            return (
                f"{niimgs.__class__.__name__}"
                f"('{_short_repr(filename, shorten=shorten)}')"
            )
        else:
            # No shortening in this case
            return (
                f"{niimgs.__class__.__name__}"
                f"(\nshape={niimgs.shape!r},"
                f"\naffine={niimgs.affine!r}\n)"
            )
    except Exception:
        pass
    return _short_repr(repr(niimgs), shorten=shorten)


def _short_repr(niimg_rep, shorten: bool = True, truncate: int = 20) -> str:
    """Give a shorter version of niimg representation."""
    # Make sure truncate has a reasonable value
    truncate = max(truncate, 10)
    path_to_niimg = Path(niimg_rep)
    if not shorten:
        return str(path_to_niimg)
    # If the name of the file itself
    # is larger than truncate,
    # then shorten the name only
    # else add some folder structure if available
    if len(path_to_niimg.name) > truncate:
        return f"{path_to_niimg.name[: (truncate - 2)]}..."
    rep = path_to_niimg.name
    if len(path_to_niimg.parts) > 1:
        for p in path_to_niimg.parts[::-1][1:]:
            if len(rep) + len(p) < truncate - 3:
                rep = str(Path(p, rep))
            else:
                rep = str(Path("...", rep))
                break
    return rep


def img_data_dtype(img):
    """Determine type of data contained in image.

    Based on the information contained in ``niimg.dataobj``, determine the
    dtype of ``np.array(niimg.dataobj).dtype``.
    """
    dataobj = img.dataobj

    # Neuroimages that scale data should be interpreted as floating point
    if is_proxy(dataobj) and (dataobj.slope, dataobj.inter) != (
        1.0,
        0.0,
    ):
        return np.float64

    # ArrayProxy gained the dtype attribute in nibabel 2.2
    return dataobj.dtype if hasattr(dataobj, "dtype") else img.get_data_dtype()


def find_non_empty_maps(maps_data: np.ndarray) -> set[int]:
    """Return the indices of the maps that contain at least one non-zero."""
    return {int(i) for i in np.flatnonzero(np.any(maps_data, axis=(0, 1, 2)))}


def apply_mask_to_maps_img(
    mask_img: Nifti1Image, maps_img: Nifti1Image
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Apply a mask image to maps image.

    Checks that any map is left and warns if some were dropped by masking.
    """
    maps_data = safe_get_data(maps_img, ensure_finite=True)

    maps_before_mask = find_non_empty_maps(maps_data)
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
