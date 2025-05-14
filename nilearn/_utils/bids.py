import warnings

import numpy as np
import pandas as pd

from nilearn._utils import check_niimg
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import safe_get_data
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data
from nilearn.typing import NiimgLike


def generate_atlas_look_up_table(
    function=None, name=None, index=None, strict=False
):
    """Generate a BIDS compatible look up table for an atlas.

    For a given deterministic atlas supported by Nilearn,
    this returns a pandas dataframe to use as look up table (LUT)
    between the name of a ROI and its index in the associated image.
    This LUT is compatible with the dseg.tsv BIDS format
    describing brain segmentations and parcellations,
    with an 'index' and 'name' column
    ('color' may be an example of an optional column).
    https://bids-specification.readthedocs.io/en/latest/derivatives/imaging.html#common-image-derived-labels

    For some atlases some 'clean up' of the LUT is done
    (for example make sure that the LUT contains the background 'ROI').

    This can also generate a look up table
    for an arbitrary niimg-like or surface image.

    Parameters
    ----------
    function : obj:`str` or None, default=None
        Atlas fetching function name as a string.
        Defaults to "unknown" in case None is passed.

    name : iterable of bytes or string, or int or None, default=None
        If an integer is passed,
        this corresponds to the number of ROIs in the atlas.
        If an iterable is passed, then it contains the ROI names.
        If None is passed, then it is inferred from index.

    index : iterable of integers, niimg like, surface image or None, \
            default=None
        If None, then the index of each ROI is derived from name.
        If a Niimg like or SurfaceImage is passed,
        then a LUT is generated for this image.

    strict: bool, default=False
        If True, an error will be thrown
        if ``name`` and ``index``have different length.
    """
    if name is None and index is None:
        raise ValueError("'index' and 'name' cannot both be None.")

    fname = "unknown" if function is None else function

    # deal with names
    if name is None:
        if fname == "unknown":
            index = _get_indices_from_image(index)
        name = [str(x) for x in index]

    # deal with indices
    if index is None:
        index = list(range(len(name)))
    else:
        index = _get_indices_from_image(index)
    if fname in ["fetch_atlas_basc_multiscale_2015"]:
        index = []
        for x in name:
            tmp = 0.0 if x in ["background", "Background"] else float(x)
            index.append(tmp)
    elif fname in ["fetch_atlas_schaefer_2018", "fetch_atlas_pauli_2017"]:
        index = list(range(1, len(name) + 1))

    if len(name) != len(index):
        if strict:
            raise ValueError(
                f"'name' ({len(name)}) and 'index' ({len(index)}) "
                "have different lengths. "
                "Cannot generate a look up table."
            )

        if len(name) < len(index):
            warnings.warn(
                "Too many indices for the names. "
                "Padding 'names' with 'unknown'.",
                stacklevel=find_stack_level(),
            )
            name += ["unknown"] * (len(index) - len(name))

        if len(name) > len(index):
            warnings.warn(
                "Too many names for the indices. "
                "Dropping excess names values.",
                stacklevel=find_stack_level(),
            )
            name = name[: len(index)]

    # convert to dataframe and do some cleaning where required
    lut = pd.DataFrame({"index": index, "name": name})

    if fname in [
        "fetch_atlas_pauli_2017",
    ]:
        lut = pd.concat(
            [pd.DataFrame([[0, "Background"]], columns=lut.columns), lut],
            ignore_index=True,
        )

    # enforce little endian of index column
    if lut["index"].dtype.byteorder == ">":
        lut["index"] = lut["index"].astype(
            lut["index"].dtype.newbyteorder("=")
        )

    return lut


def check_look_up_table(lut, atlas, strict=False, verbose=1):
    """Validate atlas look up table (LUT).

    Make sure it complies with BIDS requirements.

    Throws warning / errors:
    - lut is not a dataframe with the required columns
    - if there are mismatches between the number of ROIs
      in the LUT and the number of unique ROIs in the associated image.

    Parameters
    ----------
    lut : :obj:`pandas.DataFrame`
        Must be a pandas dataframe with at least "name" and "index" columns.

    atlas : Niimg like object or SurfaceImage or numpy array

    strict : bool, default = False
        Errors are raised instead of warnings if strict == True.

    verbose: int
        No warning thrown if set to 0.

    Raises
    ------
    AssertionError
        If:
        - lut is not a dataframe with the required columns
        - if there are mismatches between the number of ROIs
          in the LUT and the number of unique ROIs in the associated image.

    ValueError
        If regions in the image do not exist in the atlas lookup table
        and `strict=True`.

    Warns
    -----
    UserWarning
        If regions in the image do not exist in the atlas lookup table
        and `strict=False`.

    """
    assert isinstance(lut, pd.DataFrame)
    assert "name" in lut.columns
    assert "index" in lut.columns

    roi_id = _get_indices_from_image(atlas)

    if len(lut) != len(roi_id):
        if missing_from_image := set(lut["index"].to_list()) - set(roi_id):
            missing_rows = lut[
                lut["index"].isin(list(missing_from_image))
            ].to_string(index=False)
            msg = (
                "\nThe following regions are present "
                "in the atlas look-up table,\n"
                "but missing from the atlas image:\n\n"
                f"{missing_rows}\n"
            )
            if strict:
                raise ValueError(msg)
            if verbose:
                warnings.warn(msg, stacklevel=find_stack_level())

        if missing_from_lut := set(roi_id) - set(lut["index"].to_list()):
            msg = (
                "\nThe following regions are present "
                "in the atlas image, \n"
                "but missing from the atlas look-up table: \n\n"
                f"{missing_from_lut}"
            )
            if strict:
                raise ValueError(msg)
            if verbose:
                warnings.warn(msg, stacklevel=find_stack_level())


def sanitize_look_up_table(lut, atlas) -> pd.DataFrame:
    """Remove entries in lut that are missing from image."""
    check_look_up_table(lut, atlas, strict=False, verbose=0)
    indices = _get_indices_from_image(atlas)
    lut = lut[lut["index"].isin(indices)]
    return lut


def _get_indices_from_image(image):
    if isinstance(image, NiimgLike):
        img = check_niimg(image)
        data = safe_get_data(img)
    elif isinstance(image, SurfaceImage):
        data = get_surface_data(image)
    elif isinstance(image, np.ndarray):
        data = image
    else:
        raise TypeError(
            "Image to extract indices from must be one of: "
            "Niimg-Like, SurfaceIamge, numpy array. "
            f"Got {type(image)}"
        )
    return np.unique(data)
