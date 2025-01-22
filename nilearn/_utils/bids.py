import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel import Nifti1Image

from nilearn._utils import check_niimg
from nilearn._utils.niimg import safe_get_data
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data


def generate_atlas_look_up_table(function=None, name=None, index=None):
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

    index : iterable of integers, niimg like or None, default=None
        If None, then the index of each ROI is derived from name.
        If a Niimg like or SurfaceImage is passed,
        then a LUT is generated for this image.
    """
    if name is None and index is None:
        raise ValueError("'index' and 'name' cannot both be None.")

    fname = "unknown" if function is None else function

    # deal with names
    if name is None:
        if fname == "unknown":
            if isinstance(index, (str, Path, Nifti1Image)):
                img = check_niimg(index)
                index = np.unique(safe_get_data(img))
            elif isinstance(index, SurfaceImage):
                index = np.unique(get_surface_data(index))
        name = [str(x) for x in index]

    # deal with indices
    if index is None:
        index = list(range(len(name)))
    if fname in ["fetch_atlas_basc_multiscale_2015"]:
        index = []
        for x in name:
            tmp = x if isinstance(x, str) else int(x)
            index.append(tmp)
    elif fname in ["fetch_atlas_schaefer_2018", "fetch_atlas_pauli_2017"]:
        index = list(range(1, len(name) + 1))

    # convert to dataframe and do some cleaning where required
    lut = pd.DataFrame({"index": index, "name": name})

    if fname in [
        "fetch_atlas_pauli_2017",
    ]:
        lut = pd.concat(
            [pd.DataFrame([[0, "Background"]], columns=lut.columns), lut],
            ignore_index=True,
        )

    return lut


def check_look_up_table(lut, atlas, strict=False):
    """Validate atlas look up table (LUT).

    Make sure it complies with BIDS requirements.

    Throws warning / errors:
    - lut is not a dataframe with the required columns
    - if there are mismatches between the number of ROIs
      in the LUT and th number of unique ROIs in the associated image.

    Parameters
    ----------
    lut : :obj:`pandas.DataFrame`
        Must be a pandas dataframe with at least "name" and "index" columns.

    atlas : Niimg like object or SurfaceImage

    strict : bool, default = False
        Errors are raised instead of warnings if strict == True.

    Raises
    ------
    AssertionError
        If:
        - lut is not a dataframe with the required columns
        - if there are mismatches between the number of ROIs
          in the LUT and th number of unique ROIs in the associated image.

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

    if isinstance(atlas, (str, Path)):
        atlas = check_niimg(atlas)

    if isinstance(atlas, Nifti1Image):
        data = safe_get_data(atlas, ensure_finite=True)
    elif isinstance(atlas, SurfaceImage):
        data = get_surface_data(atlas)
    elif isinstance(atlas, np.ndarray):
        data = atlas

    roi_id = np.unique(data)

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
            warnings.warn(msg, stacklevel=3)

        if missing_from_lut := set(roi_id) - set(lut["index"].to_list()):
            msg = (
                "\nThe following regions are present "
                "in the atlas image,\n"
                "but missing from the atlas look-up table:\n\n"
                f"{missing_from_lut}"
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=3)
