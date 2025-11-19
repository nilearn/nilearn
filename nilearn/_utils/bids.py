import warnings

import pandas as pd

from nilearn._utils.logger import find_stack_level
from nilearn.image.image import get_indices_from_image


def generate_atlas_look_up_table(
    function=None, name=None, index=None, strict=False, background_label=None
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

    background_label: str or None, default=None
        If not None and no 'name' was passed,
        this label is used to describe the background value in the image.
    """
    if name is None and index is None:
        raise ValueError("'index' and 'name' cannot both be None.")

    fname = "unknown" if function is None else function

    # deal with names
    if name is None:
        if fname == "unknown":
            index = get_indices_from_image(index)
        name = []
        for x in index:
            if background_label is not None and x == background_label:
                name.append("Background")
            else:
                name.append(str(x))

    # deal with indices
    if index is None:
        index = list(range(len(name)))
    else:
        index = get_indices_from_image(index).tolist()
    if fname in ["fetch_atlas_basc_multiscale_2015"]:
        index = []
        for x in name:
            tmp = 0.0 if x in ["background", "Background"] else float(x)
            index.append(tmp)
    elif fname in ["fetch_atlas_schaefer_2018", "fetch_atlas_pauli_2017"]:
        index = list(range(1, len(name) + 1))

    if (
        background_label is not None
        and "Background" not in name
        and background_label in index
    ):
        name.insert(index.index(background_label), "Background")

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


def check_look_up_table(lut: pd.DataFrame, atlas, strict=False, verbose=1):
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

    roi_id = get_indices_from_image(atlas)

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


def sanitize_look_up_table(lut: pd.DataFrame, atlas) -> pd.DataFrame:
    """Sanitize lookup table.

    - remove entries in lut that are missing from image.
    - add 'unknown' entries in lut image indices missing from lut.
    """
    check_look_up_table(lut, atlas, strict=False, verbose=0)

    indices = get_indices_from_image(atlas)
    lut = lut[lut["index"].isin(indices)]

    missing_from_lut = sorted(
        set(indices.tolist()) - set(lut["index"].to_list())
    )
    if missing_from_lut:
        missing_rows = pd.DataFrame(
            {
                "name": ["unknown"] * len(missing_from_lut),
                "index": missing_from_lut,
            }
        )
        lut = pd.concat([lut, missing_rows], ignore_index=True)

    return lut
