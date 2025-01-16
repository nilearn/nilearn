"""Downloading NeuroImaging datasets: atlas datasets."""

import json
import re
import shutil
import warnings
from pathlib import Path
from tempfile import mkdtemp
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from nibabel import Nifti1Image, freesurfer, load
from sklearn.utils import Bunch

from nilearn._utils import check_niimg, fill_doc, logger, rename_parameters
from nilearn._utils.niimg import safe_get_data
from nilearn.datasets._utils import (
    PACKAGE_DIRECTORY,
    fetch_files,
    get_dataset_descr,
    get_dataset_dir,
)
from nilearn.image import get_data as get_img_data
from nilearn.image import new_img_like, reorder_img
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data

_TALAIRACH_LEVELS = ["hemisphere", "lobe", "gyrus", "tissue", "ba"]


dec_to_hex_nums = pd.DataFrame(
    {"hex": [f"{x:02x}" for x in range(256)]}, dtype=str
)

deprecation_message = (
    "From release >={version}, "
    "instead of returning several atlas image accessible "
    "via different keys, "
    "this fetcher will return the atlas as a dictionary "
    "with a single atlas image, "
    "accessible through a 'maps' key. "
)


def rgb_to_hex_lookup(
    red: pd.Series, green: pd.Series, blue: pd.Series
) -> pd.Series:
    """Turn RGB in hex."""
    # see https://stackoverflow.com/questions/53875880/convert-a-pandas-dataframe-of-rgb-colors-to-hex
    # Look everything up
    rr = dec_to_hex_nums.loc[red, "hex"]
    gg = dec_to_hex_nums.loc[green, "hex"]
    bb = dec_to_hex_nums.loc[blue, "hex"]
    # Reindex
    rr.index = red.index
    gg.index = green.index
    bb.index = blue.index
    # Concatenate and return
    return rr + gg + bb


def _generate_atlas_look_up_table(function=None, name=None, index=None):
    """Generate a look up table for an atlas.

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


def _check_look_up_table(lut, atlas, strict=False):
    """Validate atlas look up table (LUT).

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


@fill_doc
def fetch_atlas_difumo(
    dimension=64,
    resolution_mm=2,
    data_dir=None,
    resume=True,
    verbose=1,
):
    """Fetch DiFuMo brain atlas.

    Dictionaries of Functional Modes, or “DiFuMo”, can serve as
    :term:`probabilistic atlases<Probabilistic atlas>` to extract
    functional signals with different dimensionalities (64, 128,
    256, 512, and 1024).
    These modes are optimized to represent well raw :term:`BOLD` timeseries,
    over a with range of experimental conditions.
    See :footcite:t:`Dadi2020`.

    .. versionadded:: 0.7.1

    Notes
    -----
    Direct download links from OSF:

    - 64: https://osf.io/pqu9r/download
    - 128: https://osf.io/wjvd5/download
    - 256: https://osf.io/3vrct/download
    - 512: https://osf.io/9b76y/download
    - 1024: https://osf.io/34792/download

    Parameters
    ----------
    dimension : :obj:`int`, default=64
        Number of dimensions in the dictionary. Valid resolutions
        available are {64, 128, 256, 512, 1024}.

    resolution_mm : :obj:`int`, default=2mm
        The resolution in mm of the atlas to fetch. Valid options
        available are {2, 3}.
    %(data_dir)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, the interest attributes are :

        - 'maps': :obj:`str`, path to 4D nifti file containing regions
            definition. The shape of the image is
            ``(104, 123, 104, dimension)`` where ``dimension`` is the
            requested dimension of the atlas.

        - 'labels': :class:`pandas.DataFrame` containing the labels of
            the regions.
            The length of the label array corresponds to the
            number of dimensions requested. ``data.labels[i]`` is the label
            corresponding to volume ``i`` in the 'maps' image.

        - %(description)s

        - %(atlas_type)s

        - %(template)s

    References
    ----------
    .. footbibliography::

    """
    atlas_type = "probabilistic"

    dic = {
        64: "pqu9r",
        128: "wjvd5",
        256: "3vrct",
        512: "9b76y",
        1024: "34792",
    }
    valid_dimensions = [64, 128, 256, 512, 1024]
    valid_resolution_mm = [2, 3]
    if dimension not in valid_dimensions:
        raise ValueError(
            f"Requested dimension={dimension} is not available. "
            f"Valid options: {valid_dimensions}"
        )
    if resolution_mm not in valid_resolution_mm:
        raise ValueError(
            f"Requested resolution_mm={resolution_mm} is not available. "
            f"Valid options: {valid_resolution_mm}"
        )

    url = f"https://osf.io/{dic[dimension]}/download"
    opts = {"uncompress": True}

    csv_file = Path(f"{dimension}", f"labels_{dimension}_dictionary.csv")
    if resolution_mm != 3:
        nifti_file = Path(f"{dimension}", "2mm", "maps.nii.gz")
    else:
        nifti_file = Path(f"{dimension}", "3mm", "maps.nii.gz")

    files = [
        (csv_file, url, opts),
        (nifti_file, url, opts),
    ]

    dataset_name = "difumo_atlases"

    data_dir = get_dataset_dir(
        dataset_name=dataset_name, data_dir=data_dir, verbose=verbose
    )

    # Download the zip file, first
    files_ = fetch_files(data_dir, files, verbose=verbose, resume=resume)
    labels = pd.read_csv(files_[0])
    labels = labels.rename(columns={c: c.lower() for c in labels.columns})

    # README
    readme_files = [
        ("README.md", "https://osf.io/4k9bf/download", {"move": "README.md"})
    ]
    if not (data_dir / "README.md").exists():
        fetch_files(data_dir, readme_files, verbose=verbose, resume=resume)

    return Atlas(
        maps=files_[1],
        labels=labels,
        description=get_dataset_descr(dataset_name),
        atlas_type=atlas_type,
    )


@fill_doc
def fetch_atlas_craddock_2012(
    data_dir=None,
    url=None,
    resume=True,
    verbose=1,
    homogeneity=None,
    grp_mean=True,
):
    """Download and return file names \
       for the Craddock 2012 :term:`parcellation`.

    This function returns a :term:`probabilistic atlas<Probabilistic atlas>`.
    The provided images are in MNI152 space. All images are 4D with
    shapes equal to ``(47, 56, 46, 43)``.

    See :footcite:t:`CreativeCommons` for the license.

    See :footcite:t:`Craddock2012` and :footcite:t:`nitrcClusterROI`
    for more information on this :term:`parcellation`.

    Parameters
    ----------
    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    homogeneity : :obj:`str`,  default=None
        The choice of the homogeneity ('spatial' or 'temporal' or 'random')
    grp_mean : :obj:`bool`, default=True
        The choice of the :term:`parcellation` (with group_mean or without)


    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, keys are:

        - ``'scorr_mean'``: :obj:`str`, path to nifti file containing
            the group-mean :term:`parcellation`
            when emphasizing spatial homogeneity.

        - ``'tcorr_mean'``: :obj:`str`, path to nifti file containing
            the group-mean parcellation when emphasizing temporal homogeneity.

        - ``'scorr_2level'``: :obj:`str`, path to nifti file containing
            the :term:`parcellation` obtained
            when emphasizing spatial homogeneity.

        - ``'tcorr_2level'``: :obj:`str`, path to nifti file containing
            the :term:`parcellation` obtained
            when emphasizing temporal homogeneity.

        - ``'random'``: :obj:`str`, path to nifti file containing
            the :term:`parcellation` obtained with random clustering.

        - %(description)s

        - %(atlas_type)s

        - %(template)s


    Warns
    -----
    DeprecationWarning
        If an homogeneity input is provided, the current behavior
        (returning multiple maps) is deprecated.
        Starting in version 0.13, one map will be returned in a 'maps' dict key
        depending on the homogeneity and grp_mean value.

    References
    ----------
    .. footbibliography::

    """
    atlas_type = "probabilistic"

    if url is None:
        url = (
            "https://cluster_roi.projects.nitrc.org"
            "/Parcellations/craddock_2011_parcellations.tar.gz"
        )
    opts = {"uncompress": True}

    dataset_name = "craddock_2012"

    keys = (
        "scorr_mean",
        "tcorr_mean",
        "scorr_2level",
        "tcorr_2level",
        "random",
    )
    filenames = [
        ("scorr05_mean_all.nii.gz", url, opts),
        ("tcorr05_mean_all.nii.gz", url, opts),
        ("scorr05_2level_all.nii.gz", url, opts),
        ("tcorr05_2level_all.nii.gz", url, opts),
        ("random_all.nii.gz", url, opts),
    ]

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    sub_files = fetch_files(
        data_dir, filenames, resume=resume, verbose=verbose
    )

    fdescr = get_dataset_descr(dataset_name)

    if homogeneity:
        if homogeneity in ["spatial", "temporal"]:
            if grp_mean:
                filename = [
                    (homogeneity[0] + "corr05_mean_all.nii.gz", url, opts)
                ]
            else:
                filename = [
                    (homogeneity[0] + "corr05_2level_all.nii.gz", url, opts)
                ]
        else:
            filename = [("random_all.nii.gz", url, opts)]
        data = fetch_files(data_dir, filename, resume=resume, verbose=verbose)

        return Atlas(
            maps=data[0],
            description=fdescr,
            atlas_type=atlas_type,
        )

    warnings.warn(
        category=DeprecationWarning,
        message=(
            deprecation_message.format(version="0.13")
            + (
                "To suppress this warning, "
                "Please use the parameters 'homogeneity' and 'grp_mean' "
                "to specify the exact atlas image you want."
            )
        ),
    )

    params = dict(
        [
            ("description", fdescr),
            *list(zip(keys, sub_files)),
        ]
    )
    params["atlas_type"] = atlas_type

    return Bunch(**params)


@fill_doc
def fetch_atlas_destrieux_2009(
    lateralized=True,
    data_dir=None,
    url=None,
    resume=True,
    verbose=1,
):
    """Download and load the Destrieux cortical \
    :term:`deterministic atlas<Deterministic atlas>` (dated 2009).

    See :footcite:t:`Fischl2004`,
    and :footcite:t:`Destrieux2009`.

    .. note::

        Some labels from the list of labels might not be present
        in the atlas image,
        in which case the integer values in the image
        might not be consecutive.

    Parameters
    ----------
    lateralized : :obj:`bool`, default=True
        If True, returns an atlas with distinct regions for right and left
        hemispheres.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'maps': :obj:`str`
            path to nifti file containing the
            :class:`~nibabel.nifti1.Nifti1Image` defining the cortical
            ROIs, lateralized or not. The image has shape ``(76, 93, 76)``,
            and contains integer values which can be interpreted as the
            indices in the list of labels.

        - %(labels)s

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s

    References
    ----------
    .. footbibliography::

    """
    atlas_type = "deterministic"

    if url is None:
        url = "https://www.nitrc.org/frs/download.php/11942/"

    url += "destrieux2009.tgz"
    opts = {"uncompress": True}
    lat = "_lateralized" if lateralized else ""

    files = [
        (f"destrieux2009_rois_labels{lat}.csv", url, opts),
        (f"destrieux2009_rois{lat}.nii.gz", url, opts),
        ("destrieux2009.rst", url, opts),
    ]

    dataset_name = "destrieux_2009"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    files_ = fetch_files(data_dir, files, resume=resume, verbose=verbose)

    labels = pd.read_csv(files_[0], index_col=0)

    return Atlas(
        maps=files_[1],
        labels=labels.name.to_list(),
        description=Path(files_[2]).read_text(),
        atlas_type=atlas_type,
        lut=pd.read_csv(files_[0]),
    )


@fill_doc
def fetch_atlas_harvard_oxford(
    atlas_name, data_dir=None, symmetric_split=False, resume=True, verbose=1
):
    """Load Harvard-Oxford parcellations from FSL.

    This function downloads Harvard Oxford atlas packaged from FSL 5.0
    and stores atlases in NILEARN_DATA folder in home directory.

    This function can also load Harvard Oxford atlas from your local directory
    specified by your FSL installed path given in `data_dir` argument.
    See documentation for details.

    .. note::

        For atlases 'cort-prob-1mm', 'cort-prob-2mm', 'cortl-prob-1mm',
        'cortl-prob-2mm', 'sub-prob-1mm', and 'sub-prob-2mm', the function
        returns a :term:`Probabilistic atlas`, and the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 4D, with shape
        ``(182, 218, 182, 48)``.
        For :term:`deterministic atlases<Deterministic atlas>`, the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 3D, with
        shape ``(182, 218, 182)`` and 48 regions (+ background).

    Parameters
    ----------
    atlas_name : :obj:`str`
        Name of atlas to load. Can be:
        "cort-maxprob-thr0-1mm", "cort-maxprob-thr0-2mm",
        "cort-maxprob-thr25-1mm", "cort-maxprob-thr25-2mm",
        "cort-maxprob-thr50-1mm", "cort-maxprob-thr50-2mm",
        "cort-prob-1mm", "cort-prob-2mm",
        "cortl-maxprob-thr0-1mm", "cortl-maxprob-thr0-2mm",
        "cortl-maxprob-thr25-1mm", "cortl-maxprob-thr25-2mm",
        "cortl-maxprob-thr50-1mm", "cortl-maxprob-thr50-2mm",
        "cortl-prob-1mm", "cortl-prob-2mm",
        "sub-maxprob-thr0-1mm", "sub-maxprob-thr0-2mm",
        "sub-maxprob-thr25-1mm", "sub-maxprob-thr25-2mm",
        "sub-maxprob-thr50-1mm", "sub-maxprob-thr50-2mm",
        "sub-prob-1mm", "sub-prob-2mm".
    %(data_dir)s
        Optionally, it can also be a FSL installation directory (which is
        dependent on your installation).
        Example, if FSL is installed in ``/usr/share/fsl/`` then
        specifying as '/usr/share/' can get you the Harvard Oxford atlas
        from your installed directory. Since we mimic the same root directory
        as FSL to load it easily from your installation.

    symmetric_split : :obj:`bool`, default=False
        If ``True``, lateralized atlases of cort or sub with maxprob will be
        returned. For subcortical types (``sub-maxprob``), we split every
        symmetric region in left and right parts. Effectively doubles the
        number of regions.

        .. note::
            Not implemented
            for full :term:`Probabilistic atlas` (*-prob-* atlases).

    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, keys are:

        - 'maps': :obj:`str`
            path to nifti file containing the
            atlas :class:`~nibabel.nifti1.Nifti1Image`.
            It is a 4D image
            if a :term:`Probabilistic atlas` is requested, and a 3D image
            if a :term:`maximum probability atlas<Deterministic atlas>` is
            requested.
            In the latter case, the image contains integer
            values which can be interpreted as the indices in the list
            of labels.

            .. note::

                For some atlases, it can be the case that some regions
                are empty. In this case, no :term:`voxels<voxel>` in the
                map are assigned to these regions. So the number of
                unique values in the map can be strictly smaller than the
                number of region names in ``labels``.

        - %(labels)s

        - 'filename': Same as 'maps', kept for backward compatibility only.

        - %(description)s

        - %(lut)s
            Only for deterministic version of the atlas.

        - %(template)s

        - %(atlas_type)s

    See Also
    --------
    nilearn.datasets.fetch_atlas_juelich

    """
    atlases = [
        "cort-maxprob-thr0-1mm",
        "cort-maxprob-thr0-2mm",
        "cort-maxprob-thr25-1mm",
        "cort-maxprob-thr25-2mm",
        "cort-maxprob-thr50-1mm",
        "cort-maxprob-thr50-2mm",
        "cort-prob-1mm",
        "cort-prob-2mm",
        "cortl-maxprob-thr0-1mm",
        "cortl-maxprob-thr0-2mm",
        "cortl-maxprob-thr25-1mm",
        "cortl-maxprob-thr25-2mm",
        "cortl-maxprob-thr50-1mm",
        "cortl-maxprob-thr50-2mm",
        "cortl-prob-1mm",
        "cortl-prob-2mm",
        "sub-maxprob-thr0-1mm",
        "sub-maxprob-thr0-2mm",
        "sub-maxprob-thr25-1mm",
        "sub-maxprob-thr25-2mm",
        "sub-maxprob-thr50-1mm",
        "sub-maxprob-thr50-2mm",
        "sub-prob-1mm",
        "sub-prob-2mm",
    ]
    if atlas_name not in atlases:
        atlases = "\n".join(atlases)
        raise ValueError(
            f"Invalid atlas name: {atlas_name}. "
            f"Please choose an atlas among:\n{atlases}"
        )

    atlas_type = "probabilistic" if "-prob-" in atlas_name else "deterministic"

    if atlas_type == "probabilistic" and symmetric_split:
        raise ValueError(
            "Region splitting not supported for probabilistic atlases"
        )
    (
        atlas_img,
        atlas_filename,
        names,
        is_lateralized,
    ) = _get_atlas_data_and_labels(
        "HarvardOxford",
        atlas_name,
        symmetric_split=symmetric_split,
        data_dir=data_dir,
        resume=resume,
        verbose=verbose,
    )

    atlas_niimg = check_niimg(atlas_img)
    if not symmetric_split or is_lateralized:
        return Atlas(
            maps=atlas_niimg,
            labels=names,
            description=get_dataset_descr("harvard_oxford"),
            atlas_type=atlas_type,
            lut=_generate_atlas_look_up_table(
                "fetch_atlas_harvard_oxford", name=names
            ),
            filename=atlas_filename,
        )

    new_atlas_data, new_names = _compute_symmetric_split(
        "HarvardOxford", atlas_niimg, names
    )
    new_atlas_niimg = new_img_like(
        atlas_niimg, new_atlas_data, atlas_niimg.affine
    )

    return Atlas(
        maps=new_atlas_niimg,
        labels=new_names,
        description=get_dataset_descr("harvard_oxford"),
        atlas_type=atlas_type,
        lut=_generate_atlas_look_up_table(
            "fetch_atlas_harvard_oxford", name=new_names
        ),
        filename=atlas_filename,
    )


@fill_doc
def fetch_atlas_juelich(
    atlas_name, data_dir=None, symmetric_split=False, resume=True, verbose=1
):
    """Load Juelich parcellations from FSL.

    This function downloads Juelich atlas packaged from FSL 5.0
    and stores atlases in NILEARN_DATA folder in home directory.

    This function can also load Juelich atlas from your local directory
    specified by your FSL installed path given in `data_dir` argument.
    See documentation for details.

    .. versionadded:: 0.8.1

    .. note::

        For atlases 'prob-1mm', and 'prob-2mm', the function returns a
        :term:`Probabilistic atlas`, and the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 4D, with shape
        ``(182, 218, 182, 62)``.
        For :term:`deterministic atlases<Deterministic atlas>`, the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 3D, with shape
        ``(182, 218, 182)`` and 62 regions (+ background).

    Parameters
    ----------
    atlas_name : :obj:`str`
        Name of atlas to load. Can be:
        "maxprob-thr0-1mm", "maxprob-thr0-2mm",
        "maxprob-thr25-1mm", "maxprob-thr25-2mm",
        "maxprob-thr50-1mm", "maxprob-thr50-2mm",
        "prob-1mm", "prob-2mm".
    %(data_dir)s
        Optionally, it can also be a FSL installation directory (which is
        dependent on your installation).
        Example, if FSL is installed in ``/usr/share/fsl/``, then
        specifying as '/usr/share/' can get you Juelich atlas
        from your installed directory. Since we mimic same root directory
        as FSL to load it easily from your installation.

    symmetric_split : :obj:`bool`, default=False
        If ``True``, lateralized atlases of cort or sub with maxprob will be
        returned. For subcortical types (``sub-maxprob``), we split every
        symmetric region in left and right parts. Effectively doubles the
        number of regions.

        .. note::
            Not implemented for full :term:`Probabilistic atlas`
            (``*-prob-*`` atlases).

    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, keys are:

        - 'maps': :class:`~nibabel.nifti1.Nifti1Image`.
            It is a 4D image if a :term:`Probabilistic atlas` is requested,
            and a 3D image
            if a :term:`maximum probability atlas<Deterministic atlas>`
            is requested.
            In the latter case, the image contains integer values
            which can be interpreted as the indices in the list of labels.

            .. note::

                For some atlases, it can be the case that some regions
                are empty. In this case, no :term:`voxels<voxel>` in the
                map are assigned to these regions. So the number of
                unique values in the map can be strictly smaller than the
                number of region names in ``labels``.

        - %(labels)s

        - 'filename': Same as 'maps', kept for backward compatibility only.

        - %(description)s

        - %(lut)s
            Only for deterministic version of the atlas.

        - %(template)s

        - %(atlas_type)s

    See Also
    --------
    nilearn.datasets.fetch_atlas_harvard_oxford

    """
    atlases = [
        "maxprob-thr0-1mm",
        "maxprob-thr0-2mm",
        "maxprob-thr25-1mm",
        "maxprob-thr25-2mm",
        "maxprob-thr50-1mm",
        "maxprob-thr50-2mm",
        "prob-1mm",
        "prob-2mm",
    ]
    if atlas_name not in atlases:
        atlases = "\n".join(atlases)
        raise ValueError(
            f"Invalid atlas name: {atlas_name}. "
            f"Please choose an atlas among:\n{atlases}"
        )

    atlas_type = (
        "probabilistic" if atlas_name.startswith("prob-") else "deterministic"
    )

    if atlas_type == "probabilistic" and symmetric_split:
        raise ValueError(
            "Region splitting not supported for probabilistic atlases"
        )
    atlas_img, atlas_filename, names, _ = _get_atlas_data_and_labels(
        "Juelich",
        atlas_name,
        data_dir=data_dir,
        resume=resume,
        verbose=verbose,
    )
    atlas_niimg = check_niimg(atlas_img)
    atlas_data = get_img_data(atlas_niimg)

    if atlas_type == "probabilistic":
        new_atlas_data, new_names = _merge_probabilistic_maps_juelich(
            atlas_data, names
        )
    elif symmetric_split:
        new_atlas_data, new_names = _compute_symmetric_split(
            "Juelich", atlas_niimg, names
        )
    else:
        new_atlas_data, new_names = _merge_labels_juelich(atlas_data, names)

    new_atlas_niimg = new_img_like(
        atlas_niimg, new_atlas_data, atlas_niimg.affine
    )

    return Atlas(
        maps=new_atlas_niimg,
        labels=list(new_names),
        description=get_dataset_descr("juelich"),
        atlas_type=atlas_type,
        lut=_generate_atlas_look_up_table(
            "fetch_atlas_juelich", name=list(new_names)
        ),
        filename=atlas_filename,
    )


def _get_atlas_data_and_labels(
    atlas_source,
    atlas_name,
    symmetric_split=False,
    data_dir=None,
    resume=True,
    verbose=1,
):
    """Implement fetching logic common to \
    both fetch_atlas_juelich and fetch_atlas_harvard_oxford.

    This function downloads the atlas image and labels.
    """
    if atlas_source == "Juelich":
        url = "https://www.nitrc.org/frs/download.php/12096/Juelich.tgz"
    elif atlas_source == "HarvardOxford":
        url = "https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz"
    else:
        raise ValueError(f"Atlas source {atlas_source} is not valid.")
    # For practical reasons, we mimic the FSL data directory here.
    data_dir = get_dataset_dir("fsl", data_dir=data_dir, verbose=verbose)
    opts = {"uncompress": True}
    root = Path("data", "atlases")

    if atlas_source == "HarvardOxford":
        if symmetric_split:
            atlas_name = atlas_name.replace("cort-max", "cortl-max")

        if atlas_name.startswith("sub-"):
            label_file = "HarvardOxford-Subcortical.xml"
            is_lateralized = False
        elif atlas_name.startswith("cortl"):
            label_file = "HarvardOxford-Cortical-Lateralized.xml"
            is_lateralized = True
        else:
            label_file = "HarvardOxford-Cortical.xml"
            is_lateralized = False
    else:
        label_file = "Juelich.xml"
        is_lateralized = False
    label_file = root / label_file
    atlas_file = root / atlas_source / f"{atlas_source}-{atlas_name}.nii.gz"
    atlas_file, label_file = fetch_files(
        data_dir,
        [(atlas_file, url, opts), (label_file, url, opts)],
        resume=resume,
        verbose=verbose,
    )
    # Reorder image to have positive affine diagonal
    atlas_img = reorder_img(atlas_file, copy_header=True)
    names = {0: "Background"}

    all_labels = ElementTree.parse(label_file).findall(".//label")
    for label in all_labels:
        new_idx = int(label.get("index")) + 1
        if new_idx in names:
            raise ValueError(
                f"Duplicate index {new_idx} for labels "
                f"'{names[new_idx]}', and '{label.text}'"
            )

        # fix typos in Harvard Oxford labels
        if atlas_source == "HarvardOxford":
            label.text = label.text.replace("Ventrical", "Ventricle")
            label.text = label.text.replace("Operculum", "Opercular")

        names[new_idx] = label.text.strip()

    # The label indices should range from 0 to nlabel + 1
    assert list(names.keys()) == list(range(len(all_labels) + 1))
    names = [item[1] for item in sorted(names.items())]
    return atlas_img, atlas_file, names, is_lateralized


def _merge_probabilistic_maps_juelich(atlas_data, names):
    """Handle probabilistic juelich atlases when symmetric_split=False.

    Helper function for fetch_atlas_juelich.

    In this situation, we need to merge labels and maps corresponding
    to left and right regions.
    """
    new_names = np.unique([re.sub(r" (L|R)$", "", name) for name in names])
    new_name_to_idx = {k: v - 1 for v, k in enumerate(new_names)}
    new_atlas_data = np.zeros((*atlas_data.shape[:3], len(new_names) - 1))
    for i, name in enumerate(names):
        if name != "Background":
            new_name = re.sub(r" (L|R)$", "", name)
            new_atlas_data[..., new_name_to_idx[new_name]] += atlas_data[
                ..., i - 1
            ]
    return new_atlas_data, new_names


def _merge_labels_juelich(atlas_data, names):
    """Handle 3D atlases when symmetric_split=False.

    Helper function for fetch_atlas_juelich.

    In this case, we need to merge the labels corresponding to
    left and right regions.
    """
    new_names = np.unique([re.sub(r" (L|R)$", "", name) for name in names])
    new_names_dict = {k: v for v, k in enumerate(new_names)}
    new_atlas_data = atlas_data.copy()
    for label, name in enumerate(names):
        new_name = re.sub(r" (L|R)$", "", name)
        new_atlas_data[atlas_data == label] = new_names_dict[new_name]
    return new_atlas_data, new_names


def _compute_symmetric_split(source, atlas_niimg, names):
    """Handle 3D atlases when symmetric_split=True.

    Helper function for both fetch_atlas_juelich and
    fetch_atlas_harvard_oxford.
    """
    # The atlas_niimg should have been passed to
    # reorder_img such that the affine's diagonal
    # should be positive. This is important to
    # correctly split left and right hemispheres.
    assert atlas_niimg.affine[0, 0] > 0
    atlas_data = get_img_data(atlas_niimg)
    labels = np.unique(atlas_data)
    # Build a mask of both halves of the brain
    middle_ind = (atlas_data.shape[0]) // 2
    # Split every zone crossing the median plane into two parts.
    left_atlas = atlas_data.copy()
    left_atlas[middle_ind:] = 0
    right_atlas = atlas_data.copy()
    right_atlas[:middle_ind] = 0

    if source == "Juelich":
        for idx, name in enumerate(names):
            if name.endswith("L"):
                name = re.sub(r" L$", "", name)
                names[idx] = f"Left {name}"
            if name.endswith("R"):
                name = re.sub(r" R$", "", name)
                names[idx] = f"Right {name}"

    new_label = 0
    new_atlas = atlas_data.copy()
    # Assumes that the background label is zero.
    new_names = [names[0]]
    for label, name in zip(labels[1:], names[1:]):
        new_label += 1
        left_elements = (left_atlas == label).sum()
        right_elements = (right_atlas == label).sum()
        n_elements = float(left_elements + right_elements)
        if (
            left_elements / n_elements < 0.05
            or right_elements / n_elements < 0.05
        ):
            new_atlas[atlas_data == label] = new_label
            new_names.append(name)
            continue
        new_atlas[left_atlas == label] = new_label
        new_names.append(f"Left {name}")
        new_label += 1
        new_atlas[right_atlas == label] = new_label
        new_names.append(f"Right {name}")
    return new_atlas, new_names


@fill_doc
def fetch_atlas_msdl(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the MSDL brain :term:`Probabilistic atlas`.

    It can be downloaded at :footcite:t:`atlas_msdl`, and cited
    using :footcite:t:`Varoquaux2011`.
    See also :footcite:t:`Varoquaux2013` for more information.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, the interest attributes are :

        - 'maps': :obj:`str`
            path to nifti file containing the
            :term:`Probabilistic atlas` image
            (shape is equal to ``(40, 48, 35, 39)``).

        - %(labels)s
            There are 39 labels such that ``data.labels[i]``
            corresponds to map ``i``.

        - 'region_coords': :obj:`list` of length-3 :obj:`tuple`
            ``data.region_coords[i]`` contains the coordinates ``(x, y, z)``
            of region ``i`` in :term:`MNI` space.

        - 'networks': :obj:`list` of :obj:`str`
            list containing the names of the networks.
            There are 39 network names such that
            ``data.networks[i]`` is the network name of region ``i``.

        - %(description)s

        - %(atlas_type)s

        - %(template)s

    References
    ----------
    .. footbibliography::


    """
    atlas_type = "probabilistic"

    url = "https://team.inria.fr/parietal/files/2015/01/MSDL_rois.zip"
    opts = {"uncompress": True}

    dataset_name = "msdl_atlas"
    files = [
        (Path("MSDL_rois", "msdl_rois_labels.csv"), url, opts),
        (Path("MSDL_rois", "msdl_rois.nii"), url, opts),
    ]

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    files = fetch_files(data_dir, files, resume=resume, verbose=verbose)

    csv_data = pd.read_csv(files[0])
    net_names = [
        net_name.strip() for net_name in csv_data["net name"].to_list()
    ]

    return Atlas(
        maps=files[1],
        labels=[name.strip() for name in csv_data["name"].to_list()],
        description=get_dataset_descr(dataset_name),
        atlas_type=atlas_type,
        region_coords=csv_data[["x", "y", "z"]].to_numpy().tolist(),
        networks=net_names,
    )


@fill_doc
def fetch_coords_power_2011():
    """Download and load the Power et al. brain atlas composed of 264 ROIs.

    See :footcite:t:`Power2011`.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'rois': :class:`pandas.DataFrame`
            Contains the coordinates of 264 ROIs in :term:`MNI` space.

        - %(description)s


    References
    ----------
    .. footbibliography::

    """
    dataset_name = "power_2011"
    fdescr = get_dataset_descr(dataset_name)
    csv = PACKAGE_DIRECTORY / "data" / "power_2011.csv"
    params = {"rois": pd.read_csv(csv), "description": fdescr}
    params["rois"] = params["rois"].rename(
        columns={c: c.lower() for c in params["rois"].columns}
    )

    return Bunch(**params)


@fill_doc
def fetch_atlas_smith_2009(
    data_dir=None,
    url=None,
    resume=True,
    verbose=1,
    mirror="origin",
    dimension=None,
    resting=True,
):
    """Download and load the Smith :term:`ICA` and BrainMap \
    :term:`Probabilistic atlas` (2009).

    See :footcite:t:`Smith2009b` and :footcite:t:`Laird2011`.

    Parameters
    ----------
    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    mirror : :obj:`str`, default='origin'
        By default, the dataset is downloaded from the original website of the
        atlas. Specifying "nitrc" will force download from a mirror, with
        potentially higher bandwidth.

    dimension : :obj:`int`, default=None
        Number of dimensions in the dictionary. Valid resolutions
        available are {10, 20, 70}.

    resting : :obj:`bool`, default=True
        Either to fetch the resting-:term:`fMRI` or BrainMap components

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - ``'rsn20'``: :obj:`str`
            Path to nifti file containing
            the 20-dimensional :term:`ICA`, resting-:term:`fMRI` components.
            The shape of the image is ``(91, 109, 91, 20)``.

        - ``'rsn10'``: :obj:`str`
            Path to nifti file containing
            the 10 well-matched maps from the 20 maps obtained as for 'rsn20',
            as shown in :footcite:t:`Smith2009b`.
            The shape of the image is ``(91, 109, 91, 10)``.

        - ``'bm20'``: :obj:`str`
            Path to nifti file containing
            the 20-dimensional :term:`ICA`, BrainMap components.
            The shape of the image is ``(91, 109, 91, 20)``.

        - ``'bm10'``: :obj:`str`
            Path to nifti file containing
            the 10 well-matched maps from the 20 maps obtained as for 'bm20',
            as shown in :footcite:t:`Smith2009b`.
            The shape of the image is ``(91, 109, 91, 10)``.

        - ``'rsn70'``: :obj:`str`
            Path to nifti file containing
            the 70-dimensional :term:`ICA`, resting-:term:`fMRI` components.
            The shape of the image is ``(91, 109, 91, 70)``.

        - ``'bm70'``: :obj:`str`
            Path to nifti file containing
            the 70-dimensional :term:`ICA`, BrainMap components.
            The shape of the image is ``(91, 109, 91, 70)``.

        - %(description)s

        - %(atlas_type)s

        - %(template)s

    Warns
    -----
    DeprecationWarning
        If a dimension input is provided, the current behavior
        (returning multiple maps) is deprecated.
        Starting in version 0.13, one map will be returned in a 'maps' dict key
        depending on the dimension and resting value.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    For more information about this dataset's structure:
    https://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/

    """
    atlas_type = "probabilistic"

    if url is None:
        if mirror == "origin":
            url = "https://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/"
        elif mirror == "nitrc":
            url = [
                "https://www.nitrc.org/frs/download.php/7730/",
                "https://www.nitrc.org/frs/download.php/7729/",
                "https://www.nitrc.org/frs/download.php/7731/",
                "https://www.nitrc.org/frs/download.php/7726/",
                "https://www.nitrc.org/frs/download.php/7728/",
                "https://www.nitrc.org/frs/download.php/7727/",
            ]
        else:
            raise ValueError(
                f'Unknown mirror "{mirror!s}". '
                'Mirror must be "origin" or "nitrc"'
            )

    files = {
        "rsn20": "rsn20.nii.gz",
        "rsn10": "PNAS_Smith09_rsn10.nii.gz",
        "rsn70": "rsn70.nii.gz",
        "bm20": "bm20.nii.gz",
        "bm10": "PNAS_Smith09_bm10.nii.gz",
        "bm70": "bm70.nii.gz",
    }

    if isinstance(url, str):
        url = [url] * len(files)

    dataset_name = "smith_2009"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    fdescr = get_dataset_descr(dataset_name)

    if dimension:
        key = f"{'rsn' if resting else 'bm'}{dimension}"
        key_index = list(files).index(key)

        file = [(files[key], url[key_index] + files[key], {})]
        data = fetch_files(data_dir, file, resume=resume, verbose=verbose)

        return Atlas(
            maps=data[0],
            description=fdescr,
            atlas_type=atlas_type,
        )

    warnings.warn(
        category=DeprecationWarning,
        message=(
            deprecation_message.format(version="0.13")
            + (
                "To suppress this warning, "
                "Please use the parameters 'dimension' and 'resting' "
                "to specify the exact atlas image you want."
            )
        ),
    )

    keys = list(files.keys())
    files = [(f, u + f, {}) for f, u in zip(files.values(), url)]
    files_ = fetch_files(data_dir, files, resume=resume, verbose=verbose)
    params = dict(zip(keys, files_))

    params["description"] = fdescr
    params["atlas_type"] = atlas_type

    return Bunch(**params)


@fill_doc
def fetch_atlas_yeo_2011(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Yeo 2011 :term:`parcellation`.

    This function retrieves the so-called yeo
    :term:`deterministic atlases<Deterministic atlas>`. The provided images
    are in MNI152 space and have shapes equal to ``(256, 256, 256, 1)``.
    They contain consecutive integers values from 0 (background) to either
    7 or 17 depending on the atlas version considered.

    For more information on this dataset's structure,
    see :footcite:t:`CorticalParcellation_Yeo2011`,
    and :footcite:t:`Yeo2011`.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, keys are:

        - 'thin_7': :obj:`str`
            Path to nifti file containing the
            7 regions :term:`parcellation` fitted to thin template cortex
            segmentations.
            The image contains integer values which can be
            interpreted as the indices in ``colors_7``.

        - 'thick_7': :obj:`str`
            Path to nifti file containing the
            7 region :term:`parcellation` fitted to thick template cortex
            segmentations.
            The image contains integer values which can be
            interpreted as the indices in ``colors_7``.

        - 'thin_17': :obj:`str`
            Path to nifti file containing the
            17 region :term:`parcellation` fitted to thin template cortex
            segmentations.
            The image contains integer values which can be
            interpreted as the indices in ``colors_17``.

        - 'thick_17': :obj:`str`
            Path to nifti file containing the
            17 region :term:`parcellation` fitted to thick template cortex
            segmentations.
            The image contains integer values which can be
            interpreted as the indices in ``colors_17``.

        - 'colors_7': :obj:`str`
            Path to colormaps text file for
            7 region :term:`parcellation`.
            This file maps :term:`voxel` integer
            values from ``data.thin_7`` and ``data.tick_7`` to network names.

        - 'colors_17': :obj:`str`
            Path to colormaps text file for
            17 region :term:`parcellation`.
            This file maps :term:`voxel` integer
            values from ``data.thin_17`` and ``data.tick_17``
            to network names.

        - 'anat': :obj:`str`
            Path to nifti file containing the anatomy image.

        - %(description)s

        - %(template)s

        - %(atlas_type)s

    References
    ----------
    .. footbibliography::

    Notes
    -----
    License: unknown.

    """
    atlas_type = "deterministic"

    if url is None:
        url = (
            "ftp://surfer.nmr.mgh.harvard.edu/pub/data/"
            "Yeo_JNeurophysiol11_MNI152.zip"
        )
    opts = {"uncompress": True}

    dataset_name = "yeo_2011"
    keys = (
        "thin_7",
        "thick_7",
        "thin_17",
        "thick_17",
        "colors_7",
        "colors_17",
        "anat",
    )
    basenames = (
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_7Networks_ColorLUT.txt",
        "Yeo2011_17Networks_ColorLUT.txt",
        "FSL_MNI152_FreeSurferConformed_1mm.nii.gz",
    )

    filenames = [
        (Path("Yeo_JNeurophysiol11_MNI152", f), url, opts) for f in basenames
    ]

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    sub_files = fetch_files(
        data_dir, filenames, resume=resume, verbose=verbose
    )

    fdescr = get_dataset_descr(dataset_name)

    params = dict(
        [
            ("description", fdescr),
            ("atlas_type", atlas_type),
            *list(zip(keys, sub_files)),
        ]
    )

    lut = pd.read_csv(
        params["colors_7"],
        sep="\\s+",
        names=["index", "name", "r", "g", "b", "fs"],
        header=0,
    )
    params["lut_7"] = _update_lut_freesurder(lut)

    lut = pd.read_csv(
        params["colors_17"],
        sep="\\s+",
        names=["index", "name", "r", "g", "b", "fs"],
        header=0,
    )
    params["lut_17"] = _update_lut_freesurder(lut)

    _check_look_up_table(params["lut_7"], params["thin_7"])
    _check_look_up_table(params["lut_7"], params["thick_7"])
    _check_look_up_table(params["lut_17"], params["thin_17"])
    _check_look_up_table(params["lut_17"], params["thick_17"])

    return Bunch(**params)


def _update_lut_freesurder(lut):
    """Update LUT formatted for Freesurfer."""
    lut = pd.concat(
        [
            pd.DataFrame([[0, "Background", 0, 0, 0, 0]], columns=lut.columns),
            lut,
        ],
        ignore_index=True,
    )
    lut["color"] = "#" + rgb_to_hex_lookup(lut.r, lut.g, lut.b).astype(str)
    lut = lut.drop(["r", "g", "b", "fs"], axis=1)
    return lut


@fill_doc
def fetch_atlas_aal(
    version="SPM12", data_dir=None, url=None, resume=True, verbose=1
):
    """Download and returns the AAL template for :term:`SPM` 12.

    This :term:`Deterministic atlas` is the result of an automated anatomical
    parcellation of the spatially normalized single-subject high-resolution
    T1 volume provided by the Montreal Neurological Institute (:term:`MNI`)
    (D. L. Collins et al., 1998, Trans. Med. Imag. 17, 463-468, PubMed).

    For more information on this dataset's structure,
    see :footcite:t:`AAL_atlas`,
    and :footcite:t:`Tzourio-Mazoyer2002`.

    .. warning::

        The integers in the map image (data.maps) that define the parcellation
        are not always consecutive, as is usually the case in Nilearn, and
        should not be interpreted as indices for the list of label names.
        In addition, the region IDs are provided as strings, so it is necessary
        to cast them to integers when indexing.
        For more information, refer to the fetcher's description:

        .. code-block:: python

            from nilearn.datasets import fetch_atlas_aal

            atlas = fetch_atlas_aal()
            print(atlas.description)

    Parameters
    ----------
    version : {'3v2', 'SPM12', 'SPM5', 'SPM8'}, default='SPM12'
        The version of the AAL atlas. Must be 'SPM5', 'SPM8', 'SPM12', or '3v2'
        for the latest SPM12 version of AAL3 software.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, keys are:

        - 'maps': :obj:`str`
            Path to nifti file containing the regions.
            The image has shape ``(91, 109, 91)`` and contains
            117 unique integer values defining the parcellation in version
            SPM 5, 8 and 12, and 167 unique integer values defining the
            parcellation in version 3v2. Please refer to the main description
            to see how to link labels to regions IDs.

        - %(labels)s
            There are 117 names in version SPM 5, 8, and 12,
            and 167 names in version 3v2.
            Please refer to the main description
            to see how to link labels to regions IDs.

        - 'indices': :obj:`list` of :obj:`str`
            Indices mapping 'labels'
            to values in the 'maps' image.
            This list has 117 elements in
            version SPM 5, 8 and 12, and 167 elements in version 3v2.
            Since the values in the 'maps' image do not correspond to
            indices in ``labels``, but rather to values in ``indices``, the
            location of a label in the ``labels`` list does not necessary
            match the associated value in the image.
            Use the ``indices``
            list to identify the appropriate image value for a given label
            (See main description above).

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s


    Warns
    -----
    DeprecationWarning
        Starting in version 0.13, the default fetched mask will be AAL 3v2.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    License: unknown.

    """
    atlas_type = "deterministic"

    versions = ["SPM5", "SPM8", "SPM12", "3v2"]
    if version not in versions:
        raise ValueError(
            f"The version of AAL requested '{version}' does not exist."
            f"Please choose one among {versions}."
        )

    dataset_name = f"aal_{version}"
    opts = {"uncompress": True}

    if url is None:
        base_url = "https://www.gin.cnrs.fr/"
        if version == "SPM12":
            url = f"{base_url}AAL_files/aal_for_SPM12.tar.gz"
            basenames = ("AAL.nii", "AAL.xml")
            filenames = [
                (Path("aal", "atlas", f), url, opts) for f in basenames
            ]
            message = (
                "Starting in version 0.13, the default fetched mask will be"
                "AAL 3v2 instead."
            )
            warnings.warn(message, DeprecationWarning)

        elif version == "3v2":
            url = f"{base_url}wp-content/uploads/AAL3v2_for_SPM12.tar.gz"
            basenames = ("AAL3v1.nii", "AAL3v1.xml")
            filenames = [(Path("AAL3", f), url, opts) for f in basenames]
        else:
            url = f"{base_url}wp-content/uploads/aal_for_{version}.zip"
            basenames = ("ROI_MNI_V4.nii", "ROI_MNI_V4.txt")
            filenames = [
                (Path(f"aal_for_{version}", f), url, opts) for f in basenames
            ]

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    atlas_img, labels_file = fetch_files(
        data_dir, filenames, resume=resume, verbose=verbose
    )
    fdescr = get_dataset_descr("aal")
    labels = ["Background"]
    indices = ["0"]
    if version in ("SPM12", "3v2"):
        xml_tree = ElementTree.parse(labels_file)
        root = xml_tree.getroot()
        for label in root.iter("label"):
            indices.append(label.find("index").text)
            labels.append(label.find("name").text)
    else:
        with Path(labels_file).open() as fp:
            for line in fp:
                _, label, index = line.strip().split("\t")
                indices.append(index)
                labels.append(label)
        fdescr = fdescr.replace("SPM 12", version)

    return Atlas(
        maps=atlas_img,
        labels=labels,
        description=fdescr,
        lut=_generate_atlas_look_up_table(
            "fetch_atlas_aal", index=[int(x) for x in indices], name=labels
        ),
        atlas_type=atlas_type,
        indices=indices,
    )


@fill_doc
def fetch_atlas_basc_multiscale_2015(
    data_dir=None,
    url=None,
    resume=True,
    verbose=1,
    resolution=None,
    version="sym",
):
    """Download and load multiscale functional brain parcellations.

    This :term:`Deterministic atlas` includes group brain parcellations
    generated from resting-state
    :term:`functional magnetic resonance images<fMRI>` from about 200 young
    healthy subjects.

    Multiple resolutions (number of networks) are available, among
    7, 12, 20, 36, 64, 122, 197, 325, 444. The brain parcellations
    have been generated using a method called bootstrap analysis of
    stable clusters called as BASC :footcite:t:`Bellec2010`,
    and the resolutions have been selected using a data-driven method
    called MSTEPS :footcite:t:`Bellec2013`.

    Note that two versions of the template are available, 'sym' or 'asym'.
    The 'asym' type contains brain images that have been registered in the
    asymmetric version of the :term:`MNI` brain template (reflecting that
    the brain is asymmetric), while the 'sym' type contains images registered
    in the symmetric version of the :term:`MNI` template.
    The symmetric template has been forced to be symmetric anatomically, and
    is therefore ideally suited to study homotopic functional connections in
    :term:`fMRI`: finding homotopic regions simply consists of flipping the
    x-axis of the template.

    .. versionadded:: 0.2.3

    Parameters
    ----------
    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    resolution : :obj:`int`, default=None
        Number of networks in the dictionary.
        Valid resolutions  available are
        {7, 12, 20, 36, 64, 122, 197, 325, 444}

    version : {'sym', 'asym'}, default='sym'
        Available versions are 'sym' or 'asym'.
        By default all scales of brain parcellations of version 'sym'
        will be returned.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, Keys are:

        - "scale007", "scale012", "scale020", "scale036", "scale064", \
          "scale122", "scale197", "scale325", "scale444": :obj:`str`
            Path to Nifti file of various scales of brain parcellations.
            Images have shape ``(53, 64, 52)`` and contain consecutive integer
            values from 0 to the selected number of networks (scale).

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s

    Warns
    -----
    DeprecationWarning
        If a resolution input is provided, the current behavior
        (returning multiple maps) is deprecated.
        Starting in version 0.13, one map will be returned in a 'maps' dict key
        depending on the resolution and version value.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    For more information on this dataset's structure, see
    https://figshare.com/articles/dataset/Group_multiscale_functional_template_generated_with_BASC_on_the_Cambridge_sample/1285615

    """
    atlas_type = "deterministic"

    versions = ["sym", "asym"]
    if version not in versions:
        raise ValueError(
            f"The version of Brain parcellations requested '{version}' "
            "does not exist. "
            f"Please choose one among them {versions}."
        )

    file_number = "1861819" if version == "sym" else "1861820"
    url = f"https://ndownloader.figshare.com/files/{file_number}"

    opts = {"uncompress": True}

    keys = [
        "scale007",
        "scale012",
        "scale020",
        "scale036",
        "scale064",
        "scale122",
        "scale197",
        "scale325",
        "scale444",
    ]

    dataset_name = "basc_multiscale_2015"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    folder_name = Path(f"template_cambridge_basc_multiscale_nii_{version}")
    fdescr = get_dataset_descr(dataset_name)

    if resolution:
        basename = (
            "template_cambridge_basc_multiscale_"
            + version
            + f"_scale{resolution:03}"
            + ".nii.gz"
        )

        filename = [(folder_name / basename, url, opts)]

        data = fetch_files(data_dir, filename, resume=resume, verbose=verbose)

        labels = ["Background"] + [str(x) for x in range(1, resolution + 1)]

        return Atlas(
            maps=data[0],
            labels=labels,
            description=fdescr,
            lut=_generate_atlas_look_up_table(
                "fetch_atlas_basc_multiscale_2015", name=labels
            ),
            atlas_type=atlas_type,
        )

    warnings.warn(
        category=DeprecationWarning,
        message=(
            deprecation_message.format(version="0.13")
            + (
                "To suppress this warning, "
                "Please use the parameters 'resolution' and 'version' "
                "to specify the exact atlas image you want."
            )
        ),
    )

    basenames = [
        "template_cambridge_basc_multiscale_" + version + "_" + key + ".nii.gz"
        for key in keys
    ]
    filenames = [(folder_name / basename, url, opts) for basename in basenames]
    data = fetch_files(data_dir, filenames, resume=resume, verbose=verbose)

    params = dict(zip(keys, data))
    params["description"] = fdescr
    params["atlas_type"] = atlas_type

    return Bunch(**params)


@fill_doc
def fetch_coords_dosenbach_2010(ordered_regions=True):
    """Load the Dosenbach et al 160 ROIs.

    These ROIs cover much of the cerebral cortex
    and cerebellum and are assigned to 6 networks.

    See :footcite:t:`Dosenbach2010`.

    Parameters
    ----------
    ordered_regions : :obj:`bool`, default=True
        ROIs from same networks are grouped together and ordered with respect
        to their names and their locations (anterior to posterior).

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'rois':  :class:`pandas.DataFrame` with the coordinates
          of the 160 ROIs in :term:`MNI` space.

        - %(labels)s

        - 'networks': :class:`numpy.ndarray` of :obj:`str`, list of network
          names for the 160 ROI.

        - %(description)s

    References
    ----------
    .. footbibliography::

    """
    dataset_name = "dosenbach_2010"
    fdescr = get_dataset_descr(dataset_name)
    csv = PACKAGE_DIRECTORY / "data" / "dosenbach_2010.csv"
    out_csv = pd.read_csv(csv)

    if ordered_regions:
        out_csv = out_csv.sort_values(by=["network", "name", "y"])

    # We add the ROI number to its name, since names are not unique
    names = out_csv["name"]
    numbers = out_csv["number"]
    labels = [f"{name} {number}" for (name, number) in zip(names, numbers)]
    params = {
        "rois": out_csv[["x", "y", "z"]],
        "labels": labels,
        "networks": out_csv["network"],
        "description": fdescr,
    }

    return Bunch(**params)


@fill_doc
def fetch_coords_seitzman_2018(ordered_regions=True):
    """Load the Seitzman et al. 300 ROIs.

    These ROIs cover cortical, subcortical and cerebellar regions and are
    assigned to one of 13 networks (Auditory, CinguloOpercular, DefaultMode,
    DorsalAttention, FrontoParietal, MedialTemporalLobe, ParietoMedial,
    Reward, Salience, SomatomotorDorsal, SomatomotorLateral, VentralAttention,
    Visual) and have a regional label (cortexL, cortexR, cerebellum, thalamus,
    hippocampus, basalGanglia, amygdala, cortexMid).

    See :footcite:t:`Seitzman2020`.

    .. versionadded:: 0.5.1

    Parameters
    ----------
    ordered_regions : :obj:`bool`, default=True
        ROIs from same networks are grouped together and ordered with respect
        to their locations (anterior to posterior).

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'rois': :class:`pandas.DataFrame` with the coordinates
          of the 300 ROIs in :term:`MNI` space.

        - 'radius': :class:`numpy.ndarray` of :obj:`int`
            Radius of each ROI in mm.

        - 'networks': :class:`numpy.ndarray` of :obj:`str`
            Names of the corresponding network for each ROI.

        - 'regions': :class:`numpy.ndarray` of :obj:`str`
            Names of the regions.

        - %(description)s

    References
    ----------
    .. footbibliography::

    """
    dataset_name = "seitzman_2018"
    fdescr = get_dataset_descr(dataset_name)
    roi_file = (
        PACKAGE_DIRECTORY
        / "data"
        / "seitzman_2018_ROIs_300inVol_MNI_allInfo.txt"
    )
    anatomical_file = (
        PACKAGE_DIRECTORY / "data" / "seitzman_2018_ROIs_anatomicalLabels.txt"
    )

    rois = pd.read_csv(roi_file, delimiter=" ")
    rois = rois.rename(columns={"netName": "network", "radius(mm)": "radius"})

    # get integer regional labels and convert to text labels with mapping
    # from header line
    with anatomical_file.open() as fi:
        header = fi.readline()
    region_mapping = {}
    for r in header.strip().split(","):
        i, region = r.split("=")
        region_mapping[int(i)] = region

    anatomical = np.genfromtxt(anatomical_file, skip_header=1, encoding=None)
    anatomical_names = np.array([region_mapping[a] for a in anatomical])

    rois = pd.concat([rois, pd.DataFrame(anatomical_names)], axis=1)
    rois.columns = [*rois.columns[:-1], "region"]

    if ordered_regions:
        rois = rois.sort_values(by=["network", "y"])

    params = {
        "rois": rois[["x", "y", "z"]],
        "radius": np.array(rois["radius"]),
        "networks": np.array(rois["network"]),
        "regions": np.array(rois["region"]),
        "description": fdescr,
    }

    return Bunch(**params)


@fill_doc
def fetch_atlas_allen_2011(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Allen and MIALAB :term:`ICA` \
    :term:`Probabilistic atlas` (dated 2011).

    See :footcite:t:`Allen2011`.

    The provided images are in MNI152 space.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, keys are:

        - 'maps': :obj:`str`
            Path to nifti file containing the
            T-maps of all 75 unthresholded components.
            The image has shape ``(53, 63, 46, 75)``.

        - 'rsn28': :obj:`str`
            Path to nifti file containing the
            T-maps of 28 RSNs included in :footcite:t:`Allen2011`.
            The image has shape ``(53, 63, 46, 28)``.

        - 'networks': :obj:`list` of :obj:`list` of :obj:`str`
            List containing the names for the 28 RSNs.

        - 'rsn_indices': :obj:`list` of :obj:`tuple`, each tuple is a \
          (:obj:`str`, :obj:`list` of :`int`).
            This maps the network names to the map indices.
            For example, the map indices for the 'Visual' network
            can be obtained:

            .. code-block:: python

                # Should return [46, 64, 67, 48, 39, 59]
                dict(data.rsn_indices)["Visual"]

        - 'comps': :obj:`str`
            Path to nifti file containing the aggregate :term:`ICA` components.

        - %(description)s

        - %(atlas_type)s

        - %(template)s

    References
    ----------
    .. footbibliography::

    Notes
    -----
    License: unknown

    See https://trendscenter.org/data/ for more information
    on this dataset.

    """
    atlas_type = "probabilistic"

    if url is None:
        url = "https://osf.io/hrcku/download"

    dataset_name = "allen_rsn_2011"
    keys = ("maps", "rsn28", "comps")

    opts = {"uncompress": True}
    files = [
        "ALL_HC_unthresholded_tmaps.nii.gz",
        "RSN_HC_unthresholded_tmaps.nii.gz",
        "rest_hcp_agg__component_ica_.nii.gz",
    ]

    labels = [
        ("Basal Ganglia", [21]),
        ("Auditory", [17]),
        ("Sensorimotor", [7, 23, 24, 38, 56, 29]),
        ("Visual", [46, 64, 67, 48, 39, 59]),
        ("Default-Mode", [50, 53, 25, 68]),
        ("Attentional", [34, 60, 52, 72, 71, 55]),
        ("Frontal", [42, 20, 47, 49]),
    ]

    networks = [[name] * len(idxs) for name, idxs in labels]

    filenames = [(Path("allen_rsn_2011", f), url, opts) for f in files]

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    sub_files = fetch_files(
        data_dir, filenames, resume=resume, verbose=verbose
    )

    fdescr = get_dataset_descr(dataset_name)

    params = [
        ("description", fdescr),
        ("atlas_type", atlas_type),
        ("rsn_indices", labels),
        ("networks", networks),
        ("template", "volume"),
        *list(zip(keys, sub_files)),
    ]
    return Bunch(**dict(params))


@fill_doc
def fetch_atlas_surf_destrieux(
    data_dir=None, url=None, resume=True, verbose=1
):
    """Download and load Destrieux et al, 2010 cortical \
    :term:`Deterministic atlas`.

    See :footcite:t:`Destrieux2010`.

    This atlas returns 76 labels per hemisphere based on sulco-gryal patterns
    as distributed with Freesurfer in fsaverage5 surface space.

    .. versionadded:: 0.3

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - %(labels)s

        - 'map_left': :class:`numpy.ndarray` of :obj:`int`
            Maps each vertex on the left hemisphere
            of the fsaverage5 surface to its index
            into the list of label name.

        - 'map_right': :class:`numpy.ndarray` of :obj:`int`
            Maps each :term:`vertex` on the right hemisphere
            of the fsaverage5 surface to its index
            into the list of label name.

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage

    References
    ----------
    .. footbibliography::

    """
    atlas_type = "deterministic"

    if url is None:
        url = "https://www.nitrc.org/frs/download.php/"

    dataset_name = "destrieux_surface"
    fdescr = get_dataset_descr(dataset_name)
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    # Download annot files, fsaverage surfaces and sulcal information
    annot_file = "%s.aparc.a2009s.annot"
    annot_url = url + "%i/%s.aparc.a2009s.annot"
    annot_nids = {"lh annot": 9343, "rh annot": 9342}

    annots = []
    for hemi in [("lh", "left"), ("rh", "right")]:
        annot = fetch_files(
            data_dir,
            [
                (
                    annot_file % (hemi[1]),
                    annot_url % (annot_nids[f"{hemi[0]} annot"], hemi[0]),
                    {"move": annot_file % (hemi[1])},
                )
            ],
            resume=resume,
            verbose=verbose,
        )[0]
        annots.append(annot)

    annot_left = freesurfer.read_annot(annots[0])
    annot_right = freesurfer.read_annot(annots[1])

    labels = [x.decode("utf-8") for x in annot_left[2]]
    lut = _generate_atlas_look_up_table(
        "fetch_atlas_surf_destrieux", name=labels
    )
    _check_look_up_table(lut=lut, atlas=annot_left[0])
    _check_look_up_table(lut=lut, atlas=annot_right[0])

    return Bunch(
        labels=labels,
        map_left=annot_left[0],
        map_right=annot_right[0],
        description=fdescr,
        lut=lut,
        atlas_type=atlas_type,
        template="fsaverage",
    )


def _separate_talairach_levels(atlas_img, labels, output_dir, verbose):
    """Separate the multiple annotation levels in talairach raw atlas.

    The Talairach atlas has five levels of annotation: hemisphere, lobe, gyrus,
    tissue, brodmann area. They are mixed up in the original atlas: each label
    in the atlas corresponds to a 5-tuple containing, for each of these levels,
    a value or the string '*' (meaning undefined, background).

    This function disentangles the levels, and stores each in a separate image.

    The label '*' is replaced by 'Background' for clarity.
    """
    logger.log(
        f"Separating talairach atlas levels: {_TALAIRACH_LEVELS}",
        verbose=verbose,
        stack_level=3,
    )
    for level_name, old_level_labels in zip(
        _TALAIRACH_LEVELS, np.asarray(labels).T
    ):
        logger.log(level_name, verbose=verbose, stack_level=3)
        # level with most regions, ba, has 72 regions
        level_data = np.zeros(atlas_img.shape, dtype="uint8")
        level_labels = {"*": 0}
        for region_nb, region_name in enumerate(old_level_labels):
            level_labels.setdefault(region_name, len(level_labels))
            level_data[get_img_data(atlas_img) == region_nb] = level_labels[
                region_name
            ]
        new_img_like(atlas_img, level_data).to_filename(
            output_dir / f"{level_name}.nii.gz"
        )

        level_labels = list(level_labels.keys())
        # rename '*' -> 'Background'
        level_labels[0] = "Background"
        (output_dir / f"{level_name}-labels.json").write_text(
            json.dumps(level_labels), "utf-8"
        )


def _download_talairach(talairach_dir, verbose):
    """Download the Talairach atlas and separate the different levels."""
    atlas_url = "https://www.talairach.org/talairach.nii"
    temp_dir = mkdtemp()
    try:
        temp_file = fetch_files(
            temp_dir, [("talairach.nii", atlas_url, {})], verbose=verbose
        )[0]
        atlas_img = load(temp_file, mmap=False)
        atlas_img = check_niimg(atlas_img)
    finally:
        shutil.rmtree(temp_dir)
    labels_text = atlas_img.header.extensions[0].get_content()
    multi_labels = labels_text.strip().decode("utf-8").split("\n")
    labels = [lab.split(".") for lab in multi_labels]
    _separate_talairach_levels(
        atlas_img, labels, talairach_dir, verbose=verbose
    )


@fill_doc
def fetch_atlas_talairach(level_name, data_dir=None, verbose=1):
    """Download the Talairach :term:`Deterministic atlas`.

    For more information, see :footcite:t:`talairach_atlas`,
    :footcite:t:`Lancaster2000`,
    and :footcite:t:`Lancaster1997`.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    level_name : {'hemisphere', 'lobe', 'gyrus', 'tissue', 'ba'}
        Which level of the atlas to use: the hemisphere, the lobe, the gyrus,
        the tissue type or the Brodmann area.
    %(data_dir)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'maps': 3D :class:`~nibabel.nifti1.Nifti1Image`
            The image has
            shape ``(141, 172, 110)`` and contains consecutive integer
            values from 0 to the number of regions, which are indices
            in the list of labels.

        - %(labels)s

            The list starts with 'Background' (region ID 0 in the image).

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s

    References
    ----------
    .. footbibliography::

    """
    atlas_type = "deterministic"

    if level_name not in _TALAIRACH_LEVELS:
        raise ValueError(f'"level_name" should be one of {_TALAIRACH_LEVELS}')
    talairach_dir = get_dataset_dir(
        "talairach_atlas", data_dir=data_dir, verbose=verbose
    )

    img_file = talairach_dir / f"{level_name}.nii.gz"
    labels_file = talairach_dir / f"{level_name}-labels.json"

    if not img_file.is_file() or not labels_file.is_file():
        _download_talairach(talairach_dir, verbose=verbose)

    atlas_img = check_niimg(img_file)
    labels = json.loads(labels_file.read_text("utf-8"))

    return Atlas(
        maps=atlas_img,
        labels=labels,
        description=get_dataset_descr("talairach_atlas").format(level_name),
        lut=_generate_atlas_look_up_table(
            "fetch_atlas_talairach", name=labels
        ),
        atlas_type=atlas_type,
        template="Talairach",
    )


@rename_parameters(
    replacement_params={"version": "atlas_type"}, end_version="0.13.1"
)
@fill_doc
def fetch_atlas_pauli_2017(
    atlas_type="probabilistic", data_dir=None, verbose=1
):
    """Download the Pauli et al. (2017) atlas.

    This atlas has 12 subcortical nodes in total. See
    :footcite:t:`pauli_atlas` and :footcite:t:`Pauli2018`.

    Parameters
    ----------
    atlas_type : {'probabilistic', 'deterministic'}, default='probabilistic'
        Which type of the atlas should be download. This can be
        'probabilistic' for the :term:`Probabilistic atlas`, or 'deterministic'
        for the :term:`Deterministic atlas`.
    %(data_dir)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'maps': :obj:`str`,
            path to nifti file containing the
            :class:`~nibabel.nifti1.Nifti1Image`.
            If ``atlas_type='probabilistic'``,
            the image shape is ``(193, 229, 193, 16)``.
            If ``atlas_type='deterministic'`` the image shape is
            ``(198, 263, 212)``, and values are indices in the list of labels
            (integers from 0 to 16).

        - %(labels)s
            The list contains values for both
            :term:`probabilitic<Probabilistic atlas>` and
            :term:`deterministic<Deterministic atlas>` types.

        - %(description)s

        - %(lut)s
            Only when atlas_type="deterministic"

        - %(template)s

        - %(atlas_type)s


    Warns
    -----
    DeprecationWarning
        The possible values for atlas_type are currently 'prob' and 'det'. From
    release 0.13.0 onwards, atlas_type will accept only 'probabilistic' or
    'deterministic' as value.

    References
    ----------
    .. footbibliography::

    """
    # TODO: remove this part after release 0.13.0
    if atlas_type in ("prob", "det"):
        atlas_type_values = (
            "The possible values for atlas_type are currently 'prob' and"
            " 'det'. From release 0.13.0 onwards, atlas_type will accept only"
            " 'probabilistic' or 'deterministic' as value."
        )
        warnings.warn(
            category=DeprecationWarning,
            message=atlas_type_values,
            stacklevel=2,
        )
        atlas_type = (
            "probabilistic" if atlas_type == "prob" else "deterministic"
        )

    if atlas_type not in {"probabilistic", "deterministic"}:
        raise NotImplementedError(
            f"{atlas_type} is not a valid type for the Pauli atlas"
        )

    url_maps = "https://osf.io/w8zq2/download"
    filename = "pauli_2017_prob.nii.gz"
    if atlas_type == "deterministic":
        url_maps = "https://osf.io/5mqfx/download"
        filename = "pauli_2017_det.nii.gz"

    url_labels = "https://osf.io/6qrcb/download"
    dataset_name = "pauli_2017"

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    files = [
        (filename, url_maps, {"move": filename}),
        ("labels.txt", url_labels, {"move": "labels.txt"}),
    ]
    atlas_file, labels = fetch_files(data_dir, files)

    labels = np.loadtxt(labels, dtype=str)[:, 1].tolist()

    return Atlas(
        maps=atlas_file,
        labels=labels,
        description=get_dataset_descr(dataset_name),
        lut=_generate_atlas_look_up_table(
            "fetch_atlas_pauli_2017", name=labels
        ),
        atlas_type=atlas_type,
    )


@fill_doc
def fetch_atlas_schaefer_2018(
    n_rois=400,
    yeo_networks=7,
    resolution_mm=1,
    data_dir=None,
    base_url=None,
    resume=True,
    verbose=1,
):
    """Download and return file names for the Schaefer 2018 parcellation.

    .. versionadded:: 0.5.1

    This function returns a :term:`Deterministic atlas`, and the provided
    images are in MNI152 space.

    For more information on this dataset, see :footcite:t:`schaefer_atlas`,
    :footcite:t:`Schaefer2017`,
    and :footcite:t:`Yeo2011`.

    Parameters
    ----------
    n_rois : {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}, default=400
        Number of regions of interest.

    yeo_networks : {7, 17}, default=7
        ROI annotation according to yeo networks.

    resolution_mm : {1, 2}, default=1mm
        Spatial resolution of atlas image in mm.
    %(data_dir)s
    base_url : :obj:`str`,  default=None
        Base URL of files to download (``None`` results in
        default ``base_url``).
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - 'maps': :obj:`str`, path to nifti file containing the
            3D :class:`~nibabel.nifti1.Nifti1Image` (its shape is
            ``(182, 218, 182)``).
            The values are consecutive integers
            between 0 and ``n_rois`` which can be interpreted as indices
            in the list of labels.

        - %(labels)s

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s

    References
    ----------
    .. footbibliography::


    Notes
    -----
    Release v0.14.3 of the Schaefer 2018 parcellation is used by
    default. Versions prior to v0.14.3 are known to contain erroneous region
    label names. For more details, see
    https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/Updates/Update_20190916_README.md

    License: MIT.

    """
    atlas_type = "deterministic"

    valid_n_rois = list(range(100, 1100, 100))
    valid_yeo_networks = [7, 17]
    valid_resolution_mm = [1, 2]
    if n_rois not in valid_n_rois:
        raise ValueError(
            f"Requested n_rois={n_rois} not available. "
            f"Valid options: {valid_n_rois}"
        )
    if yeo_networks not in valid_yeo_networks:
        raise ValueError(
            f"Requested yeo_networks={yeo_networks} not available. "
            f"Valid options: {valid_yeo_networks}"
        )
    if resolution_mm not in valid_resolution_mm:
        raise ValueError(
            f"Requested resolution_mm={resolution_mm} not available. "
            f"Valid options: {valid_resolution_mm}"
        )

    if base_url is None:
        base_url = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/"
            "v0.14.3-Update_Yeo2011_Schaefer2018_labelname/"
            "stable_projects/brain_parcellation/"
            "Schaefer2018_LocalGlobal/Parcellations/MNI/"
        )

    labels_file_template = "Schaefer2018_{}Parcels_{}Networks_order.txt"
    img_file_template = (
        "Schaefer2018_{}Parcels_{}Networks_order_FSLMNI152_{}mm.nii.gz"
    )
    files = [
        (f, base_url + f, {})
        for f in [
            labels_file_template.format(n_rois, yeo_networks),
            img_file_template.format(n_rois, yeo_networks, resolution_mm),
        ]
    ]

    dataset_name = "schaefer_2018"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    labels_file, atlas_file = fetch_files(
        data_dir, files, resume=resume, verbose=verbose
    )

    lut = pd.read_csv(
        labels_file,
        delimiter="\t",
        names=["index", "name", "r", "g", "b", "fs"],
    )
    lut = _update_lut_freesurder(lut)

    return Atlas(
        maps=atlas_file,
        labels=list(lut["name"]),
        description=get_dataset_descr(dataset_name),
        lut=lut,
        atlas_type=atlas_type,
    )


class Atlas(Bunch):
    """Sub class of Bunch to help standardize atlases.

    Parameters
    ----------
    maps : Niimg-like object or SurfaceImage object
        single image or list of images for that atlas

    description : str
        atlas description

    atlas_type: {"deterministic", "probabilistic"}

    labels: list of str
        labels for the atlas

    lut: pandas.DataFrame
        look up table for the atlas

    template: str
        name of the template used for the atlas
    """

    def __init__(
        self,
        maps,
        description,
        atlas_type,
        labels=None,
        lut=None,
        template=None,
        **kwargs,
    ):
        assert atlas_type in ["probabilistic", "deterministic"]

        # TODO: improve
        if template is None:
            template = "volume"

        if atlas_type == "probabilistic":
            super().__init__(
                maps=maps,
                labels=labels,
                description=description,
                atlas_type=atlas_type,
                template=template,
                **kwargs,
            )

            return None

        _check_look_up_table(lut=lut, atlas=maps)

        super().__init__(
            maps=maps,
            labels=lut.name.to_list(),
            description=description,
            lut=lut,
            atlas_type=atlas_type,
            template=template,
            **kwargs,
        )
