"""Downloading NeuroImaging datasets: atlas datasets."""

import json
import re
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from nibabel import freesurfer, load
from requests.exceptions import SSLError
from sklearn.utils import Bunch

from nilearn._utils import logger
from nilearn._utils.bids import (
    check_look_up_table,
    generate_atlas_look_up_table,
)
from nilearn._utils.docs import Description, fill_doc
from nilearn._utils.niimg import _get_data
from nilearn._utils.param_validation import (
    check_parameter_in_allowed,
    check_params,
)
from nilearn.datasets._utils import (
    PACKAGE_DIRECTORY,
    fetch_files,
    fetch_single_file,
    get_dataset_dir,
)
from nilearn.image import check_niimg, new_img_like, reorder_img
from nilearn.nilearn_typing import DataDir, Resume, Url, Verbose


class Atlas(Bunch):
    """Sub class of Bunch to help standardize atlases.

    Parameters
    ----------
    maps : Niimg-like object or SurfaceImage object
        single image or list of images for that atlas

    description : :obj:`str`
        atlas description

    atlas_type : {"deterministic", "probabilistic"}

    labels : :obj:`list` of str
        labels for the atlas

    lut : pandas.DataFrame
        look up table for the atlas

    template : :obj:`str`
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
        check_parameter_in_allowed(
            atlas_type, ["probabilistic", "deterministic"], "atlas_type"
        )

        # TODO: improve
        if template is None:
            template = "MNI?"

        if atlas_type == "probabilistic":
            if labels is None:
                super().__init__(
                    maps=maps,
                    description=description,
                    atlas_type=atlas_type,
                    template=template,
                    **kwargs,
                )
            else:
                super().__init__(
                    maps=maps,
                    labels=labels,
                    description=description,
                    atlas_type=atlas_type,
                    template=template,
                    **kwargs,
                )

            return None

        check_look_up_table(lut=lut, atlas=maps, verbose=1)

        super().__init__(
            maps=maps,
            labels=lut.name.to_list(),
            description=description,
            lut=lut,
            atlas_type=atlas_type,
            template=template,
            **kwargs,
        )


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


@fill_doc
def fetch_atlas_difumo(
    dimension: Literal[64, 128, 256, 512, 1024] = 64,
    resolution_mm: Literal[2, 3] = 2,
    data_dir: DataDir = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Fetch DiFuMo brain atlas.

    Dictionaries of Functional Modes, or “DiFuMo”, can serve as
    :term:`probabilistic atlases<Probabilistic atlas>` to extract
    functional signals with different dimensionalities (64, 128,
    256, 512, and 1024).

    For more information,
    see the :ref:`dataset description <difumo_atlas>`.

    .. nilearn_versionadded:: 0.7.1

    Notes
    -----
    %(fetcher_note)s

    Parameters
    ----------
    dimension : :obj:`int`, default=64
        Number of dimensions in the dictionary. Valid resolutions
        available are {64, 128, 256, 512, 1024}.

    resolution_mm : :obj:`int`, default=2
        The resolution of the atlas to fetch, in mm. Valid options
        available are {2, 3}.

    %(data_dir)s

    %(resume)s

    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(difumo_atlas_content)s

    """
    check_params(locals())
    atlas_type = "probabilistic"

    dic = {
        64: "pqu9r",
        128: "wjvd5",
        256: "3vrct",
        512: "9b76y",
        1024: "34792",
    }
    valid_dimensions = [64, 128, 256, 512, 1024]
    check_parameter_in_allowed(dimension, valid_dimensions, "dimension")
    valid_resolution_mm = [2, 3]
    check_parameter_in_allowed(
        resolution_mm, valid_resolution_mm, "resolution_mm"
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

    dataset_name = "difumo_atlas"

    dataset_dir = get_dataset_dir(
        dataset_name=dataset_name, data_dir=data_dir, verbose=verbose
    )

    # Download the zip file, first
    files_ = fetch_files(dataset_dir, files, verbose=verbose, resume=resume)
    labels = pd.read_csv(files_[0])
    labels = labels.rename(columns={c: c.lower() for c in labels.columns})

    # README
    readme_files = [
        ("README.md", "https://osf.io/4k9bf/download", {"move": "README.md"})
    ]
    if not (dataset_dir / "README.md").exists():
        fetch_files(dataset_dir, readme_files, verbose=verbose, resume=resume)

    return Atlas(
        maps=files_[1],
        labels=labels,
        description=Description.from_registry(dataset_name),
        atlas_type=atlas_type,
        template="MNI152NLin6Asym",
    )


@fill_doc
def fetch_atlas_craddock_2012(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
    homogeneity: Literal["spatial", "temporal", "random"] = "spatial",
    grp_mean: bool = True,
) -> Atlas:
    """Download and return file names \
       for the Craddock 2012 :term:`parcellation`.

    For more information,
    see the :ref:`dataset description <craddock_2012_atlas>`.

    Parameters
    ----------
    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    homogeneity : :obj:`str`,  default='spatial'
        The choice of the homogeneity ('spatial' or 'temporal' or 'random')

    grp_mean : :obj:`bool`, default=True
        The choice of the :term:`parcellation` (with group_mean or without)

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(craddock_2012_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())
    atlas_type = "probabilistic"

    if url is None:
        url = (
            "https://cluster_roi.projects.nitrc.org"
            "/Parcellations/craddock_2011_parcellations.tar.gz"
        )
    opts = {"uncompress": True}

    dataset_name = "craddock_2012_atlas"

    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    allowed_homogeneity = {"spatial", "temporal", "random"}
    check_parameter_in_allowed(homogeneity, allowed_homogeneity, "homogeneity")

    if homogeneity in ["spatial", "temporal"]:
        if grp_mean:
            filename = [(homogeneity[0] + "corr05_mean_all.nii.gz", url, opts)]
        else:
            filename = [
                (homogeneity[0] + "corr05_2level_all.nii.gz", url, opts)
            ]
    else:
        filename = [("random_all.nii.gz", url, opts)]
    data = fetch_files(data_dir, filename, resume=resume, verbose=verbose)

    return Atlas(
        maps=data[0],
        description=Description.from_registry(dataset_name),
        atlas_type=atlas_type,
        template="MNI152",
    )


@fill_doc
def fetch_atlas_destrieux_2009(
    lateralized: bool = True,
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Download and load the Destrieux cortical \
    :term:`deterministic atlas<Deterministic atlas>` (dated 2009).

    For more information,
    see the :ref:`dataset description <destrieux_2009_atlas>`.

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

        %(destrieux_2009_atlas_content)s

    Notes
    -----
    %(fetcher_note)s

    See Also
    --------
    nilearn.datasets.fetch_atlas_surf_destrieux
    """
    check_params(locals())

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

    dataset_name = "destrieux_2009_atlas"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    files_ = fetch_files(data_dir, files, resume=resume, verbose=verbose)

    labels = pd.read_csv(files_[0], index_col=0)

    return Atlas(
        maps=files_[1],
        labels=labels.name.to_list(),
        description=Description.from_registry(dataset_name),
        atlas_type=atlas_type,
        lut=pd.read_csv(files_[0]),
        template="fsaverage",
    )


@fill_doc
def fetch_atlas_harvard_oxford(
    atlas_name: str,
    data_dir: DataDir = None,
    symmetric_split: bool = False,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Load Harvard-Oxford parcellations from FSL.

    This function downloads Harvard Oxford atlas packaged from FSL 5.0
    and stores atlases in NILEARN_DATA folder in home directory.

    This function can also load Harvard Oxford atlas from your local directory
    specified by your FSL installed path given in `data_dir` argument.

    For more information,
    see the :ref:`dataset description <harvard_oxford_atlas>`.

    .. note::

        For atlases 'cort-prob-1mm', 'cort-prob-2mm', 'cortl-prob-1mm',
        'cortl-prob-2mm', 'sub-prob-1mm', and 'sub-prob-2mm', the function
        returns a :term:`Probabilistic atlas`, and the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 4D.
        For :term:`deterministic atlases<Deterministic atlas>`, the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 3D.

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
        Dictionary-like object, contains:

            %(harvard_oxford_atlas_content)s

        .. note::

            For some atlases, it can be the case that some regions are empty.
            In this case, no :term:`voxels<voxel>` in the map are assigned
            to these regions.
            So the number of unique values in the map
            can be strictly smaller
            than the number of region names in ``labels``.

    Notes
    -----
    %(fetcher_note)s

    See Also
    --------
    nilearn.datasets.fetch_atlas_juelich

    """
    check_params(locals())

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
    check_parameter_in_allowed(atlas_name, atlases, "atlas_name")

    atlas_type = "probabilistic" if "-prob-" in atlas_name else "deterministic"

    if atlas_type == "probabilistic" and symmetric_split:
        raise ValueError(
            "Region splitting not supported for probabilistic atlases"
        )
    (
        atlas_img,
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

    description = Description.from_registry("harvard_oxford_atlas")

    if not symmetric_split or is_lateralized:
        return Atlas(
            maps=atlas_niimg,
            labels=names,
            description=description,
            atlas_type=atlas_type,
            lut=generate_atlas_look_up_table(
                "fetch_atlas_harvard_oxford", name=names
            ),
            template="MNI152NLin6Asym",
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
        description=description,
        atlas_type=atlas_type,
        lut=generate_atlas_look_up_table(
            "fetch_atlas_harvard_oxford", name=new_names
        ),
        template="MNI152NLin6Asym",
    )


@fill_doc
def fetch_atlas_juelich(
    atlas_name: str,
    data_dir: DataDir = None,
    symmetric_split: bool = False,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Load Juelich parcellations from FSL.

    This function downloads Juelich atlas packaged from FSL 5.0
    and stores atlases in NILEARN_DATA folder in home directory.

    This function can also load Juelich atlas from your local directory
    specified by your FSL installed path given in `data_dir` argument.

    For more information,
    see the :ref:`dataset description <juelich_atlas>`.

    .. nilearn_versionadded:: 0.8.1

    .. note::

        For atlases 'prob-1mm', and 'prob-2mm', the function returns a
        :term:`Probabilistic atlas`, and the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 4D.
        For :term:`deterministic atlases<Deterministic atlas>`, the
        :class:`~nibabel.nifti1.Nifti1Image` returned is 3D.

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
        Dictionary-like object, contains:

            %(juelich_atlas_content)s

        .. note::

            For some atlases, it can be the case that some regions are empty.
            In this case, no :term:`voxels<voxel>` in the map are assigned
            to these regions.
            So the number of unique values in the map
            can be strictly smaller
            than the number of region names in ``labels``.

    Notes
    -----
    %(fetcher_note)s

    See Also
    --------
    nilearn.datasets.fetch_atlas_harvard_oxford

    """
    check_params(locals())

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
    check_parameter_in_allowed(atlas_name, atlases, "atlas_name")

    atlas_type = (
        "probabilistic" if atlas_name.startswith("prob-") else "deterministic"
    )

    if atlas_type == "probabilistic" and symmetric_split:
        raise ValueError(
            "Region splitting not supported for probabilistic atlases"
        )
    atlas_img, names, _ = _get_atlas_data_and_labels(
        "Juelich",
        atlas_name,
        data_dir=data_dir,
        resume=resume,
        verbose=verbose,
    )
    atlas_niimg = check_niimg(atlas_img)
    atlas_data = _get_data(atlas_niimg)

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
        description=Description.from_registry("juelich_atlas"),
        atlas_type=atlas_type,
        lut=generate_atlas_look_up_table(
            "fetch_atlas_juelich", name=list(new_names)
        ),
        template="?",
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
    check_parameter_in_allowed(
        atlas_source,
        ["Juelich", "HarvardOxford", "atlas_source"],
        "atlas_source",
    )
    if atlas_source == "Juelich":
        url = "https://www.nitrc.org/frs/download.php/12096/Juelich.tgz"
    elif atlas_source == "HarvardOxford":
        url = "https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz"

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
    atlas_img = reorder_img(atlas_file)
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
    return atlas_img, names, is_lateralized


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
    atlas_data = _get_data(atlas_niimg)
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
    for label, name in zip(labels[1:], names[1:], strict=False):
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
def fetch_atlas_msdl(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Download and load the MSDL brain :term:`Probabilistic atlas`.

    For more information,
    see the :ref:`dataset description <msdl_atlas>`.

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

        %(msdl_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())

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
        description=Description.from_registry(dataset_name),
        atlas_type=atlas_type,
        region_coords=csv_data[["x", "y", "z"]].to_numpy().tolist(),
        networks=net_names,
    )


@fill_doc
def fetch_coords_power_2011() -> Bunch[str, pd.DataFrame | str]:
    """Download and load the Power et al. brain atlas composed of 264 ROIs.

    For more information
    see the :ref:`dataset description <power_2011_atlas>`.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(power_2011_atlas_content)s

    """
    csv = PACKAGE_DIRECTORY / "data" / "power_2011.csv"
    rois = pd.read_csv(csv)
    rois = rois.rename(columns={c: c.lower() for c in rois.columns})
    params = {
        "rois": rois,
        "description": Description.from_registry("power_2011_atlas"),
        "template": "MNI?",
        "atlas_type": "deterministic",
    }
    return Bunch(**params)


@fill_doc
def fetch_atlas_smith_2009(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
    mirror: Literal["origin", "nitrc"] = "origin",
    dimension: Literal[10, 20, 70] = 10,
    resting: bool = True,
) -> Atlas:
    """Download and load the Smith :term:`ICA` and BrainMap \
    :term:`Probabilistic atlas` (2009).

    For more information
    see the :ref:`dataset description <smith_2009_atlas>`.

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

    dimension : :obj:`int`, default=10
        Number of dimensions in the dictionary. Valid dimension
        available are {10, 20, 70}.

    resting : :obj:`bool`, default=True
        Either to fetch the resting-:term:`fMRI` or BrainMap components

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(smith_2009_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())

    atlas_type = "probabilistic"

    files = {
        "rsn20": "rsn20.nii.gz",
        "rsn10": "PNAS_Smith09_rsn10.nii.gz",
        "rsn70": "rsn70.nii.gz",
        "bm20": "bm20.nii.gz",
        "bm10": "PNAS_Smith09_bm10.nii.gz",
        "bm70": "bm70.nii.gz",
    }

    if url is None:
        check_parameter_in_allowed(mirror, ["origin", "nitrc"], "mirror")
        if mirror == "origin":
            list_url = [
                "https://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/"
            ] * len(files)
        elif mirror == "nitrc":
            list_url = [
                "https://www.nitrc.org/frs/download.php/7730/",
                "https://www.nitrc.org/frs/download.php/7729/",
                "https://www.nitrc.org/frs/download.php/7731/",
                "https://www.nitrc.org/frs/download.php/7726/",
                "https://www.nitrc.org/frs/download.php/7728/",
                "https://www.nitrc.org/frs/download.php/7727/",
            ]
    elif isinstance(url, str):
        list_url = [url] * len(files)

    dataset_name = "smith_2009_atlas"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    fdescr = Description.from_registry(dataset_name)

    key = f"{'rsn' if resting else 'bm'}{dimension}"
    key_index = list(files).index(key)

    file: list[tuple[str, str, dict[str, str]]] = [
        (files[key], list_url[key_index] + files[key], {})
    ]
    data = fetch_files(data_dir, file, resume=resume, verbose=verbose)

    return Atlas(
        maps=data[0],
        description=fdescr,
        atlas_type=atlas_type,
    )


@fill_doc
def fetch_atlas_yeo_2011(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
    n_networks: Literal[7, 17] = 7,
    thickness: Literal["thin", "thick"] = "thick",
) -> Atlas:
    """Download and return file names for the Yeo 2011 :term:`parcellation`.

    For more information
    see the :ref:`dataset description <yeo_2011_atlas>`.

    Parameters
    ----------
    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    n_networks : {7, 17}, default = 7
        Specify the version of the atlas that is returned:

        - 7 networks parcellation,
        - 17 networks parcellation.

        .. nilearn_versionadded:: 0.12.0

        .. nilearn_versionchanged:: 0.13.0

          The default was changed to 7.

    thickness : {"thin", "thick"}, default = "thick"
        Specific the version of the atlas that is returned:

        - ``"thick"``: parcellation fitted to thick cortex segmentations,
        - ``"thin"``: parcellation fitted to thin cortex segmentations.

        .. nilearn_versionadded:: 0.12.0

        .. nilearn_versionchanged:: 0.13.0

          The default was changed to "thick".

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(yeo_2011_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())

    atlas_type = "deterministic"

    check_parameter_in_allowed(n_networks, (7, 17), "n_networks")
    check_parameter_in_allowed(thickness, ("thin", "thick"), "thickness")

    if url is None:
        url = (
            "ftp://surfer.nmr.mgh.harvard.edu/pub/data/"
            "Yeo_JNeurophysiol11_MNI152.zip"
        )
    opts = {"uncompress": True}

    dataset_name = "yeo_2011_atlas"
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

    fdescr = Description.from_registry(dataset_name)

    params = dict(
        [
            ("description", fdescr),
            ("atlas_type", atlas_type),
            *list(zip(keys, sub_files, strict=False)),
        ]
    )

    lut_file = params["colors_7"] if n_networks == 7 else params["colors_17"]
    lut = pd.read_csv(
        lut_file,
        sep="\\s+",
        names=["index", "name", "r", "g", "b", "fs"],
        header=0,
    )
    lut = _update_lut_freesurder(lut)

    maps = params[f"{thickness}_{n_networks}"]

    return Atlas(
        maps=maps,
        labels=lut.name.to_list(),
        description=fdescr,
        template="MNI152NLin6Asym",
        lut=lut,
        atlas_type=atlas_type,
        anat=params["anat"],
    )


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
    version: Literal["3v2", "SPM12", "SPM5", "SPM8"] = "3v2",
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Download and returns the AAL template for :term:`SPM` 12.

    For more information
    see the :ref:`dataset description <aal_atlas>`.

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
    version : {'3v2', 'SPM12', 'SPM5', 'SPM8'}, default='3v2'
        The version of the AAL atlas. Must be 'SPM5', 'SPM8', 'SPM12', or '3v2'
        for the latest SPM12 version of AAL3 software.

        .. nilearn_versionchanged:: 0.13.0

          The default was changed to '3v2'.

    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(aal_atlas_content)s

    Notes
    -----
    %(fetcher_note)s

    """
    check_params(locals())

    atlas_type = "deterministic"

    versions = ["SPM5", "SPM8", "SPM12", "3v2"]
    check_parameter_in_allowed(version, versions, "version")

    dataset_name = f"aal_{version}"
    opts = {"uncompress": True}

    backup_url = {
        "3v2": "https://osf.io/6jngh/download",
        "SPM12": "https://osf.io/s94qg/download",
        "SPM8": "https://osf.io/rkpeh/download",
        "SPM5": "https://osf.io/948y2/download",
    }

    if url is None:
        base_url = "https://www.gin.cnrs.fr/"
        if version == "SPM12":
            url = f"{base_url}AAL_files/aal_for_SPM12.tar.gz"
            basenames = ("AAL.nii", "AAL.xml")
            filenames = [
                (Path("aal", "atlas", f), url, opts) for f in basenames
            ]
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
    try:
        atlas_img, labels_file = fetch_files(
            data_dir, filenames, resume=resume, verbose=verbose
        )
    except SSLError:
        if version == "SPM12":
            filenames = [
                (Path("aal", "atlas", f), backup_url[version], opts)
                for f in basenames
            ]
        elif version == "3v2":
            filenames = [
                (Path("AAL3", f), backup_url[version], opts) for f in basenames
            ]
        else:
            filenames = [
                (Path(f"aal_for_{version}", f), backup_url[version], opts)
                for f in basenames
            ]
        atlas_img, labels_file = fetch_files(
            data_dir, filenames, resume=resume, verbose=verbose
        )

    labels = ["Background"]
    indices = ["0"]
    if version in ("SPM12", "3v2"):
        xml_tree = ElementTree.parse(labels_file)
        root = xml_tree.getroot()
        for lbl in root.iter("label"):
            if (
                (idx := lbl.find("index")) is None
                or (name := lbl.find("name")) is None
                or idx.text is None
                or name.text is None
            ):
                continue
            indices.append(idx.text)
            labels.append(name.text)
    else:
        with Path(labels_file).open(encoding="utf-8") as fp:
            for line in fp:
                _, label, index = line.strip().split("\t")
                indices.append(index)
                labels.append(label)

    return Atlas(
        maps=atlas_img,
        labels=labels,
        description=Description.from_registry("aal_atlas"),
        lut=generate_atlas_look_up_table(
            "fetch_atlas_aal",
            index=np.array([int(x) for x in indices]),
            name=labels,
        ),
        atlas_type=atlas_type,
        template="MNIColin27",
        indices=indices,
    )


@fill_doc
def fetch_atlas_basc_multiscale_2015(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
    resolution: Literal[7, 12, 20, 36, 64, 122, 197, 325, 444] = 7,
    version: Literal["sym", "asym"] = "sym",
) -> Atlas:
    """Download and load multiscale functional brain parcellations.

    This :term:`Deterministic atlas` includes group brain parcellations
    generated from resting-state
    :term:`functional magnetic resonance images<fMRI>` from about 200 young
    healthy subjects.

    For more information
    see the :ref:`dataset description <basc_multiscale_2015_atlas>`.

    .. nilearn_versionadded:: 0.2.3

    Parameters
    ----------
    %(data_dir)s

    %(url)s

    %(resume)s

    %(verbose)s

    resolution : :obj:`int`, default=7
        Number of networks in the dictionary.
        Valid resolutions available are
        {7, 12, 20, 36, 64, 122, 197, 325, 444}

        .. nilearn_versionchanged: 0.13.0

          Default changed to ``7``.

    version : {'sym', 'asym'}, default='sym'
        Available versions are 'sym' or 'asym'.
        By default all scales of brain parcellations of version 'sym'
        will be returned.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        - maps: :obj:`str`
            Path to Nifti file of the brain parcellation.
            Images have shape ``(53, 64, 52)`` and contain consecutive integer
            values from 0 to the selected number of networks (scale).

        - %(description)s

        - %(lut)s

        - %(template)s

        - %(atlas_type)s

    Notes
    -----
    %(fetcher_note)s

    """
    check_params(locals())

    atlas_type = "deterministic"

    versions = ["sym", "asym"]
    check_parameter_in_allowed(version, versions, "version")

    allowed_resolutions = {7, 12, 20, 36, 64, 122, 197, 325, 444}
    if resolution not in allowed_resolutions:
        raise ValueError(
            f"Requested {resolution=} not available. "
            f"Valid options: {allowed_resolutions}"
        )

    file_number = "1861819" if version == "sym" else "1861820"
    url = f"https://ndownloader.figshare.com/files/{file_number}"

    opts = {"uncompress": True}

    dataset_name = "basc_multiscale_2015_atlas"
    data_dir = get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )

    folder_name = Path(f"template_cambridge_basc_multiscale_nii_{version}")

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
        description=Description.from_registry("basc_multiscale_2015_atlas"),
        lut=generate_atlas_look_up_table(
            "fetch_atlas_basc_multiscale_2015", name=labels
        ),
        atlas_type=atlas_type,
        template=f"MNI152{version}",
    )


@fill_doc
def fetch_coords_dosenbach_2010(
    ordered_regions: bool = True,
) -> Bunch[str, str | pd.DataFrame | list[str] | np.ndarray]:
    """Load the Dosenbach et al 160 ROIs.

    These ROIs cover much of the cerebral cortex
    and cerebellum and are assigned to 6 networks.

    For more information
    see the :ref:`dataset description <dosenbach_2010_atlas>`.

    Parameters
    ----------
    ordered_regions : :obj:`bool`, default=True
        ROIs from same networks are grouped together and ordered with respect
        to their names and their locations (anterior to posterior).

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(seitzman_2018_atlas_content)s

    """
    csv = PACKAGE_DIRECTORY / "data" / "dosenbach_2010.csv"
    out_csv = pd.read_csv(csv)

    if ordered_regions:
        out_csv = out_csv.sort_values(by=["network", "name", "y"])

    # We add the ROI number to its name, since names are not unique
    names = out_csv["name"]
    numbers = out_csv["number"]
    labels = [
        f"{name} {number}"
        for (name, number) in zip(names, numbers, strict=False)
    ]
    params = {
        "rois": out_csv[["x", "y", "z"]],
        "labels": labels,
        "networks": out_csv["network"],
        "description": Description.from_registry("dosenbach_2010_atlas"),
        "template": "MNI",
    }
    return Bunch(**params)


@fill_doc
def fetch_coords_seitzman_2018(
    ordered_regions: bool = True,
) -> Bunch[str, str | pd.DataFrame | np.ndarray]:
    """Load the Seitzman et al. 300 ROIs.

    For more information
    see the :ref:`dataset description <seitzman_2018_atlas>`.

    .. nilearn_versionadded:: 0.5.1

    Parameters
    ----------
    ordered_regions : :obj:`bool`, default=True
        ROIs from same networks are grouped together and ordered with respect
        to their locations (anterior to posterior).

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object, contains:

        %(seitzman_2018_atlas_content)s

    """
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

    anatomical = np.genfromtxt(anatomical_file, skip_header=1)
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
        "description": Description.from_registry("seitzman_2018_atlas"),
        "template": "MNI",
    }
    return Bunch(**params)


@fill_doc
def fetch_atlas_allen_2011(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Bunch[str, Any]:
    """Download and return file names for the Allen and MIALAB :term:`ICA` \
    :term:`Probabilistic atlas` (dated 2011).

    For more information
    see the :ref:`dataset description <allen_2011_atlas>`.

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

        %(allen_2011_atlas_content)s

    Notes
    -----
    %(fetcher_note)s

    """
    check_params(locals())

    atlas_type = "probabilistic"

    if url is None:
        url = "https://osf.io/hrcku/download"

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
        "allen_rsn_2011_atlas", data_dir=data_dir, verbose=verbose
    )
    sub_files = fetch_files(
        data_dir, filenames, resume=resume, verbose=verbose
    )

    params = [
        (
            "description",
            Description.from_registry("allen_2011_atlas"),
        ),
        ("atlas_type", atlas_type),
        ("rsn_indices", labels),
        ("networks", networks),
        ("template", "MNI152"),
        *list(zip(keys, sub_files, strict=False)),
    ]

    return Bunch(**dict(params))


@fill_doc
def fetch_atlas_surf_destrieux(
    data_dir: DataDir = None,
    url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Bunch[str, Any]:
    """Download and load Destrieux et al, 2010 cortical \
    :term:`Deterministic atlas`.

    This atlas returns 76 labels per hemisphere based on sulco-gryal patterns
    as distributed with Freesurfer in fsaverage5 surface space.

    For more information,
    see the :ref:`dataset description <surf_destrieux_atlas>`.

    .. nilearn_versionadded:: 0.3

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

        %(surf_destrieux_atlas_content)s

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage
    nilearn.datasets.fetch_atlas_destrieux_2009

    Notes
    -----
    %(fetcher_note)s

    Examples
    --------
    The code snippet below shows how to use this dataset
    to generate a :class:`~nilearn.surface.SurfaceImage`.

    .. code-block::

        from nilearn.datasets import load_fsaverage, fetch_atlas_surf_destrieux
        from nilearn.surface import SurfaceImage

        fsaverage = load_fsaverage("fsaverage5")
        destrieux = fetch_atlas_surf_destrieux()
        labels_img = SurfaceImage(
            mesh=fsaverage.pial,
            data={
                "left": destrieux.map_left,
                "right": destrieux.map_right,
            },
        )

    """
    check_params(locals())

    atlas_type = "deterministic"

    if url is None:
        url = "https://www.nitrc.org/frs/download.php/"

    dataset_name = "surf_destrieux_atlas"
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
    lut = generate_atlas_look_up_table(
        "fetch_atlas_surf_destrieux", name=labels
    )
    check_look_up_table(lut=lut, atlas=annot_left[0], verbose=verbose)
    check_look_up_table(lut=lut, atlas=annot_right[0], verbose=verbose)

    return Bunch(
        labels=labels,
        map_left=annot_left[0],
        map_right=annot_right[0],
        description=Description.from_registry(dataset_name),
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
    )
    atlas_data = _get_data(atlas_img)
    for level_name, old_level_labels in zip(
        _TALAIRACH_LEVELS, np.asarray(labels).T, strict=False
    ):
        logger.log(level_name, verbose=verbose)
        # level with most regions, ba, has 72 regions
        level_data = np.zeros(atlas_img.shape, dtype="uint8")
        level_labels = {"*": 0}
        for region_nb, region_name in enumerate(old_level_labels):
            level_labels.setdefault(region_name, len(level_labels))
            level_data[atlas_data == region_nb] = level_labels[region_name]
        new_img_like(atlas_img, level_data).to_filename(
            output_dir / f"{level_name}.nii.gz"
        )

        level_labels = list(level_labels.keys())
        # rename '*' -> 'Background'
        level_labels[0] = "Background"
        (output_dir / f"{level_name}-labels.json").write_text(
            json.dumps(level_labels), "utf-8"
        )


def _download_talairach(talairach_dir, verbose) -> None:
    """Download the Talairach atlas and separate the different levels."""
    temp_dir = mkdtemp()
    try:
        atlas_url = "https://www.talairach.org/talairach.nii"
        temp_file = fetch_files(
            temp_dir, [("talairach.nii", atlas_url, {})], verbose=verbose
        )[0]
    except SSLError:
        # See https://github.com/nilearn/nilearn/issues/5896
        # A copy of the atlas was hence added
        # to Nilearn OSF
        backup_url = "https://osf.io/x4b2w/download"
        temp_file = fetch_single_file(
            backup_url, Path(temp_dir), verbose=verbose
        )
        shutil.move(temp_file, Path(temp_dir) / "talairach.nii")
        temp_file = Path(temp_dir) / "talairach.nii"

    atlas_img = load(temp_file, mmap=False)
    atlas_img = check_niimg(atlas_img)
    labels_text = atlas_img.header.extensions[0].get_content()
    multi_labels = labels_text.strip().decode("utf-8").split("\n")
    labels = [lab.split(".") for lab in multi_labels]
    _separate_talairach_levels(
        atlas_img, labels, talairach_dir, verbose=verbose
    )

    shutil.rmtree(temp_dir)


@fill_doc
def fetch_atlas_talairach(
    level_name: Literal["hemisphere", "lobe", "gyrus", "tissue", "ba"],
    data_dir: DataDir = None,
    verbose: Verbose = 1,
) -> Atlas:
    """Download the Talairach :term:`Deterministic atlas`.

    For more information,
    see the :ref:`dataset description <talairach_atlas>`.

    .. nilearn_versionadded:: 0.4.0

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

        %(talairach_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())

    atlas_type = "deterministic"

    check_parameter_in_allowed(level_name, _TALAIRACH_LEVELS, "level_name")
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
        description=Description.from_registry("talairach_atlas"),
        lut=generate_atlas_look_up_table("fetch_atlas_talairach", name=labels),
        atlas_type=atlas_type,
        template="Talairach",
    )


@fill_doc
def fetch_atlas_pauli_2017(
    atlas_type: Literal["probabilistic", "deterministic"] = "probabilistic",
    data_dir: DataDir = None,
    verbose: Verbose = 1,
) -> Atlas:
    """Download the Pauli et al. (2017) atlas.

    For more information,
    see the :ref:`dataset description <pauli_2017_atlas>`.

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

        %(pauli_2017_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())
    check_parameter_in_allowed(
        atlas_type, {"probabilistic", "deterministic"}, "atlas_type"
    )

    url_maps = "https://osf.io/w8zq2/download"
    filename = "pauli_2017_prob.nii.gz"
    if atlas_type == "deterministic":
        url_maps = "https://osf.io/5mqfx/download"
        filename = "pauli_2017_det.nii.gz"

    url_labels = "https://osf.io/6qrcb/download"
    dataset_name = "pauli_2017_atlas"

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
        description=Description.from_registry(dataset_name),
        lut=generate_atlas_look_up_table(
            "fetch_atlas_pauli_2017", name=labels
        ),
        atlas_type=atlas_type,
    )


@fill_doc
def fetch_atlas_schaefer_2018(
    n_rois: Literal[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] = 400,
    yeo_networks: Literal[7, 17] = 7,
    resolution_mm: Literal[1, 2] = 1,
    data_dir: DataDir = None,
    base_url: Url = None,
    resume: Resume = True,
    verbose: Verbose = 1,
) -> Atlas:
    """Download and return file names for the Schaefer 2018 parcellation.

    .. nilearn_versionadded:: 0.5.1

    This function returns a :term:`Deterministic atlas`, and the provided
    images are in MNI152 space.

    For more information
    see the :ref:`dataset description <schaefer_2018_atlas>`.

    Parameters
    ----------
    n_rois : {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}, default=400
        Number of regions of interest.

    yeo_networks : {7, 17}, default=7
        ROI annotation according to yeo networks.

    resolution_mm : {1, 2}, default=1
        Spatial resolution of atlas image, in mm.

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

        %(schaefer_2018_atlas_content)s

    Notes
    -----
    %(fetcher_note)s
    """
    check_params(locals())

    atlas_type = "deterministic"

    valid_n_rois = list(range(100, 1100, 100))
    check_parameter_in_allowed(n_rois, valid_n_rois, "n_rois")
    valid_yeo_networks = [7, 17]
    check_parameter_in_allowed(
        yeo_networks, valid_yeo_networks, "yeo_networks"
    )
    valid_resolution_mm = [1, 2]
    check_parameter_in_allowed(
        resolution_mm, valid_resolution_mm, "resolution_mm"
    )

    if base_url is None:
        url = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/"
            "v0.14.3-Update_Yeo2011_Schaefer2018_labelname/"
            "stable_projects/brain_parcellation/"
            "Schaefer2018_LocalGlobal/Parcellations/MNI/"
        )
    else:
        url = base_url

    labels_file_template = "Schaefer2018_{}Parcels_{}Networks_order.txt"
    img_file_template = (
        "Schaefer2018_{}Parcels_{}Networks_order_FSLMNI152_{}mm.nii.gz"
    )
    files: list[tuple[str, str, dict[str, str]]] = [
        (f, url + f, {})
        for f in [
            labels_file_template.format(n_rois, yeo_networks),
            img_file_template.format(n_rois, yeo_networks, resolution_mm),
        ]
    ]

    dataset_name = "schaefer_2018_atlas"
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
        description=Description.from_registry(dataset_name),
        lut=lut,
        atlas_type=atlas_type,
        template="MNI152NLin6Asym",
    )
