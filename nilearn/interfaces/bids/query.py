"""Functions for working with BIDS datasets."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from warnings import warn

from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level


def _get_metadata_from_bids(
    field,
    json_files,
    bids_path=None,
):
    """Get a metadata field from a BIDS json sidecar files.

    This assumes that all the json files in the list have the same value
    for that field,
    hence the metadata is read only from the first json file in the list.

    Parameters
    ----------
    field : :obj:`str`
        Name of the field to be read. For example 'RepetitionTime'.

    json_files : :obj:`list` of :obj:`str`
        List of path to json files, for example returned by get_bids_files.

    bids_path : :obj:`str` or :obj:`pathlib.Path`, optional
        Fullpath to the BIDS dataset.

    Returns
    -------
    float or None
        value of the field or None if the field is not found.
    """
    if json_files:
        assert isinstance(json_files, list) and isinstance(
            json_files[0], (Path, str)
        )
        with Path(json_files[0]).open() as f:
            specs = json.load(f)
        value = specs.get(field)
        if value is not None:
            return value
        else:
            warn(
                f"'{field}' not found in file {json_files[0]}.",
                stacklevel=find_stack_level(),
            )
    else:
        msg_suffix = f" in:\n {bids_path}" if bids_path else ""
        warn(
            f"\nNo bold.json found in BIDS folder{msg_suffix}.",
            stacklevel=find_stack_level(),
        )

    return None


@fill_doc
def infer_slice_timing_start_time_from_dataset(bids_path, filters, verbose=0):
    """Return the StartTime metadata field from a BIDS derivatives dataset.

    This corresponds to the reference time (in seconds) used for the slice
    timing correction.

    See https://github.com/bids-standard/bids-specification/issues/836

    Parameters
    ----------
    bids_path : :obj:`str` or :obj:`pathlib.Path`
        Fullpath to the derivatives folder of the BIDS dataset.

    filters : :obj:`list` of :obj:`tuple` (:obj:`str`, :obj:`str`), optional
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Filter examples would be ('ses', '01'), ('dir', 'ap') and
        ('task', 'localizer').

    %(verbose0)s

    Returns
    -------
    float or None
        Value of the field or None if the field is not found.

    """
    img_specs = get_bids_files(
        bids_path,
        modality_folder="func",
        file_tag="bold",
        file_type="json",
        filters=filters,
    )
    if not img_specs:
        if verbose:
            msg_suffix = f" in:\n {bids_path}"
            warn(
                f"\nNo bold.json found in BIDS folder{msg_suffix}.",
                stacklevel=find_stack_level(),
            )
        return None

    return _get_metadata_from_bids(
        field="StartTime",
        json_files=img_specs,
        bids_path=bids_path,
    )


@fill_doc
def infer_repetition_time_from_dataset(bids_path, filters, verbose=0):
    """Return the RepetitionTime metadata field from a BIDS dataset.

    Parameters
    ----------
    bids_path : :obj:`str` or :obj:`pathlib.Path`
        Fullpath to the raw folder of the BIDS dataset.

    filters : :obj:`list` of :obj:`tuple` (:obj:`str`, :obj:`str`), optional
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Filter examples would be ('ses', '01'), ('dir', 'ap') and
        ('task', 'localizer').

    %(verbose0)s

    Returns
    -------
    float or None
        Value of the field or None if the field is not found.

    """
    img_specs = get_bids_files(
        main_path=bids_path,
        modality_folder="func",
        file_tag="bold",
        file_type="json",
        filters=filters,
    )

    if not img_specs:
        if verbose:
            msg_suffix = f" in:\n {bids_path}"
            warn(
                f"\nNo bold.json found in BIDS folder{msg_suffix}.",
                stacklevel=find_stack_level(),
            )
        return None

    return _get_metadata_from_bids(
        field="RepetitionTime",
        json_files=img_specs,
        bids_path=bids_path,
    )


def get_bids_files(
    main_path,
    file_tag="*",
    file_type="*",
    sub_label="*",
    modality_folder="*",
    filters=None,
    sub_folder=True,
):
    """Search for files in a :term:`BIDS` dataset following given constraints.

    This utility function allows to filter files in the :term:`BIDS` dataset by
    any of the fields contained in the file names. Moreover it allows to search
    for specific types of files or particular tags.

    The provided filters have to correspond to a file name field, so
    any file not containing the field will be ignored. For example the filter
    ('sub', '01') would return all files corresponding to the first
    subject that specifically contain in the file name 'sub-01'. If more
    filters are given then we constraint the possible files names accordingly.

    Notice that to search in the derivatives folder, it has to be given as
    part of the main_path. This is useful since the current convention gives
    exactly the same inner structure to derivatives than to the main
    :term:`BIDS` dataset folder, so we can search it in the same way.

    Parameters
    ----------
    main_path : :obj:`str` or :obj:`pathlib.Path`
        Directory of the :term:`BIDS` dataset.

    file_tag : :obj:`str` accepted by glob, default='*'
        The final tag of the desired files. For example 'bold' if one is
        interested in the files related to the neuroimages.

    file_type : :obj:`str` accepted by glob, default='*'
        The type of the desired files. For example to be able to request only
        'nii' or 'json' files for the 'bold' tag.

    sub_label : :obj:`str` accepted by glob, default='*'
        Such a common filter is given as a direct option since it applies also
        at the level of directories. the label is what follows the 'sub' field
        in the :term:`BIDS` convention as 'sub-label'.

    modality_folder : :obj:`str` accepted by glob, default='*'
        Inside the subject and optional session folders a final level of
        folders is expected in the :term:`BIDS` convention that groups files
        according to different neuroimaging modalities and any other additions
        of the dataset provider. For example the 'func' and 'anat' standard
        folders. If given as the empty string '', files will be searched
        inside the sub-label/ses-label directories.

    filters : :obj:`list` of :obj:`tuple` (:obj:`str`, :obj:`str`), \
              default=None
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Filter examples would be ('ses', '01'), ('dir', 'ap') and
        ('task', 'localizer').

    sub_folder : :obj:`bool`, default=True
        Determines if the files searched are at the level of
        subject/session folders or just below the dataset main folder.
        Setting this option to False with other default values would return
        all the files below the main directory, ignoring files in subject
        or derivatives folders.

    Returns
    -------
    files : :obj:`list` of :obj:`str`
        List of file paths found.

    """
    main_path = Path(main_path)
    if sub_folder:
        files = main_path / "sub-*" / "ses-*"
        session_folder_exists = glob.glob(str(files))
        ses_level = "ses-*" if session_folder_exists else ""
        files = (
            main_path
            / f"sub-{sub_label}"
            / ses_level
            / modality_folder
            / f"sub-{sub_label}*_{file_tag}.{file_type}"
        )
    else:
        files = main_path / f"*{file_tag}.{file_type}"

    files = glob.glob(str(files))
    files.sort()

    filters = filters or []
    if filters:
        files = [parse_bids_filename(file_, legacy=False) for file_ in files]
        for entity, label in filters:
            files = [
                file_
                for file_ in files
                if (entity not in file_["entities"] and label == "")
                or (
                    entity in file_["entities"]
                    and file_["entities"][entity] == label
                )
            ]
        return [ref_file["file_path"] for ref_file in files]

    return files


def parse_bids_filename(img_path, legacy=True):
    r"""Return dictionary with parsed information from file path.

    Parameters
    ----------
    img_path : :obj:`str`
        Path to file from which to parse information.

    legacy : :obj:`bool`, default=True
        Whether to return a dictionary that uses BIDS terms (``False``)
        or the legacy content for the output (``True``).
        ``False`` will become the default in version >= 0.13.0.

        .. versionadded :: 0.12.0

    Returns
    -------
    reference : :obj:`dict`
        Returns a dictionary with all key-value pairs in the file name
        parsed and other useful fields.

        The dictionary will contain ``'file_path'``, ``'file_basename'``.

        If ``legacy`` is set to ``True``,
        the dictionary will also contain
        'file_tag', 'file_type' and 'file_fields'.
        The 'file_tag' field refers to the last part of the file under the
        :term:`BIDS` convention that is of the form \*_tag.type.
        Contrary to the rest of the file name it is not a key-value pair.
        This notion should be revised in the case we are handling derivatives
        since so far the convention will keep the tag prepended to any fields
        added in the case of preprocessed files that also end with another tag.
        This parser will consider any tag in the middle of the file name as a
        key with no value and will be included in the 'file_fields' key.

        If ``legacy`` is set to ``False``,
        the dictionary will instead contain
        ``'extension'``, ``'suffix'`` and ``'entities'``.
        (See the documentation on
        `typical bids filename <https://bids.neuroimaging.io/getting_started/folders_and_files/files.html#filename-template>`_
        for more information).

    """
    reference = {
        "file_path": img_path,
        "file_basename": Path(img_path).name,
    }
    parts = reference["file_basename"].split("_")
    suffix, extension = parts[-1].split(".", 1)

    if legacy:
        # TODO (nilearn >= 0.13.0)
        warn(
            (
                "For versions >= 0.13.0 this function will always return "
                "a dictionary that uses BIDS terms as keys. "
                "Set 'legacy=False' to start using this new behavior."
            ),
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )

        reference["file_tag"] = suffix
        reference["file_type"] = extension
        reference["file_fields"] = []
        for part in parts[:-1]:
            field = part.split("-")[0]
            reference["file_fields"].append(field)
            # In derivatives is not clear if the source file name will
            # be parsed as a field with no value.
            reference[field] = None
            if len(part.split("-")) > 1:
                value = part.split("-")[1]
                reference[field] = value

    else:
        reference["extension"] = extension
        reference["suffix"] = suffix
        reference["entities"] = {}
        for part in parts[:-1]:
            entity = part.split("-")[0]
            # In derivatives is not clear if the source file name will
            # be parsed as a field with no value.
            label = None
            if len(part.split("-")) > 1:
                value = part.split("-")[1]
                label = value
            reference["entities"][entity] = label

    return reference
