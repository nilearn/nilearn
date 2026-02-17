"""Multiformat-capable streamline format read / write interface"""

import os
import warnings

from .array_sequence import ArraySequence
from .header import Field
from .tck import TckFile
from .tractogram import LazyTractogram, Tractogram
from .tractogram_file import ExtensionWarning
from .trk import TrkFile

# List of all supported formats
FORMATS = {
    '.trk': TrkFile,
    '.tck': TckFile,
}


def is_supported(fileobj):
    """Checks if the file-like object if supported by NiBabel.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a streamlines file (and ready to read from the beginning of the
        header)

    Returns
    -------
    is_supported : boolean
    """
    return detect_format(fileobj) is not None


def detect_format(fileobj):
    """Returns the StreamlinesFile object guessed from the file-like object.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a tractogram file (and ready to read from the beginning of the
        header)

    Returns
    -------
    tractogram_file : :class:`TractogramFile` class
        The class type guessed from the content of `fileobj`.
    """
    for format in FORMATS.values():
        try:
            if format.is_correct_format(fileobj):
                return format
        except OSError:
            pass

    if isinstance(fileobj, str):
        _, ext = os.path.splitext(fileobj)
        return FORMATS.get(ext.lower())

    return None


def load(fileobj, lazy_load=False):
    """Loads streamlines in *RAS+* and *mm* space from a file-like object.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object
        pointing to a streamlines file (and ready to read from the beginning
        of the streamlines file's header).
    lazy_load : {False, True}, optional
        If True, load streamlines in a lazy manner i.e. they will not be kept
        in memory and only be loaded when needed.
        Otherwise, load all streamlines in memory.

    Returns
    -------
    tractogram_file : :class:`TractogramFile` object
        Returns an instance of a :class:`TractogramFile` containing data and
        metadata of the tractogram loaded from `fileobj`.

    Notes
    -----
    The streamline coordinate (0,0,0) refers to the center of the voxel.
    """
    tractogram_file = detect_format(fileobj)

    if tractogram_file is None:
        raise ValueError(f"Unknown format for 'fileobj': {fileobj}")

    return tractogram_file.load(fileobj, lazy_load=lazy_load)


def save(tractogram, filename, **kwargs):
    r"""Saves a tractogram to a file.

    Parameters
    ----------
    tractogram : :class:`Tractogram` object or :class:`TractogramFile` object
        If :class:`Tractogram` object, the file format will be guessed from
        `filename` and a :class:`TractogramFile` object will be created using
        provided keyword arguments.
        If :class:`TractogramFile` object, the file format is known and will
        be used to save its content to `filename`.
    filename : str
        Name of the file where the tractogram will be saved.
    \*\*kwargs : keyword arguments
        Keyword arguments passed to :class:`TractogramFile` constructor.
        Should not be specified if `tractogram` is already an instance of
        :class:`TractogramFile`.
    """
    tractogram_file_class = detect_format(filename)
    if isinstance(tractogram, Tractogram):
        if tractogram_file_class is None:
            msg = f"Unknown tractogram file format: '{filename}'"
            raise ValueError(msg)

        tractogram_file = tractogram_file_class(tractogram, **kwargs)

    else:  # Assume it's a TractogramFile object.
        tractogram_file = tractogram
        if tractogram_file_class is None or not isinstance(tractogram_file, tractogram_file_class):
            msg = (
                'The extension you specified is unusual for the provided'
                " 'TractogramFile' object."
            )
            warnings.warn(msg, ExtensionWarning)

        if kwargs:
            msg = "A 'TractogramFile' object was provided, no need for keyword arguments."
            raise ValueError(msg)

    tractogram_file.save(filename)
