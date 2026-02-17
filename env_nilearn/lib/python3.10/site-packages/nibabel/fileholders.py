# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Fileholder class"""

from __future__ import annotations

import typing as ty
from copy import copy

from .openers import ImageOpener

if ty.TYPE_CHECKING:
    import io


class FileHolderError(Exception):
    pass


class FileHolder:
    """class to contain filename, fileobj and file position"""

    def __init__(
        self,
        filename: str | None = None,
        fileobj: io.IOBase | None = None,
        pos: int = 0,
    ):
        """Initialize FileHolder instance

        Parameters
        ----------
        filename : str, optional
           filename.  Default is None
        fileobj : file-like object, optional
           Should implement at least 'seek' (for the purposes for this
           class).  Default is None
        pos : int, optional
           position in filename or fileobject at which to start reading
           or writing data; defaults to 0
        """
        self.filename = filename
        self.fileobj = fileobj
        self.pos = pos

    def get_prepare_fileobj(self, *args, **kwargs) -> ImageOpener:
        """Return fileobj if present, or return fileobj from filename

        Set position to that given in self.pos

        Parameters
        ----------
        *args : tuple
           positional arguments to file open.  Ignored if there is a
           defined ``self.fileobj``.  These might include the mode, such
           as 'rb'
        **kwargs : dict
           named arguments to file open.  Ignored if there is a
           defined ``self.fileobj``

        Returns
        -------
        fileobj : file-like object
           object has position set (via ``fileobj.seek()``) to
           ``self.pos``
        """
        if self.fileobj is not None:
            obj = ImageOpener(self.fileobj)  # for context manager
            obj.seek(self.pos)
        elif self.filename is not None:
            obj = ImageOpener(self.filename, *args, **kwargs)
            if self.pos != 0:
                obj.seek(self.pos)
        else:
            raise FileHolderError('No filename or fileobj present')
        return obj

    def same_file_as(self, other: FileHolder) -> bool:
        """Test if `self` refers to same files / fileobj as `other`

        Parameters
        ----------
        other : object
            object with `filename` and `fileobj` attributes

        Returns
        -------
        tf : bool
            True if `other` has the same filename (or both have None) and the
            same fileobj (or both have None
        """
        return (self.filename == other.filename) and (self.fileobj == other.fileobj)

    @property
    def file_like(self) -> str | io.IOBase | None:
        """Return ``self.fileobj`` if not None, otherwise ``self.filename``"""
        return self.fileobj if self.fileobj is not None else self.filename


FileMap = ty.Mapping[str, FileHolder]


def copy_file_map(file_map: FileMap) -> FileMap:
    r"""Copy mapping of fileholders given by `file_map`

    Parameters
    ----------
    file_map : mapping
       mapping of ``FileHolder`` instances

    Returns
    -------
    fm_copy : dict
       Copy of `file_map`, using shallow copy of ``FileHolder``\s

    """
    return {key: copy(fh) for key, fh in file_map.items()}
