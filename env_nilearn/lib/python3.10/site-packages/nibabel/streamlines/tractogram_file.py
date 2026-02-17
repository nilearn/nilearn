"""Define abstract interface for Tractogram file classes"""

from abc import ABC, abstractmethod

from .header import Field


class ExtensionWarning(Warning):
    """Base class for warnings about tractogram file extension."""


class HeaderWarning(Warning):
    """Base class for warnings about tractogram file header."""


class DataWarning(Warning):
    """Base class for warnings about tractogram file data."""


class HeaderError(Exception):
    """Raised when a tractogram file header contains invalid information."""


class DataError(Exception):
    """Raised when data is missing or inconsistent in a tractogram file."""


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class TractogramFile(ABC):
    """Convenience class to encapsulate tractogram file format."""

    def __init__(self, tractogram, header=None):
        self._tractogram = tractogram
        self._header = self.create_empty_header() if header is None else header

    @property
    def tractogram(self):
        return self._tractogram

    @property
    def streamlines(self):
        return self.tractogram.streamlines

    @property
    def header(self):
        return self._header

    @property
    def affine(self):
        """voxmm -> rasmm affine."""
        return self.header.get(Field.VOXEL_TO_RASMM)

    @abstractclassmethod
    def is_correct_format(cls, fileobj):
        """Checks if the file has the right streamlines file format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is in the right streamlines file format,
            otherwise returns False.
        """
        raise NotImplementedError

    @classmethod
    def create_empty_header(cls):
        """Returns an empty header for this streamlines file format."""
        return {}

    @abstractclassmethod
    def load(cls, fileobj, lazy_load=True):
        """Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).
        lazy_load : {False, True}, optional
            If True, load streamlines in a lazy manner i.e. they will not be
            kept in memory. Otherwise, load all streamlines in memory.

        Returns
        -------
        tractogram_file : :class:`TractogramFile` object
            Returns an object containing tractogram data and header
            information.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, fileobj):
        """Saves streamlines to a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            opened and ready to write.
        """
        raise NotImplementedError
