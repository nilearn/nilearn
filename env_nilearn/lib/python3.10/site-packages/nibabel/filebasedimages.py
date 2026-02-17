# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common interface for any image format--volume or surface, binary or xml"""

from __future__ import annotations

import io
import typing as ty
from copy import deepcopy
from urllib import request

from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener

if ty.TYPE_CHECKING:
    from .filename_parser import ExtensionSpec, FileSpec

FileSniff = tuple[bytes, str]

ImgT = ty.TypeVar('ImgT', bound='FileBasedImage')
HdrT = ty.TypeVar('HdrT', bound='FileBasedHeader')

StreamImgT = ty.TypeVar('StreamImgT', bound='SerializableImage')


class ImageFileError(Exception):
    pass


class FileBasedHeader:
    """Template class to implement header protocol"""

    @classmethod
    def from_header(klass: type[HdrT], header: FileBasedHeader | ty.Mapping | None = None) -> HdrT:
        if header is None:
            return klass()
        # I can't do isinstance here because it is not necessarily true
        # that a subclass has exactly the same interface as its parent
        # - for example Nifti1Images inherit from Analyze, but have
        # different field names
        if type(header) == klass:
            return header.copy()
        raise NotImplementedError(
            f'Header class requires a conversion from {klass} to {type(header)}'
        )

    @classmethod
    def from_fileobj(klass: type[HdrT], fileobj: io.IOBase) -> HdrT:
        raise NotImplementedError

    def write_to(self, fileobj: io.IOBase) -> None:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        return not self == other

    def copy(self: HdrT) -> HdrT:
        """Copy object to independent representation

        The copy should not be affected by any changes to the original
        object.
        """
        return deepcopy(self)


class FileBasedImage:
    """
    Abstract image class with interface for loading/saving images from disk.

    The class doesn't define any image properties.

    It has:

    attributes:

       * extra

    properties:

       * header

    methods:

       * to_filename(fname) - writes data to filename(s) derived from
         ``fname``, where the derivation may differ between formats.
       * to_file_map() - save image to files with which the image is already
         associated.

    classmethods:

       * from_filename(fname) - make instance by loading from filename
       * from_file_map(fmap) - make instance from file map
       * instance_to_filename(img, fname) - save ``img`` instance to
         filename ``fname``.

    It also has a ``header`` - some standard set of meta-data that is specific
    to the image format, and ``extra`` - a dictionary container for any other
    metadata.

    You cannot slice an image, and trying to slice an image generates an
    informative TypeError.

    **There are several ways of writing data**

    There is the usual way, which is the default::

        img.to_filename(fname)

    and that is, to take the data encapsulated by the image and cast it to
    the datatype the header expects, setting any available header scaling
    into the header to help the data match.

    You can load the data into an image from file with::

       img.from_filename(fname)

    The image stores its associated files in its ``file_map`` attribute.  In
    order to just save an image, for which you know there is an associated
    filename, or other storage, you can do::

       img.to_file_map()

    **Files interface**

    The image has an attribute ``file_map``.  This is a mapping, that has keys
    corresponding to the file types that an image needs for storage.  For
    example, the Analyze data format needs an ``image`` and a ``header``
    file type for storage:

       >>> import numpy as np
       >>> import nibabel as nib
       >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
       >>> img = nib.AnalyzeImage(data, np.eye(4))
       >>> sorted(img.file_map)
       ['header', 'image']

    The values of ``file_map`` are not in fact files but objects with
    attributes ``filename``, ``fileobj`` and ``pos``.

    The reason for this interface, is that the contents of files has to
    contain enough information so that an existing image instance can save
    itself back to the files pointed to in ``file_map``.  When a file holder
    holds active file-like objects, then these may be affected by the
    initial file read; in this case, the file-like objects need to
    carry the position at which a write (with ``to_file_map``) should place the
    data.  The ``file_map`` contents should therefore be such, that this will
    work.
    """

    header_class: type[FileBasedHeader] = FileBasedHeader
    _meta_sniff_len: int = 0
    files_types: tuple[ExtensionSpec, ...] = (('image', None),)
    valid_exts: tuple[str, ...] = ()
    _compressed_suffixes: tuple[str, ...] = ()

    makeable: bool = True  # Used in test code
    rw: bool = True  # Used in test code

    def __init__(
        self,
        header: FileBasedHeader | ty.Mapping | None = None,
        extra: ty.Mapping | None = None,
        file_map: FileMap | None = None,
    ):
        """Initialize image

        The image is a combination of (header), with
        optional metadata in `extra`, and filename / file-like objects
        contained in the `file_map` mapping.

        Parameters
        ----------
        header : None or mapping or header instance, optional
           metadata for this image format
        extra : None or mapping, optional
           metadata to associate with image that cannot be stored in the
           metadata of this image type
        file_map : mapping, optional
           mapping giving file information for this image format
        """
        self._header = self.header_class.from_header(header)
        if extra is None:
            extra = {}
        self.extra = dict(extra)

        if file_map is None:
            file_map = self.__class__.make_file_map()
        self.file_map = file_map

    @property
    def header(self) -> FileBasedHeader:
        return self._header

    def __getitem__(self, key) -> None:
        """No slicing or dictionary interface for images"""
        raise TypeError('Cannot slice image objects.')

    def get_filename(self) -> str | None:
        """Fetch the image filename

        Parameters
        ----------
        None

        Returns
        -------
        fname : None or str
           Returns None if there is no filename, or a filename string.
           If an image may have several filenames associated with it (e.g.
           Analyze ``.img, .hdr`` pair) then we return the more characteristic
           filename (the ``.img`` filename in the case of Analyze')
        """
        # which filename is returned depends on the ordering of the
        # 'files_types' class attribute - we return the name
        # corresponding to the first in that tuple
        characteristic_type = self.files_types[0][0]
        return self.file_map[characteristic_type].filename

    def set_filename(self, filename: str) -> None:
        """Sets the files in the object from a given filename

        The different image formats may check whether the filename has
        an extension characteristic of the format, and raise an error if
        not.

        Parameters
        ----------
        filename : str or os.PathLike
           If the image format only has one file associated with it,
           this will be the only filename set into the image
           ``.file_map`` attribute. Otherwise, the image instance will
           try and guess the other filenames from this given filename.
        """
        self.file_map = self.__class__.filespec_to_file_map(filename)

    @classmethod
    def from_filename(klass: type[ImgT], filename: FileSpec) -> ImgT:
        file_map = klass.filespec_to_file_map(filename)
        return klass.from_file_map(file_map)

    @classmethod
    def from_file_map(klass: type[ImgT], file_map: FileMap) -> ImgT:
        raise NotImplementedError

    @classmethod
    def filespec_to_file_map(klass, filespec: FileSpec) -> FileMap:
        """Make `file_map` for this class from filename `filespec`

        Class method

        Parameters
        ----------
        filespec : str or os.PathLike
            Filename that might be for this image file type.

        Returns
        -------
        file_map : dict
            `file_map` dict with (key, value) pairs of (``file_type``,
            FileHolder instance), where ``file_type`` is a string giving the
            type of the contained file.

        Raises
        ------
        ImageFileError
            if `filespec` is not recognizable as being a filename for this
            image type.
        """
        try:
            filenames = types_filenames(
                filespec, klass.files_types, trailing_suffixes=klass._compressed_suffixes
            )
        except TypesFilenamesError:
            raise ImageFileError(f'Filespec "{filespec}" does not look right for class {klass}')
        file_map = {}
        for key, fname in filenames.items():
            file_map[key] = FileHolder(filename=fname)
        return file_map

    def to_filename(self, filename: FileSpec, **kwargs) -> None:
        r"""Write image to files implied by filename string

        Parameters
        ----------
        filename : str or os.PathLike
           filename to which to save image.  We will parse `filename`
           with ``filespec_to_file_map`` to work out names for image,
           header etc.
        \*\*kwargs : keyword arguments
           Keyword arguments to format-specific save

        Returns
        -------
        None
        """
        self.file_map = self.filespec_to_file_map(filename)
        self.to_file_map(**kwargs)

    def to_file_map(self, file_map: FileMap | None = None, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def make_file_map(klass, mapping: ty.Mapping[str, str | io.IOBase] | None = None) -> FileMap:
        """Class method to make files holder for this image type

        Parameters
        ----------
        mapping : None or mapping, optional
           mapping with keys corresponding to image file types (such as
           'image', 'header' etc, depending on image class) and values
           that are filenames or file-like.  Default is None

        Returns
        -------
        file_map : dict
           dict with string keys given by first entry in tuples in
           sequence klass.files_types, and values of type FileHolder,
           where FileHolder objects have default values, other than
           those given by `mapping`
        """
        if mapping is None:
            mapping = {}
        file_map = {}
        for key, ext in klass.files_types:
            file_map[key] = FileHolder()
            mapval = mapping.get(key, None)
            if isinstance(mapval, str):
                file_map[key].filename = mapval
            elif hasattr(mapval, 'tell'):
                file_map[key].fileobj = mapval
        return file_map

    load = from_filename

    @classmethod
    def instance_to_filename(klass, img: FileBasedImage, filename: FileSpec) -> None:
        """Save `img` in our own format, to name implied by `filename`

        This is a class method

        Parameters
        ----------
        img : ``any FileBasedImage`` instance

        filename : str
           Filename, implying name to which to save image.
        """
        img = klass.from_image(img)
        img.to_filename(filename)

    @classmethod
    def from_image(klass: type[ImgT], img: FileBasedImage) -> ImgT:
        """Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : ``FileBasedImage`` instance
           In fact, an object with the API of ``FileBasedImage``.

        Returns
        -------
        img : ``FileBasedImage`` instance
           Image, of our own class
        """
        raise NotImplementedError

    @classmethod
    def _sniff_meta_for(
        klass,
        filename: FileSpec,
        sniff_nbytes: int,
        sniff: FileSniff | None = None,
    ) -> FileSniff | None:
        """Sniff metadata for image represented by `filename`

        Parameters
        ----------
        filename : str or os.PathLike
            Filename for an image, or an image header (metadata) file.
            If `filename` points to an image data file, and the image type has
            a separate "header" file, we work out the name of the header file,
            and read from that instead of `filename`.
        sniff_nbytes : int
            Number of bytes to read from the image or metadata file
        sniff : (bytes, fname), optional
            The result of a previous call to `_sniff_meta_for`.  If fname
            matches the computed header file name, `sniff` is returned without
            rereading the file.

        Returns
        -------
        sniff : None or (bytes, fname)
            None if we could not read the image or metadata file.  `sniff[0]`
            is either length `sniff_nbytes` or the length of the image /
            metadata file, whichever is the shorter. `fname` is the name of
            the sniffed file.
        """
        froot, ext, trailing = splitext_addext(filename, klass._compressed_suffixes)
        # Determine the metadata location
        t_fnames = types_filenames(
            filename, klass.files_types, trailing_suffixes=klass._compressed_suffixes
        )
        meta_fname = t_fnames.get('header', _stringify_path(filename))

        # Do not re-sniff if it would be from the same file
        if sniff is not None and sniff[1] == meta_fname:
            return sniff

        # Attempt to sniff from metadata location
        try:
            with ImageOpener(meta_fname, 'rb') as fobj:
                binaryblock = fobj.read(sniff_nbytes)
        except COMPRESSION_ERRORS + (OSError, EOFError):
            return None
        return (binaryblock, meta_fname)

    @classmethod
    def path_maybe_image(
        klass,
        filename: FileSpec,
        sniff: FileSniff | None = None,
        sniff_max: int = 1024,
    ) -> tuple[bool, FileSniff | None]:
        """Return True if `filename` may be image matching this class

        Parameters
        ----------
        filename : str or os.PathLike
            Filename for an image, or an image header (metadata) file.
            If `filename` points to an image data file, and the image type has
            a separate "header" file, we work out the name of the header file,
            and read from that instead of `filename`.
        sniff : None or (bytes, filename), optional
            Bytes content read from a previous call to this method, on another
            class, with metadata filename.  This allows us to read metadata
            bytes once from the image or header, and pass this read set of
            bytes to other image classes, therefore saving a repeat read of the
            metadata.  `filename` is used to validate that metadata would be
            read from the same file, re-reading if not.  None forces this
            method to read the metadata.
        sniff_max : int, optional
            The maximum number of bytes to read from the metadata.  If the
            metadata file is long enough, we read this many bytes from the
            file, otherwise we read to the end of the file.  Longer values
            sniff more of the metadata / image file, making it more likely that
            the returned sniff will be useful for later calls to
            ``path_maybe_image`` for other image classes.

        Returns
        -------
        maybe_image : bool
            True if `filename` may be valid for an image of this class.
        sniff : None or (bytes, filename)
            Read bytes content from found metadata.  May be None if the file
            does not appear to have useful metadata.
        """
        froot, ext, trailing = splitext_addext(filename, klass._compressed_suffixes)
        if ext.lower() not in klass.valid_exts:
            return False, sniff
        if not hasattr(klass.header_class, 'may_contain_header'):
            return True, sniff

        # Force re-sniff on too-short sniff
        if sniff is not None and len(sniff[0]) < klass._meta_sniff_len:
            sniff = None
        sniff = klass._sniff_meta_for(filename, max(klass._meta_sniff_len, sniff_max), sniff)
        if sniff is None or len(sniff[0]) < klass._meta_sniff_len:
            return False, sniff
        return klass.header_class.may_contain_header(sniff[0]), sniff


class SerializableImage(FileBasedImage):
    """
    Abstract image class for (de)serializing images to/from byte streams/strings.

    The class doesn't define any image properties.

    It has:

    methods:

       * to_bytes() - serialize image to byte string

    classmethods:

       * from_bytes(bytestring) - make instance by deserializing a byte string
       * from_url(url) - make instance by fetching and deserializing a URL

    Loading from byte strings should provide round-trip equivalence:

    .. code:: python

        img_a = klass.from_bytes(bstr)
        img_b = klass.from_bytes(img_a.to_bytes())

        np.allclose(img_a.get_fdata(), img_b.get_fdata())
        np.allclose(img_a.affine, img_b.affine)

    Further, for images that are single files on disk, the following methods of loading
    the image must be equivalent:

    .. code:: python

        img = klass.from_filename(fname)

        with open(fname, 'rb') as fobj:
            img = klass.from_bytes(fobj.read())

    And the following methods of saving a file must be equivalent:

    .. code:: python

        img.to_filename(fname)

        with open(fname, 'wb') as fobj:
            fobj.write(img.to_bytes())

    Images that consist of separate header and data files (e.g., Analyze
    images) currently do not support this interface.
    For multi-file images, ``to_bytes()`` and ``from_bytes()`` must be
    overridden, and any encoding details should be documented.
    """

    @classmethod
    def _filemap_from_iobase(klass, io_obj: io.IOBase) -> FileMap:
        """For single-file image types, make a file map with the correct key"""
        if len(klass.files_types) > 1:
            raise NotImplementedError('(de)serialization is undefined for multi-file images')
        return klass.make_file_map({klass.files_types[0][0]: io_obj})

    @classmethod
    def from_stream(klass: type[StreamImgT], io_obj: io.IOBase) -> StreamImgT:
        """Load image from readable IO stream

        Convert to BytesIO to enable seeking, if input stream is not seekable

        Parameters
        ----------
        io_obj : IOBase object
            Readable stream
        """
        if not io_obj.seekable():
            io_obj = io.BytesIO(io_obj.read())
        return klass.from_file_map(klass._filemap_from_iobase(io_obj))

    def to_stream(self, io_obj: io.IOBase, **kwargs) -> None:
        r"""Save image to writable IO stream

        Parameters
        ----------
        io_obj : IOBase object
            Writable stream
        \*\*kwargs : keyword arguments
            Keyword arguments that may be passed to ``img.to_file_map()``
        """
        self.to_file_map(self._filemap_from_iobase(io_obj), **kwargs)

    @classmethod
    def from_bytes(klass: type[StreamImgT], bytestring: bytes) -> StreamImgT:
        """Construct image from a byte string

        Class method

        Parameters
        ----------
        bytestring : bytes
            Byte string containing the on-disk representation of an image
        """
        return klass.from_stream(io.BytesIO(bytestring))

    def to_bytes(self, **kwargs) -> bytes:
        r"""Return a ``bytes`` object with the contents of the file that would
        be written if the image were saved.

        Parameters
        ----------
        \*\*kwargs : keyword arguments
            Keyword arguments that may be passed to ``img.to_file_map()``

        Returns
        -------
        bytes
            Serialized image
        """
        bio = io.BytesIO()
        self.to_stream(bio, **kwargs)
        return bio.getvalue()

    @classmethod
    def from_url(
        klass: type[StreamImgT], url: str | request.Request, timeout: float = 5
    ) -> StreamImgT:
        """Retrieve and load an image from a URL

        Class method

        Parameters
        ----------
        url : str or urllib.request.Request object
            URL of file to retrieve
        timeout : float, optional
            Time (in seconds) to wait for a response
        """
        response = request.urlopen(url, timeout=timeout)
        return klass.from_stream(response)
