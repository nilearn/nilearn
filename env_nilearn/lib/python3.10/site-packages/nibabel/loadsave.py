# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# module imports
"""Utilities to load and save image objects"""

from __future__ import annotations

import os
import typing as ty

import numpy as np

from .arrayproxy import is_proxy
from .deprecated import deprecate_with_version
from .filebasedimages import ImageFileError
from .filename_parser import _stringify_path, splitext_addext
from .imageclasses import all_image_classes
from .openers import ImageOpener

_compressed_suffixes = ('.gz', '.bz2', '.zst')


if ty.TYPE_CHECKING:
    from .filebasedimages import FileBasedImage
    from .filename_parser import FileSpec

    P = ty.ParamSpec('P')

    class Signature(ty.TypedDict):
        signature: bytes
        format_name: str


def _signature_matches_extension(filename: FileSpec) -> tuple[bool, str]:
    """Check if signature aka magic number matches filename extension.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file to check

    Returns
    -------
    matches : bool
       - `True` if the filename extension is not recognized (not .gz nor .bz2)
       - `True` if the magic number was successfully read and corresponds to
         the format indicated by the extension.
       - `False` otherwise.
    error_message : str
       An error message if opening the file failed or a mismatch is detected;
       the empty string otherwise.

    """
    signatures: dict[str, Signature] = {
        '.gz': {'signature': b'\x1f\x8b', 'format_name': 'gzip'},
        '.bz2': {'signature': b'BZh', 'format_name': 'bzip2'},
        '.zst': {'signature': b'\x28\xb5\x2f\xfd', 'format_name': 'ztsd'},
    }
    filename = _stringify_path(filename)
    *_, ext = splitext_addext(filename)
    ext = ext.lower()
    if ext not in signatures:
        return True, ''
    expected_signature = signatures[ext]['signature']
    try:
        with open(filename, 'rb') as fh:
            sniff = fh.read(len(expected_signature))
    except OSError:
        return False, f'Could not read file: {filename}'
    if sniff.startswith(expected_signature):
        return True, ''
    format_name = signatures[ext]['format_name']
    return False, f'File {filename} is not a {format_name} file'


def load(filename: FileSpec, **kwargs) -> FileBasedImage:
    r"""Load file given filename, guessing at file type

    Parameters
    ----------
    filename : str or os.PathLike
       specification of file to load
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type
    """
    filename = _stringify_path(filename)

    # Check file exists and is not empty
    try:
        stat_result = os.stat(filename)
    except OSError:
        raise FileNotFoundError(f"No such file or no access: '{filename}'")
    if stat_result.st_size <= 0:
        raise ImageFileError(f"Empty file: '{filename}'")

    sniff = None
    for image_klass in all_image_classes:
        is_valid, sniff = image_klass.path_maybe_image(filename, sniff)
        if is_valid:
            img = image_klass.from_filename(filename, **kwargs)
            return img

    matches, msg = _signature_matches_extension(filename)
    if not matches:
        raise ImageFileError(msg)

    raise ImageFileError(f'Cannot work out file type of "{filename}"')


@deprecate_with_version('guessed_image_type deprecated.', '3.2', '5.0')
def guessed_image_type(filename):
    """Guess image type from file `filename`

    Parameters
    ----------
    filename : str
        File name containing an image

    Returns
    -------
    image_class : class
        Class corresponding to guessed image type
    """
    sniff = None
    for image_klass in all_image_classes:
        is_valid, sniff = image_klass.path_maybe_image(filename, sniff)
        if is_valid:
            return image_klass

    raise ImageFileError(f'Cannot work out file type of "{filename}"')


def save(img: FileBasedImage, filename: FileSpec, **kwargs) -> None:
    r"""Save an image to file adapting format to `filename`

    Parameters
    ----------
    img : ``SpatialImage``
       image to save
    filename : str or os.PathLike
       filename (often implying filenames) to which to save `img`.
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific save

    Returns
    -------
    None
    """
    filename = _stringify_path(filename)

    # Save the type as expected
    try:
        img.to_filename(filename, **kwargs)
    except ImageFileError:
        pass
    else:
        return

    # Be nice to users by making common implicit conversions
    froot, ext, trailing = splitext_addext(filename, _compressed_suffixes)
    lext = ext.lower()

    # Special-case Nifti singles and Pairs
    # Inline imports, as this module really shouldn't reference any image type
    from .nifti1 import Nifti1Image, Nifti1Pair
    from .nifti2 import Nifti2Image, Nifti2Pair

    converted: FileBasedImage
    if type(img) == Nifti1Image and lext in ('.img', '.hdr'):
        converted = Nifti1Pair.from_image(img)
    elif type(img) == Nifti2Image and lext in ('.img', '.hdr'):
        converted = Nifti2Pair.from_image(img)
    elif type(img) == Nifti1Pair and lext == '.nii':
        converted = Nifti1Image.from_image(img)
    elif type(img) == Nifti2Pair and lext == '.nii':
        converted = Nifti2Image.from_image(img)
    else:  # arbitrary conversion
        valid_klasses = [klass for klass in all_image_classes if lext in klass.valid_exts]
        if not valid_klasses:  # if list is empty
            raise ImageFileError(f'Cannot work out file type of "{filename}"')

        # Got a list of valid extensions, but that's no guarantee
        #   the file conversion will work. So, try each image
        #   in order...
        for klass in valid_klasses:
            try:
                converted = klass.from_image(img)
                break
            except Exception as e:
                err = e
        else:
            raise err

    converted.to_filename(filename, **kwargs)


@deprecate_with_version(
    'read_img_data deprecated. Please use ``img.dataobj.get_unscaled()`` instead.',
    '3.2',
    '5.0',
)
def read_img_data(img, prefer='scaled'):
    """Read data from image associated with files

    If you want unscaled data, please use ``img.dataobj.get_unscaled()``
    instead.  If you want scaled data, use ``img.get_fdata()`` (which will cache
    the loaded array) or ``np.array(img.dataobj)`` (which won't cache the
    array). If you want to load the data as for a modified header, save the
    image with the modified header, and reload.

    Parameters
    ----------
    img : ``SpatialImage``
       Image with valid image file in ``img.file_map``.  Unlike the
       ``img.get_fdata()`` method, this function returns the data read
       from the image file, as specified by the *current* image header
       and *current* image files.
    prefer : str, optional
       Can be 'scaled' - in which case we return the data with the
       scaling suggested by the format, or 'unscaled', in which case we
       return, if we can, the raw data from the image file, without the
       scaling applied.

    Returns
    -------
    arr : ndarray
       array as read from file, given parameters in header

    Notes
    -----
    Summary: please use the ``get_data`` method of `img` instead of this
    function unless you are sure what you are doing.

    In general, you will probably prefer ``prefer='scaled'``, because
    this gives the data as the image format expects to return it.

    Use `prefer` == 'unscaled' with care; the modified Analyze-type
    formats such as SPM formats, and nifti1, specify that the image data
    array is given by the raw data on disk, multiplied by a scalefactor
    and maybe with the addition of a constant.  This function, with
    ``unscaled`` returns the data on the disk, without these
    format-specific scalings applied.  Please use this function only if
    you absolutely need the unscaled data, and the magnitude of the
    data, as given by the scalefactor, is not relevant to your
    application.  The Analyze-type formats have a single scalefactor +/-
    offset per image on disk. If you do not care about the absolute
    values, and will be removing the mean from the data, then the
    unscaled values will have preserved intensity ratios compared to the
    mean-centered scaled data.  However, this is not necessarily true of
    other formats with more complicated scaling - such as MINC.
    """
    if prefer not in ('scaled', 'unscaled'):
        raise ValueError(f'Invalid string "{prefer}" for "prefer"')
    hdr = img.header
    if not hasattr(hdr, 'raw_data_from_fileobj'):
        # We can only do scaled
        if prefer == 'unscaled':
            raise ValueError('Can only do unscaled for Analyze types')
        return np.array(img.dataobj)
    # Analyze types
    img_fh = img.file_map['image']
    img_file_like = img_fh.filename if img_fh.fileobj is None else img_fh.fileobj
    if img_file_like is None:
        raise ImageFileError('No image file specified for this image')
    # Check the consumable values in the header
    hdr = img.header
    dao = img.dataobj
    default_offset = hdr.get_data_offset() == 0
    default_scaling = hdr.get_slope_inter() == (None, None)
    # If we have a proxy object and the header has any consumed fields, we load
    # the consumed values back from the proxy
    if is_proxy(dao) and (default_offset or default_scaling):
        hdr = hdr.copy()
        if default_offset and dao.offset != 0:
            hdr.set_data_offset(dao.offset)
        if default_scaling and (dao.slope, dao.inter) != (1, 0):
            hdr.set_slope_inter(dao.slope, dao.inter)
    with ImageOpener(img_file_like) as fileobj:
        if prefer == 'scaled':
            return hdr.data_from_fileobj(fileobj)
        return hdr.raw_data_from_fileobj(fileobj)
