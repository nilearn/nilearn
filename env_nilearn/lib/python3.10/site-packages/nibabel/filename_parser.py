# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Create filename pairs, triplets etc, with expected extensions"""

from __future__ import annotations

import os
import pathlib
import typing as ty

if ty.TYPE_CHECKING:
    FileSpec = str | os.PathLike[str]
    ExtensionSpec = tuple[str, str | None]


class TypesFilenamesError(Exception):
    pass


def _stringify_path(filepath_or_buffer: FileSpec) -> str:
    """Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : str or os.PathLike

    Returns
    -------
    str_filepath_or_buffer : str

    Notes
    -----
    Adapted from:
    https://github.com/pandas-dev/pandas/blob/325dd68/pandas/io/common.py#L131-L160
    """
    return pathlib.Path(filepath_or_buffer).expanduser().as_posix()


def types_filenames(
    template_fname: FileSpec,
    types_exts: ty.Sequence[ExtensionSpec],
    trailing_suffixes: ty.Sequence[str] = ('.gz', '.bz2'),
    enforce_extensions: bool = True,
    match_case: bool = False,
) -> dict[str, str]:
    """Return filenames with standard extensions from template name

    The typical case is returning image and header filenames for an
    Analyze image, that expects an 'image' file type with extension ``.img``,
    and a 'header' file type, with extension ``.hdr``.

    Parameters
    ----------
    template_fname : str or os.PathLike
       template filename from which to construct output dict of
       filenames, with given `types_exts` type to extension mapping.  If
       ``self.enforce_extensions`` is True, then filename must have one
       of the defined extensions from the types list.  If
       ``self.enforce_extensions`` is False, then the other filenames
       are guessed at by adding extensions to the base filename.
       Ignored suffixes (from `trailing_suffixes`) append themselves to
       the end of all the filenames.
    types_exts : sequence of sequences
       sequence of (name, extension) str sequences defining type to
       extension mapping.
    trailing_suffixes : sequence of strings, optional
        suffixes that should be ignored when looking for
        extensions - default is ``('.gz', '.bz2')``
    enforce_extensions : {True, False}, optional
        If True, raise an error when attempting to set value to
        type which has the wrong extension
    match_case : bool, optional
       If True, match case of extensions and trailing suffixes when
       searching in `template_fname`, otherwise do case-insensitive
       match.

    Returns
    -------
    types_fnames : dict
       dict with types as keys, and generated filenames as values.  The
       types are given by the first elements of the tuples in
       `types_exts`.

    Examples
    --------
    >>> types_exts = (('t1','.ext1'),('t2', '.ext2'))
    >>> tfns = types_filenames('/path/test.ext1', types_exts)
    >>> tfns == {'t1': '/path/test.ext1', 't2': '/path/test.ext2'}
    True

    Bare file roots without extensions get them added

    >>> tfns = types_filenames('/path/test', types_exts)
    >>> tfns == {'t1': '/path/test.ext1', 't2': '/path/test.ext2'}
    True

    With enforce_extensions == False, allow first type to have any
    extension.

    >>> tfns = types_filenames('/path/test.funny', types_exts,
    ...                        enforce_extensions=False)
    >>> tfns == {'t1': '/path/test.funny', 't2': '/path/test.ext2'}
    True
    """
    template_fname = _stringify_path(template_fname)
    if not isinstance(template_fname, str):
        raise TypesFilenamesError('Need file name as input to set_filenames')
    if template_fname.endswith('.'):
        template_fname = template_fname[:-1]
    filename, found_ext, ignored, guessed_name = parse_filename(
        template_fname, types_exts, trailing_suffixes, match_case
    )
    # Flag cases where we just set the input name directly
    direct_set_name = None
    if enforce_extensions:
        if guessed_name is None:
            # no match - maybe there was no extension atall or the
            # wrong extension. In either case we raise an error
            if found_ext:
                # an extension, but the wrong one
                raise TypesFilenamesError(
                    f'File extension "{found_ext}" was not in '
                    f'expected list: {[e for t, e in types_exts]}'
                )
            elif ignored:  # there was no extension, but an ignored suffix
                # This is a special case like 'test.gz' (where .gz
                # is ignored). It's confusing to change
                # this to test.img.gz, or test.gz.img, so error
                raise TypesFilenamesError(f'Confusing ignored suffix {ignored} without extension')
        # if we've got to here, we have a guessed name and a found
        # extension.
    else:  # not enforcing extensions. If there's an extension, we set the
        # filename directly from input, for the first types_exts type
        # only.  Also, if there was no extension, but an ignored suffix
        # ('test.gz' type case), we set the filename directly.
        # Otherwise (no extension, no ignored suffix), we stay with the
        # default, which is to add the default extensions according to
        # type.
        if found_ext or ignored:
            direct_set_name = types_exts[0][0]
    tfns = {}
    # now we have an extension case matching problem.  For example, if
    # we've found .IMG as the extension, we want .HDR as the matching
    # one.  Let's only do this when the extension is all upper or all
    # lower case.
    proc_ext: ty.Callable[[str], str] = lambda s: s
    if found_ext:
        if found_ext == found_ext.upper():
            proc_ext = str.upper
        elif found_ext == found_ext.lower():
            proc_ext = str.lower
    for name, ext in types_exts:
        if name == direct_set_name:
            tfns[name] = template_fname
            continue
        fname = filename
        if ext:
            fname += proc_ext(ext)
        if ignored:
            fname += ignored
        tfns[name] = fname
    return tfns


def parse_filename(
    filename: FileSpec,
    types_exts: ty.Sequence[ExtensionSpec],
    trailing_suffixes: ty.Sequence[str],
    match_case: bool = False,
) -> tuple[str, str, str | None, str | None]:
    """Split filename into fileroot, extension, trailing suffix; guess type.

    Parameters
    ----------
    filename : str or os.PathLike
       filename in which to search for type extensions
    types_exts : sequence of sequences
       sequence of (name, extension) str sequences defining type to
       extension mapping.
    trailing_suffixes : sequence of strings
        suffixes that should be ignored when looking for
        extensions
    match_case : bool, optional
       If True, match case of extensions and trailing suffixes when
       searching in `filename`, otherwise do case-insensitive match.

    Returns
    -------
    pth : str
       path with any matching extensions or trailing suffixes removed
    ext : str
       If there were any matching extensions, in `types_exts` return
       that; otherwise return extension derived from
       ``os.path.splitext``.
    trailing : str
       If there were any matching `trailing_suffixes` return that
       matching suffix, otherwise ''
    guessed_type : str
       If we found a matching extension in `types_exts` return the
       corresponding ``type``

    Examples
    --------
    >>> types_exts = (('t1', 'ext1'),('t2', 'ext2'))
    >>> parse_filename('/path/fname.funny', types_exts, ())
    ('/path/fname', '.funny', None, None)
    >>> parse_filename('/path/fnameext2', types_exts, ())
    ('/path/fname', 'ext2', None, 't2')
    >>> parse_filename('/path/fnameext2', types_exts, ('.gz',))
    ('/path/fname', 'ext2', None, 't2')
    >>> parse_filename('/path/fnameext2.gz', types_exts, ('.gz',))
    ('/path/fname', 'ext2', '.gz', 't2')
    """
    filename = _stringify_path(filename)

    ignored = None
    if match_case:
        endswith = _endswith
    else:
        endswith = _iendswith
    for ext in trailing_suffixes:
        if endswith(filename, ext):
            extpos = -len(ext)
            ignored = filename[extpos:]
            filename = filename[:extpos]
            break
    guessed_name = None
    found_ext = None
    for name, type_ext in types_exts:
        if type_ext and endswith(filename, type_ext):
            extpos = -len(type_ext)
            found_ext = filename[extpos:]
            filename = filename[:extpos]
            guessed_name = name
            break
    else:
        filename, found_ext = os.path.splitext(filename)
    return (filename, found_ext, ignored, guessed_name)


def _endswith(whole: str, end: str) -> bool:
    return whole.endswith(end)


def _iendswith(whole: str, end: str) -> bool:
    return whole.lower().endswith(end.lower())


def splitext_addext(
    filename: FileSpec,
    addexts: ty.Sequence[str] = ('.gz', '.bz2', '.zst'),
    match_case: bool = False,
) -> tuple[str, str, str]:
    """Split ``/pth/fname.ext.gz`` into ``/pth/fname, .ext, .gz``

    where ``.gz`` may be any of passed `addext` trailing suffixes.

    Parameters
    ----------
    filename : str or os.PathLike
       filename that may end in any or none of `addexts`
    match_case : bool, optional
       If True, match case of `addexts` and `filename`, otherwise do
       case-insensitive match.

    Returns
    -------
    froot : str
       Root of filename - e.g. ``/pth/fname`` in example above
    ext : str
       Extension, where extension is not in `addexts` - e.g. ``.ext`` in
       example above
    addext : str
       Any suffixes appearing in `addext` occurring at end of filename

    Examples
    --------
    >>> splitext_addext('fname.ext.gz')
    ('fname', '.ext', '.gz')
    >>> splitext_addext('fname.ext')
    ('fname', '.ext', '')
    >>> splitext_addext('fname.ext.foo', ('.foo', '.bar'))
    ('fname', '.ext', '.foo')
    """
    filename = _stringify_path(filename)

    if match_case:
        endswith = _endswith
    else:
        endswith = _iendswith
    for ext in addexts:
        if endswith(filename, ext):
            extpos = -len(ext)
            filename, addext = filename[:extpos], filename[extpos:]
            break
    else:
        addext = ''
    # os.path.splitext() behaves unexpectedly when filename starts with '.'
    extpos = filename.rfind('.')
    if extpos < 0 or filename.strip('.') == '':
        root, ext = filename, ''
    else:
        root, ext = filename[:extpos], filename[extpos:]
    return (root, ext, addext)
