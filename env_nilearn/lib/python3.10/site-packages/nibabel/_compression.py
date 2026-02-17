# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Constants and types for dealing transparently with compression"""

from __future__ import annotations

import bz2
import gzip
import typing as ty

from .optpkg import optional_package

if ty.TYPE_CHECKING:
    import io

    import indexed_gzip  # type: ignore[import]
    import pyzstd

    HAVE_INDEXED_GZIP = True
    HAVE_ZSTD = True
else:
    indexed_gzip, HAVE_INDEXED_GZIP, _ = optional_package('indexed_gzip')
    pyzstd, HAVE_ZSTD, _ = optional_package('pyzstd')


# Collections of types for isinstance or exception matching
COMPRESSED_FILE_LIKES: tuple[type[io.IOBase], ...] = (
    bz2.BZ2File,
    gzip.GzipFile,
)
COMPRESSION_ERRORS: tuple[type[BaseException], ...] = (
    OSError,  # BZ2File
    gzip.BadGzipFile,
)

if HAVE_INDEXED_GZIP:
    COMPRESSED_FILE_LIKES += (indexed_gzip.IndexedGzipFile,)
    COMPRESSION_ERRORS += (indexed_gzip.ZranError,)
    from indexed_gzip import IndexedGzipFile  # type: ignore[import-not-found]
else:
    IndexedGzipFile = gzip.GzipFile

if HAVE_ZSTD:
    COMPRESSED_FILE_LIKES += (pyzstd.ZstdFile,)
    COMPRESSION_ERRORS += (pyzstd.ZstdError,)
