# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for filename container"""

import pathlib

import pytest

from ..filename_parser import (
    TypesFilenamesError,
    _stringify_path,
    parse_filename,
    splitext_addext,
    types_filenames,
)


def test_filenames():
    types_exts = (('image', '.img'), ('header', '.hdr'))
    for t_fname in ('test.img', 'test.hdr', 'test', 'test.'):
        tfns = types_filenames(t_fname, types_exts)
        assert tfns == {'image': 'test.img', 'header': 'test.hdr'}
    # enforcing extensions raises an error for bad extension
    with pytest.raises(TypesFilenamesError):
        types_filenames('test.funny', types_exts)
    # If not enforcing extensions, it does the best job it can,
    # assuming the passed filename is for the first type (in this case
    # 'image')
    tfns = types_filenames('test.funny', types_exts, enforce_extensions=False)
    assert tfns == {'header': 'test.hdr', 'image': 'test.funny'}
    # .gz and .bz2 suffixes to extensions, by default, are removed
    # before extension checking etc, and then put back onto every
    # returned filename.
    tfns = types_filenames('test.img.gz', types_exts)
    assert tfns == {'header': 'test.hdr.gz', 'image': 'test.img.gz'}
    tfns = types_filenames('test.img.bz2', types_exts)
    assert tfns == {'header': 'test.hdr.bz2', 'image': 'test.img.bz2'}
    # of course, if we don't know about e.g. gz, and enforce_extensions
    # is on, we get an error
    with pytest.raises(TypesFilenamesError):
        types_filenames('test.img.gz', types_exts, ())
    # if we don't know about .gz extension, and not enforcing, then we
    # get something a bit odd
    tfns = types_filenames(
        'test.img.gz', types_exts, trailing_suffixes=(), enforce_extensions=False
    )
    assert tfns == {'header': 'test.img.hdr', 'image': 'test.img.gz'}
    # the suffixes we remove and replaces can be any suffixes.
    tfns = types_filenames('test.img.bzr', types_exts, ('.bzr',))
    assert tfns == {'header': 'test.hdr.bzr', 'image': 'test.img.bzr'}
    # If we specifically pass the remove / replace suffixes, then we
    # don't remove / replace the .gz and .bz2, unless they are passed
    # specifically.
    tfns = types_filenames(
        'test.img.bzr', types_exts, trailing_suffixes=('.bzr',), enforce_extensions=False
    )
    assert tfns == {'header': 'test.hdr.bzr', 'image': 'test.img.bzr'}
    # but, just .gz or .bz2 as extension gives an error, if enforcing is on
    with pytest.raises(TypesFilenamesError):
        types_filenames('test.gz', types_exts)
    with pytest.raises(TypesFilenamesError):
        types_filenames('test.bz2', types_exts)
    # if enforcing is off, it tries to work out what the other files
    # should be assuming the passed filename is of the first input type
    tfns = types_filenames('test.gz', types_exts, enforce_extensions=False)
    assert tfns == {'image': 'test.gz', 'header': 'test.hdr.gz'}
    # case (in)sensitivity, and effect of uppercase, lowercase
    tfns = types_filenames('test.IMG', types_exts)
    assert tfns == {'image': 'test.IMG', 'header': 'test.HDR'}
    tfns = types_filenames('test.img', (('image', '.IMG'), ('header', '.HDR')))
    assert tfns == {'header': 'test.hdr', 'image': 'test.img'}
    tfns = types_filenames('test.IMG.Gz', types_exts)
    assert tfns == {'image': 'test.IMG.Gz', 'header': 'test.HDR.Gz'}


def test_parse_filename():
    types_exts = (('t1', 'ext1'), ('t2', 'ext2'))
    exp_in_outs = (
        (('/path/fname.funny', ()), ('/path/fname', '.funny', None, None)),
        (('/path/fnameext2', ()), ('/path/fname', 'ext2', None, 't2')),
        (('/path/fnameext2', ('.gz',)), ('/path/fname', 'ext2', None, 't2')),
        (('/path/fnameext2.gz', ('.gz',)), ('/path/fname', 'ext2', '.gz', 't2')),
    )
    for inps, exps in exp_in_outs:
        pth, sufs = inps
        res = parse_filename(pth, types_exts, sufs)
        assert res == exps
        upth = pth.upper()
        uexps = (exps[0].upper(), exps[1].upper(), exps[2].upper() if exps[2] else None, exps[3])
        res = parse_filename(upth, types_exts, sufs)
        assert res == uexps
        # test case sensitivity
        res = parse_filename(
            '/path/fnameext2.GZ', types_exts, ('.gz',), False
        )  # case insensitive again
        assert res == ('/path/fname', 'ext2', '.GZ', 't2')
        res = parse_filename('/path/fnameext2.GZ', types_exts, ('.gz',), True)  # case sensitive
        assert res == ('/path/fnameext2', '.GZ', None, None)
        res = parse_filename('/path/fnameEXT2.gz', types_exts, ('.gz',), False)  # case insensitive
        assert res == ('/path/fname', 'EXT2', '.gz', 't2')
        res = parse_filename('/path/fnameEXT2.gz', types_exts, ('.gz',), True)  # case sensitive
        assert res == ('/path/fnameEXT2', '', '.gz', None)


def test_splitext_addext():
    res = splitext_addext('fname.ext.gz')
    assert res == ('fname', '.ext', '.gz')
    res = splitext_addext('fname.ext')
    assert res == ('fname', '.ext', '')
    res = splitext_addext('fname.ext.foo', ('.foo', '.bar'))
    assert res == ('fname', '.ext', '.foo')
    res = splitext_addext('fname.ext.FOO', ('.foo', '.bar'))
    assert res == ('fname', '.ext', '.FOO')
    # case sensitive
    res = splitext_addext('fname.ext.FOO', ('.foo', '.bar'), True)
    assert res == ('fname.ext', '.FOO', '')
    # edge cases
    res = splitext_addext('.nii')
    assert res == ('', '.nii', '')
    res = splitext_addext('...nii')
    assert res == ('..', '.nii', '')
    res = splitext_addext('.')
    assert res == ('.', '', '')
    res = splitext_addext('..')
    assert res == ('..', '', '')
    res = splitext_addext('...')
    assert res == ('...', '', '')


def test__stringify_path():
    res = _stringify_path('fname.ext.gz')
    assert res == 'fname.ext.gz'
    res = _stringify_path(pathlib.Path('fname.ext.gz'))
    assert res == 'fname.ext.gz'

    home = pathlib.Path.home().as_posix()
    res = _stringify_path(pathlib.Path('~/fname.ext.gz'))
    assert res == f'{home}/fname.ext.gz'

    res = _stringify_path(pathlib.Path('./fname.ext.gz'))
    assert res == 'fname.ext.gz'
    res = _stringify_path(pathlib.Path('../fname.ext.gz'))
    assert res == '../fname.ext.gz'
