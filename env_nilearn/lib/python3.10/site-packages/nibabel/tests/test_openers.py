# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test for openers module"""

import contextlib
import hashlib
import os
import time
import unittest
from gzip import GzipFile
from io import BytesIO, UnsupportedOperation
from unittest import mock

import pytest
from packaging.version import Version

from ..openers import HAVE_INDEXED_GZIP, BZ2File, DeterministicGzipFile, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory

pyzstd, HAVE_ZSTD, _ = optional_package('pyzstd')


class Lunk:
    # bare file-like for testing
    closed = False

    def __init__(self, message):
        self.message = message

    def write(self):
        pass

    def read(self, size=-1, /):
        return self.message


def test_Opener():
    # Test default mode is 'rb'
    fobj = Opener(__file__)
    assert fobj.mode == 'rb'
    fobj.close()
    # That it's a context manager
    with Opener(__file__) as fobj:
        assert fobj.mode == 'rb'
    # That we can set the mode
    with Opener(__file__, 'r') as fobj:
        assert fobj.mode == 'r'
    # with keyword arguments
    with Opener(__file__, mode='r') as fobj:
        assert fobj.mode == 'r'
    # fileobj returns fileobj passed through
    message = b"Wine?  Wouldn't you?"
    for obj in (BytesIO(message), Lunk(message)):
        with Opener(obj) as fobj:
            assert fobj.read() == message
        # Which does not close the object
        assert not obj.closed
        # mode is gently ignored
        fobj = Opener(obj, mode='r')


def test_Opener_various():
    # Check we can do all sorts of files here
    message = b'Oh what a giveaway'
    bz2_fileno = hasattr(BZ2File, 'fileno')
    if HAVE_INDEXED_GZIP:
        import indexed_gzip as igzip
    with InTemporaryDirectory():
        sobj = BytesIO()
        files_to_test = ['test.txt', 'test.txt.gz', 'test.txt.bz2', sobj]
        if HAVE_ZSTD:
            files_to_test += ['test.txt.zst']
        for input in files_to_test:
            with Opener(input, 'wb') as fobj:
                fobj.write(message)
                assert fobj.tell() == len(message)
            if input == sobj:
                input.seek(0)
            with Opener(input, 'rb') as fobj:
                message_back = fobj.read()
                assert message == message_back
                if input == sobj:
                    # Fileno is unsupported for BytesIO
                    with pytest.raises(UnsupportedOperation):
                        fobj.fileno()
                elif input.endswith('.bz2') and not bz2_fileno:
                    with pytest.raises(AttributeError):
                        fobj.fileno()
                # indexed gzip is used by default, and drops file
                # handles by default, so we don't have a fileno.
                elif (
                    input.endswith('gz')
                    and HAVE_INDEXED_GZIP
                    and Version(igzip.__version__) >= Version('0.7.0')
                ):
                    with pytest.raises(igzip.NoHandleError):
                        fobj.fileno()
                else:
                    # Just check there is a fileno
                    assert fobj.fileno() != 0


class MockIndexedGzipFile(GzipFile):
    def __init__(self, *args, **kwargs):
        self._drop_handles = kwargs.pop('drop_handles', False)
        super().__init__(*args, **kwargs)


@contextlib.contextmanager
def patch_indexed_gzip(state):
    # Make it look like we do (state==True) or do not (state==False) have
    # the indexed gzip module.
    if state:
        values = (True, MockIndexedGzipFile)
    else:
        values = (False, GzipFile)
    with (
        mock.patch('nibabel.openers.HAVE_INDEXED_GZIP', values[0]),
        mock.patch('nibabel.openers.IndexedGzipFile', values[1], create=True),
    ):
        yield


def test_Opener_gzip_type(tmp_path):
    # Test that GzipFile or IndexedGzipFile are used as appropriate

    data = b'this is some test data'
    fname = tmp_path / 'test.gz'

    # make some test data
    with GzipFile(fname, mode='wb') as f:
        f.write(data)

    # Each test is specified by a tuple containing:
    #   (indexed_gzip present, Opener kwargs, expected file type)
    tests = [
        (False, {'mode': 'rb', 'keep_open': True}, GzipFile),
        (False, {'mode': 'rb', 'keep_open': False}, GzipFile),
        (False, {'mode': 'wb', 'keep_open': True}, GzipFile),
        (False, {'mode': 'wb', 'keep_open': False}, GzipFile),
        (True, {'mode': 'rb', 'keep_open': True}, MockIndexedGzipFile),
        (True, {'mode': 'rb', 'keep_open': False}, MockIndexedGzipFile),
        (True, {'mode': 'wb', 'keep_open': True}, GzipFile),
        (True, {'mode': 'wb', 'keep_open': False}, GzipFile),
    ]

    for test in tests:
        igzip_present, kwargs, expected = test
        with patch_indexed_gzip(igzip_present):
            opener = Opener(fname, **kwargs)
            assert isinstance(opener.fobj, expected)
            # Explicit close to appease Windows
            del opener


class TestImageOpener(unittest.TestCase):
    def test_vanilla(self):
        # Test that ImageOpener does add '.mgz' as gzipped file type
        with InTemporaryDirectory():
            with ImageOpener('test.gz', 'w') as fobj:
                assert hasattr(fobj.fobj, 'compress')
            with ImageOpener('test.mgz', 'w') as fobj:
                assert hasattr(fobj.fobj, 'compress')

    @mock.patch.dict('nibabel.openers.ImageOpener.compress_ext_map')
    def test_new_association(self):
        def file_opener(fileish, mode):
            return open(fileish, mode)

        # Add the association
        n_associations = len(ImageOpener.compress_ext_map)
        ImageOpener.compress_ext_map['.foo'] = (file_opener, ('mode',))
        assert n_associations + 1 == len(ImageOpener.compress_ext_map)
        assert '.foo' in ImageOpener.compress_ext_map

        with InTemporaryDirectory():
            with ImageOpener('test.foo', 'w'):
                pass
            assert os.path.exists('test.foo')

        # Check this doesn't add anything to parent
        assert '.foo' not in Opener.compress_ext_map


def test_file_like_wrapper():
    # Test wrapper using BytesIO (full API)
    message = b'History of the nude in'
    sobj = BytesIO()
    fobj = Opener(sobj)
    assert fobj.tell() == 0
    fobj.write(message)
    assert fobj.tell() == len(message)
    fobj.seek(0)
    assert fobj.tell() == 0
    assert fobj.read(6) == message[:6]
    assert not fobj.closed
    fobj.close()
    assert fobj.closed
    # Added the fileobj name
    assert fobj.name is None


def test_compressionlevel():
    # Check default and set compression level
    with open(__file__, 'rb') as fobj:
        my_self = fobj.read()
    # bzip2 needs a fairly large file to show differences in compression level
    many_selves = my_self * 50
    # Test we can set default compression at class level

    class MyOpener(Opener):
        default_compresslevel = 5

    with InTemporaryDirectory():
        for ext in ('gz', 'bz2', 'GZ', 'gZ', 'BZ2', 'Bz2'):
            for opener, default_val in ((Opener, 1), (MyOpener, 5)):
                sizes = {}
                for compresslevel in ('default', 1, 5):
                    fname = 'test.' + ext
                    kwargs = {'mode': 'wb'}
                    if compresslevel != 'default':
                        kwargs['compresslevel'] = compresslevel
                    with opener(fname, **kwargs) as fobj:
                        fobj.write(many_selves)
                    with open(fname, 'rb') as fobj:
                        my_selves_smaller = fobj.read()
                    sizes[compresslevel] = len(my_selves_smaller)
                assert sizes['default'] == sizes[default_val]
                assert sizes[1] > sizes[5]


def test_compressed_ext_case():
    # Test openers usually ignore case for compressed exts
    contents = b'palindrome of Bolton is notlob'

    class StrictOpener(Opener):
        compress_ext_icase = False

    exts = ('gz', 'bz2', 'GZ', 'gZ', 'BZ2', 'Bz2')
    if HAVE_ZSTD:
        exts += ('zst', 'ZST', 'Zst')
    with InTemporaryDirectory():
        # Make a basic file to check type later
        with open(__file__, 'rb') as a_file:
            file_class = type(a_file)
        for ext in exts:
            fname = 'test.' + ext
            with Opener(fname, 'wb') as fobj:
                fobj.write(contents)
            with Opener(fname, 'rb') as fobj:
                assert fobj.read() == contents
            os.unlink(fname)
            with StrictOpener(fname, 'wb') as fobj:
                fobj.write(contents)
            with StrictOpener(fname, 'rb') as fobj:
                assert fobj.read() == contents
            lext = ext.lower()
            if lext != ext:  # extension should not be recognized -> file
                assert isinstance(fobj.fobj, file_class)
            elif lext == 'gz':
                try:
                    from ..openers import IndexedGzipFile
                except ImportError:
                    IndexedGzipFile = GzipFile
                assert isinstance(fobj.fobj, (GzipFile, IndexedGzipFile))
            elif lext == 'zst':
                assert isinstance(fobj.fobj, pyzstd.ZstdFile)
            else:
                assert isinstance(fobj.fobj, BZ2File)


def test_name():
    # The wrapper gives everything a name, maybe None
    sobj = BytesIO()
    lunk = Lunk('in ART')
    with InTemporaryDirectory():
        files_to_test = ['test.txt', 'test.txt.gz', 'test.txt.bz2', sobj, lunk]
        if HAVE_ZSTD:
            files_to_test += ['test.txt.zst']
        for input in files_to_test:
            exp_name = input if type(input) == str else None
            with Opener(input, 'wb') as fobj:
                assert fobj.name == exp_name


def test_set_extensions():
    # Test that we can add extensions that are compressed
    with InTemporaryDirectory():
        with Opener('test.gz', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')
        with Opener('test.glrph', 'w') as fobj:
            assert not hasattr(fobj.fobj, 'compress')

        class MyOpener(Opener):
            compress_ext_map = Opener.compress_ext_map.copy()
            compress_ext_map['.glrph'] = Opener.gz_def

        with MyOpener('test.glrph', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')


def test_close_if_mine():
    # Test that we close the file iff we opened it
    with InTemporaryDirectory():
        sobj = BytesIO()
        lunk = Lunk('')
        for input in ('test.txt', 'test.txt.gz', 'test.txt.bz2', sobj, lunk):
            fobj = Opener(input, 'wb')
            # gzip objects have no 'closed' attribute
            has_closed = hasattr(fobj.fobj, 'closed')
            if has_closed:
                assert not fobj.closed
            fobj.close_if_mine()
            is_str = type(input) is str
            if has_closed:
                assert fobj.closed == is_str


def test_iter():
    # Check we can iterate over lines, if the underlying file object allows it
    lines = """On the
blue ridged mountains
of
virginia
""".splitlines()
    with InTemporaryDirectory():
        sobj = BytesIO()
        files_to_test = [
            ('test.txt', True),
            ('test.txt.gz', False),
            ('test.txt.bz2', False),
            (sobj, True),
        ]
        if HAVE_ZSTD:
            files_to_test += [('test.txt.zst', False)]
        for input, does_t in files_to_test:
            with Opener(input, 'wb') as fobj:
                for line in lines:
                    fobj.write(str.encode(line + os.linesep))
            with Opener(input, 'rb') as fobj:
                for back_line, line in zip(fobj, lines):
                    assert back_line.decode().rstrip() == line
            if not does_t:
                continue
            with Opener(input, 'rt') as fobj:
                for back_line, line in zip(fobj, lines):
                    assert back_line.rstrip() == line
        lobj = Opener(Lunk(''))
        with pytest.raises(TypeError):
            list(lobj)


def md5sum(fname):
    with open(fname, 'rb') as fobj:
        return hashlib.md5(fobj.read()).hexdigest()


def test_DeterministicGzipFile():
    with InTemporaryDirectory():
        msg = b"Hello, I'd like to have an argument."

        # No filename, no mtime
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', fileobj=fobj, mtime=0) as gzobj:
                gzobj.write(msg)
        anon_chksum = md5sum('ref.gz')

        with DeterministicGzipFile('default.gz', 'wb') as fobj:
            internal_fobj = fobj.myfileobj
            fobj.write(msg)
        # Check that myfileobj is being closed by GzipFile.close()
        # This is in case GzipFile changes its internal implementation
        assert internal_fobj.closed

        assert md5sum('default.gz') == anon_chksum

        # No filename, current mtime
        now = time.time()
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', fileobj=fobj, mtime=now) as gzobj:
                gzobj.write(msg)
        now_chksum = md5sum('ref.gz')

        with DeterministicGzipFile('now.gz', 'wb', mtime=now) as fobj:
            fobj.write(msg)

        assert md5sum('now.gz') == now_chksum

        # Change in default behavior
        with mock.patch('time.time') as t:
            t.return_value = now

            # GzipFile will use time.time()
            with open('ref.gz', 'wb') as fobj:
                with GzipFile(filename='', mode='wb', fileobj=fobj) as gzobj:
                    gzobj.write(msg)
            assert md5sum('ref.gz') == now_chksum

            # DeterministicGzipFile will use 0
            with DeterministicGzipFile('now.gz', 'wb') as fobj:
                fobj.write(msg)
            assert md5sum('now.gz') == anon_chksum

        # GzipFile is filename dependent, DeterministicGzipFile is independent
        with GzipFile('filenameA.gz', mode='wb', mtime=0) as gzobj:
            gzobj.write(msg)
        fnameA_chksum = md5sum('filenameA.gz')
        assert fnameA_chksum != anon_chksum

        with DeterministicGzipFile('filenameA.gz', 'wb') as fobj:
            fobj.write(msg)

        # But the contents are the same with different filenames
        assert md5sum('filenameA.gz') == anon_chksum


def test_DeterministicGzipFile_fileobj():
    with InTemporaryDirectory():
        msg = b"Hello, I'd like to have an argument."
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', fileobj=fobj, mtime=0) as gzobj:
                gzobj.write(msg)
        ref_chksum = md5sum('ref.gz')

        with open('test.gz', 'wb') as fobj:
            with DeterministicGzipFile(filename='', mode='wb', fileobj=fobj) as gzobj:
                gzobj.write(msg)
        assert md5sum('test.gz') == ref_chksum

        with open('test.gz', 'wb') as fobj:
            with DeterministicGzipFile(fileobj=fobj, mode='wb') as gzobj:
                gzobj.write(msg)
        assert md5sum('test.gz') == ref_chksum

        with open('test.gz', 'wb') as fobj:
            with DeterministicGzipFile(filename='test.gz', mode='wb', fileobj=fobj) as gzobj:
                gzobj.write(msg)
        assert md5sum('test.gz') == ref_chksum


def test_bitwise_determinism():
    with InTemporaryDirectory():
        msg = b"Hello, I'd like to have an argument."
        # Canonical reference: No filename, no mtime
        # Use default compresslevel
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', compresslevel=1, fileobj=fobj, mtime=0) as gzobj:
                gzobj.write(msg)
        anon_chksum = md5sum('ref.gz')

        # Different times, different filenames
        now = time.time()
        with mock.patch('time.time') as t:
            t.return_value = now
            with Opener('a.gz', 'wb') as fobj:
                fobj.write(msg)
            t.return_value = now + 1
            with Opener('b.gz', 'wb') as fobj:
                fobj.write(msg)

        assert md5sum('a.gz') == anon_chksum
        assert md5sum('b.gz') == anon_chksum

        # Users can still set mtime, but filenames will not be embedded
        with Opener('filenameA.gz', 'wb', mtime=0xCAFE10C0) as fobj:
            fobj.write(msg)
        with Opener('filenameB.gz', 'wb', mtime=0xCAFE10C0) as fobj:
            fobj.write(msg)
        fnameA_chksum = md5sum('filenameA.gz')
        fnameB_chksum = md5sum('filenameB.gz')
        assert fnameA_chksum == fnameB_chksum != anon_chksum
