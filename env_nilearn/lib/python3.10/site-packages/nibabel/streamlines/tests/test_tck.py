import os
import unittest
from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ...testing import data_path, error_warnings
from ..array_sequence import ArraySequence
from ..tck import TckFile
from ..tractogram import Tractogram
from ..tractogram_file import DataError, HeaderError, HeaderWarning
from .test_tractogram import assert_tractogram_equal

DATA = {}


def setup_module():
    global DATA

    DATA['empty_tck_fname'] = pjoin(data_path, 'empty.tck')
    DATA['no_magic_number_tck_fname'] = pjoin(data_path, 'no_magic_number.tck')
    DATA['no_header_end_tck_fname'] = pjoin(data_path, 'no_header_end.tck')
    DATA['no_header_end_eof_tck_fname'] = pjoin(data_path, 'no_header_end_eof.tck')
    # simple.tck contains only streamlines
    DATA['simple_tck_fname'] = pjoin(data_path, 'simple.tck')
    DATA['simple_tck_big_endian_fname'] = pjoin(data_path, 'simple_big_endian.tck')
    # standard.tck contains only streamlines
    DATA['standard_tck_fname'] = pjoin(data_path, 'standard.tck')
    DATA['matlab_nan_tck_fname'] = pjoin(data_path, 'matlab_nan.tck')
    DATA['multiline_header_fname'] = pjoin(data_path, 'multiline_header_field.tck')

    DATA['streamlines'] = [
        np.arange(1 * 3, dtype='f4').reshape((1, 3)),
        np.arange(2 * 3, dtype='f4').reshape((2, 3)),
        np.arange(5 * 3, dtype='f4').reshape((5, 3)),
    ]

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))


class TestTCK(unittest.TestCase):
    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['empty_tck_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(tck.tractogram, DATA['empty_tractogram'])

    def test_load_no_magic_number_file(self):
        for lazy_load in [False, True]:
            with pytest.raises(HeaderError):
                TckFile.load(DATA['no_magic_number_tck_fname'], lazy_load=lazy_load)

    def test_load_no_header_end_file(self):
        for lazy_load in [False, True]:
            with pytest.raises(HeaderError):
                TckFile.load(DATA['no_header_end_tck_fname'], lazy_load=lazy_load)

    def test_load_no_header_end_eof_file(self):
        for lazy_load in [False, True]:
            with pytest.raises(HeaderError):
                TckFile.load(DATA['no_header_end_eof_tck_fname'], lazy_load=lazy_load)

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])

        # Force TCK loading to use buffering.
        buffer_size = 1.0 / 1024**2  # 1 bytes
        hdr = TckFile._read_header(DATA['simple_tck_fname'])
        tck_reader = TckFile._read(DATA['simple_tck_fname'], hdr, buffer_size)
        streamlines = ArraySequence(tck_reader)
        tractogram = Tractogram(streamlines)
        tractogram.affine_to_rasmm = np.eye(4)
        tck = TckFile(tractogram, header=hdr)
        assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])

    def test_load_matlab_nan_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['matlab_nan_tck_fname'], lazy_load=lazy_load)
            streamlines = list(tck.tractogram.streamlines)
            assert len(streamlines) == 1
            assert streamlines[0].shape == (108, 3)

    def test_load_multiline_header_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['multiline_header_fname'], lazy_load=lazy_load)
            streamlines = list(tck.tractogram.streamlines)
            assert len(tck.header['command_history'].splitlines()) == 3
            assert len(streamlines) == 1
            assert streamlines[0].shape == (253, 3)

    def test_writeable_data(self):
        data = DATA['simple_tractogram']
        for key in ('simple_tck_fname', 'simple_tck_big_endian_fname'):
            for lazy_load in [False, True]:
                tck = TckFile.load(DATA[key], lazy_load=lazy_load)
                for actual, expected_tgi in zip(tck.streamlines, data):
                    assert_array_equal(actual, expected_tgi.streamline)
                    # Test we can write to arrays
                    assert actual.flags.writeable
                    actual[0, 0] = 99

    def test_load_simple_file_in_big_endian(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_big_endian_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])
            assert tck.header['datatype'] == 'Float32BE'

    def test_load_file_with_wrong_information(self):
        tck_file = open(DATA['simple_tck_fname'], 'rb').read()

        # Simulate a TCK file where `datatype` has not the right endianness.
        new_tck_file = tck_file.replace(b'Float32LE', b'Float32BE')

        with pytest.raises(DataError):
            TckFile.load(BytesIO(new_tck_file))

        # Simulate a TCK file with unsupported `datatype`.
        new_tck_file = tck_file.replace(b'Float32LE', b'int32')
        with pytest.raises(HeaderError):
            TckFile.load(BytesIO(new_tck_file))

        # Simulate a TCK file with no `datatype` field.
        new_tck_file = tck_file.replace(b'datatype: Float32LE\n', b'')
        # Need to adjust data offset.
        new_tck_file = new_tck_file.replace(b'file: . 67\n', b'file: . 47\n')
        with pytest.warns(HeaderWarning, match="Missing 'datatype'"):
            tck = TckFile.load(BytesIO(new_tck_file))
        assert_array_equal(tck.header['datatype'], 'Float32LE')

        # Simulate a TCK file with no `file` field.
        new_tck_file = tck_file.replace(b'\nfile: . 67', b'')
        with pytest.warns(HeaderWarning, match="Missing 'file'"):
            tck = TckFile.load(BytesIO(new_tck_file))
        assert_array_equal(tck.header['file'], '. 56')

        # Simulate a TCK file with `file` field pointing to another file.
        new_tck_file = tck_file.replace(b'file: . 67\n', b'file: dummy.mat 75\n')
        with pytest.raises(HeaderError):
            TckFile.load(BytesIO(new_tck_file))

        # Simulate a TCK file which is missing a streamline delimiter.
        eos = TckFile.FIBER_DELIMITER.tobytes()
        eof = TckFile.EOF_DELIMITER.tobytes()
        new_tck_file = tck_file[: -(len(eos) + len(eof))] + tck_file[-len(eof) :]

        # Force TCK loading to use buffering.
        buffer_size = 1.0 / 1024**2  # 1 bytes
        hdr = TckFile._read_header(BytesIO(new_tck_file))
        tck_reader = TckFile._read(BytesIO(new_tck_file), hdr, buffer_size)
        with pytest.raises(DataError):
            list(tck_reader)

        # Simulate a TCK file which is missing the end-of-file delimiter.
        new_tck_file = tck_file[: -len(eof)]
        with pytest.raises(DataError):
            TckFile.load(BytesIO(new_tck_file))

    def test_write_empty_file(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))

        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        new_tck = TckFile.load(tck_file)
        assert_tractogram_equal(new_tck.tractogram, tractogram)

        new_tck_orig = TckFile.load(DATA['empty_tck_fname'])
        assert_tractogram_equal(new_tck.tractogram, new_tck_orig.tractogram)

        tck_file.seek(0, os.SEEK_SET)
        assert tck_file.read() == open(DATA['empty_tck_fname'], 'rb').read()

    def test_write_simple_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))

        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        new_tck = TckFile.load(tck_file)
        assert_tractogram_equal(new_tck.tractogram, tractogram)

        new_tck_orig = TckFile.load(DATA['simple_tck_fname'])
        assert_tractogram_equal(new_tck.tractogram, new_tck_orig.tractogram)

        tck_file.seek(0, os.SEEK_SET)
        assert tck_file.read() == open(DATA['simple_tck_fname'], 'rb').read()

        # TCK file containing not well formatted entries in its header.
        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.header['new_entry'] = 'val:ue'  # : not allowed
        with pytest.raises(HeaderError):
            tck.save(tck_file)

    def test_write_bigheader_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))

        # Offset is represented by 2 characters.
        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.header['new_entry'] = ' ' * 20
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        new_tck = TckFile.load(tck_file)
        assert_tractogram_equal(new_tck.tractogram, tractogram)
        assert new_tck.header['_offset_data'] == 99

        # We made the jump, now offset is represented by 3 characters
        # and we need to adjust the offset!
        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.header['new_entry'] = ' ' * 21
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        new_tck = TckFile.load(tck_file)
        assert_tractogram_equal(new_tck.tractogram, tractogram)
        assert new_tck.header['_offset_data'] == 101

    def test_load_write_file(self):
        for fname in [DATA['empty_tck_fname'], DATA['simple_tck_fname']]:
            for lazy_load in [False, True]:
                tck = TckFile.load(fname, lazy_load=lazy_load)
                tck_file = BytesIO()
                tck.save(tck_file)

                loaded_tck = TckFile.load(fname, lazy_load=False)
                assert_tractogram_equal(loaded_tck.tractogram, tck.tractogram)

                # Check that the written file is the same as the one read.
                tck_file.seek(0, os.SEEK_SET)
                assert tck_file.read() == open(fname, 'rb').read()

        # Save tractogram that has an affine_to_rasmm.
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_fname'], lazy_load=lazy_load)
            affine = np.eye(4)
            affine[0, 0] *= -1  # Flip in X
            tractogram = Tractogram(tck.streamlines, affine_to_rasmm=affine)

            new_tck = TckFile(tractogram, tck.header)
            tck_file = BytesIO()
            new_tck.save(tck_file)
            tck_file.seek(0, os.SEEK_SET)

            loaded_tck = TckFile.load(tck_file, lazy_load=False)
            assert_tractogram_equal(loaded_tck.tractogram, tractogram.to_world(lazy=True))

    def test_str(self):
        tck = TckFile.load(DATA['simple_tck_fname'])
        str(tck)  # Simply test it's not failing when called.
