import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
    TrkFile,
    decode_value_from_name,
    encode_value_in_name,
    get_affine_trackvis_to_rasmm,
)
from .test_tractogram import assert_tractogram_equal

DATA = {}


def setup_module():
    global DATA

    DATA['empty_trk_fname'] = pjoin(data_path, 'empty.trk')
    # simple.trk contains only streamlines
    DATA['simple_trk_fname'] = pjoin(data_path, 'simple.trk')
    # standard.trk contains only streamlines
    DATA['standard_trk_fname'] = pjoin(data_path, 'standard.trk')
    # standard.LPS.trk contains only streamlines
    DATA['standard_LPS_trk_fname'] = pjoin(data_path, 'standard.LPS.trk')

    # complex.trk contains streamlines, scalars and properties
    DATA['complex_trk_fname'] = pjoin(data_path, 'complex.trk')
    DATA['complex_trk_big_endian_fname'] = pjoin(data_path, 'complex_big_endian.trk')

    DATA['streamlines'] = [
        np.arange(1 * 3, dtype='f4').reshape((1, 3)),
        np.arange(2 * 3, dtype='f4').reshape((2, 3)),
        np.arange(5 * 3, dtype='f4').reshape((5, 3)),
    ]

    DATA['fa'] = [
        np.array([[0.2]], dtype='f4'),
        np.array([[0.3], [0.4]], dtype='f4'),
        np.array([[0.5], [0.6], [0.6], [0.7], [0.8]], dtype='f4'),
    ]

    DATA['colors'] = [
        np.array([(1, 0, 0)] * 1, dtype='f4'),
        np.array([(0, 1, 0)] * 2, dtype='f4'),
        np.array([(0, 0, 1)] * 5, dtype='f4'),
    ]

    DATA['mean_curvature'] = [
        np.array([1.11], dtype='f4'),
        np.array([2.11], dtype='f4'),
        np.array([3.11], dtype='f4'),
    ]

    DATA['mean_torsion'] = [
        np.array([1.22], dtype='f4'),
        np.array([2.22], dtype='f4'),
        np.array([3.22], dtype='f4'),
    ]

    DATA['mean_colors'] = [
        np.array([1, 0, 0], dtype='f4'),
        np.array([0, 1, 0], dtype='f4'),
        np.array([0, 0, 1], dtype='f4'),
    ]

    DATA['data_per_point'] = {'colors': DATA['colors'], 'fa': DATA['fa']}
    DATA['data_per_streamline'] = {
        'mean_curvature': DATA['mean_curvature'],
        'mean_torsion': DATA['mean_torsion'],
        'mean_colors': DATA['mean_colors'],
    }

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
    DATA['complex_tractogram'] = Tractogram(
        DATA['streamlines'],
        DATA['data_per_streamline'],
        DATA['data_per_point'],
        affine_to_rasmm=np.eye(4),
    )


class TestTRK(unittest.TestCase):
    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['empty_trk_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['empty_tractogram'])

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['simple_trk_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['simple_tractogram'])

    def test_load_complex_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['complex_trk_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['complex_tractogram'])

    def trk_with_bytes(self, trk_key='simple_trk_fname', endian='<'):
        """Return example trk file bytes and struct view onto bytes"""
        with open(DATA[trk_key], 'rb') as fobj:
            trk_bytes = bytearray(fobj.read())
        dt = trk_module.header_2_dtype.newbyteorder(endian)
        trk_struct = np.ndarray((1,), dt, buffer=trk_bytes)
        trk_struct.flags.writeable = True
        return trk_struct, trk_bytes

    def test_load_file_with_wrong_information(self):
        # Simulate a TRK file where `voxel_order` is lowercase.
        trk_struct1, trk_bytes1 = self.trk_with_bytes()
        trk_struct1[Field.VOXEL_ORDER] = b'LAS'
        trk1 = TrkFile.load(BytesIO(trk_bytes1))
        trk_struct2, trk_bytes2 = self.trk_with_bytes()
        trk_struct2[Field.VOXEL_ORDER] = b'las'
        trk2 = TrkFile.load(BytesIO(trk_bytes2))
        trk1_aff2rasmm = get_affine_trackvis_to_rasmm(trk1.header)
        trk2_aff2rasmm = get_affine_trackvis_to_rasmm(trk2.header)
        assert_array_equal(trk1_aff2rasmm, trk2_aff2rasmm)

        # Simulate a TRK file where `count` was not provided.
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.NB_STREAMLINES] = 0
        trk = TrkFile.load(BytesIO(trk_bytes), lazy_load=False)
        assert_tractogram_equal(trk.tractogram, DATA['simple_tractogram'])

        # Simulate a TRK where `vox_to_ras` is not recorded (i.e. all zeros).
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_TO_RASMM] = np.zeros((4, 4))
        with pytest.warns(HeaderWarning, match='identity'):
            trk = TrkFile.load(BytesIO(trk_bytes))
        assert_array_equal(trk.affine, np.eye(4))

        # Simulate a TRK where `vox_to_ras` is invalid.
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_TO_RASMM] = np.diag([0, 0, 0, 1])
        with clear_and_catch_warnings(modules=[trk_module]):
            with pytest.raises(HeaderError):
                TrkFile.load(BytesIO(trk_bytes))

        # Simulate a TRK file where `voxel_order` was not provided.
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_ORDER] = b''
        with pytest.warns(HeaderWarning, match='LPS'):
            TrkFile.load(BytesIO(trk_bytes))

        # Simulate a TRK file with an unsupported version.
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct['version'] = 123
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))

        # Simulate a TRK file with a wrong hdr_size.
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct['hdr_size'] = 1234
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))

        # Simulate a TRK file with a wrong scalar_name.
        trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_fname')
        trk_struct['scalar_name'][0, 0] = b'colors\x003\x004'
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))

        # Simulate a TRK file with a wrong property_name.
        trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_fname')
        trk_struct['property_name'][0, 0] = b'colors\x003\x004'
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))

    def test_load_trk_version_1(self):
        # Simulate and test a TRK (version 1).
        # First check that setting the RAS affine works in version 2.
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_TO_RASMM] = np.diag([2, 3, 4, 1])
        trk = TrkFile.load(BytesIO(trk_bytes))
        assert_array_equal(trk.affine, np.diag([2, 3, 4, 1]))
        # Next check that affine assumed identity if version 1.
        trk_struct['version'] = 1
        with pytest.warns(HeaderWarning, match='identity'):
            trk = TrkFile.load(BytesIO(trk_bytes))
        assert_array_equal(trk.affine, np.eye(4))
        assert_array_equal(trk.header['version'], 1)

    def test_load_complex_file_in_big_endian(self):
        trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_big_endian_fname', endian='>')
        # We use hdr_size as an indicator of little vs big endian.
        good_orders = '>' if sys.byteorder == 'little' else '>='
        hdr_size = trk_struct['hdr_size']
        assert hdr_size.dtype.byteorder in good_orders
        assert hdr_size == 1000

        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['complex_trk_big_endian_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['complex_tractogram'])

    def test_tractogram_file_properties(self):
        trk = TrkFile.load(DATA['simple_trk_fname'])
        assert trk.streamlines == trk.tractogram.streamlines
        assert_array_equal(trk.affine, trk.header[Field.VOXEL_TO_RASMM])

    def test_write_empty_file(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        new_trk_orig = TrkFile.load(DATA['empty_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['empty_trk_fname'], 'rb').read()

    def test_write_simple_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        new_trk_orig = TrkFile.load(DATA['simple_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['simple_trk_fname'], 'rb').read()

    def test_write_complex_file(self):
        # With scalars
        tractogram = Tractogram(
            DATA['streamlines'], data_per_point=DATA['data_per_point'], affine_to_rasmm=np.eye(4)
        )

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        # With properties
        data_per_streamline = DATA['data_per_streamline']
        tractogram = Tractogram(
            DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4)
        )

        trk = TrkFile(tractogram)
        trk_file = BytesIO()
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        # With scalars and properties
        data_per_streamline = DATA['data_per_streamline']
        tractogram = Tractogram(
            DATA['streamlines'],
            data_per_point=DATA['data_per_point'],
            data_per_streamline=data_per_streamline,
            affine_to_rasmm=np.eye(4),
        )

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        new_trk_orig = TrkFile.load(DATA['complex_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['complex_trk_fname'], 'rb').read()

    def test_load_write_file(self):
        for fname in [
            DATA['empty_trk_fname'],
            DATA['simple_trk_fname'],
            DATA['complex_trk_fname'],
        ]:
            for lazy_load in [False, True]:
                trk = TrkFile.load(fname, lazy_load=lazy_load)
                trk_file = BytesIO()
                trk.save(trk_file)

                new_trk = TrkFile.load(fname, lazy_load=False)
                assert_tractogram_equal(new_trk.tractogram, trk.tractogram)

    def test_load_write_LPS_file(self):
        # Load the RAS and LPS version of the standard.
        trk_RAS = TrkFile.load(DATA['standard_trk_fname'], lazy_load=False)
        trk_LPS = TrkFile.load(DATA['standard_LPS_trk_fname'], lazy_load=False)
        assert_tractogram_equal(trk_LPS.tractogram, trk_RAS.tractogram)

        # Write back the standard.
        trk_file = BytesIO()
        trk = TrkFile(trk_LPS.tractogram, trk_LPS.header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)

        assert_arr_dict_equal(new_trk.header, trk.header)
        assert_tractogram_equal(new_trk.tractogram, trk.tractogram)

        new_trk_orig = TrkFile.load(DATA['standard_LPS_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['standard_LPS_trk_fname'], 'rb').read()

        # Test writing a file where the header is missing the
        # Field.VOXEL_ORDER.
        trk_file = BytesIO()

        # For TRK file format, the default voxel order is LPS.
        header = copy.deepcopy(trk_LPS.header)
        header[Field.VOXEL_ORDER] = b''

        trk = TrkFile(trk_LPS.tractogram, header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)

        assert_arr_dict_equal(new_trk.header, trk_LPS.header)
        assert_tractogram_equal(new_trk.tractogram, trk.tractogram)

        new_trk_orig = TrkFile.load(DATA['standard_LPS_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['standard_LPS_trk_fname'], 'rb').read()

    def test_write_optional_header_fields(self):
        # The TRK file format doesn't support additional header fields.
        # If provided, they will be ignored.
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))

        trk_file = BytesIO()
        header = {'extra': 1234}
        trk = TrkFile(tractogram, header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)
        assert 'extra' not in new_trk.header

    def test_write_too_many_scalars_and_properties(self):
        # TRK supports up to 10 data_per_point.
        data_per_point = {}
        for i in range(10):
            data_per_point[f'#{i}'] = DATA['fa']

            tractogram = Tractogram(
                DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4)
            )

            trk_file = BytesIO()
            trk = TrkFile(tractogram)
            trk.save(trk_file)
            trk_file.seek(0, os.SEEK_SET)

            new_trk = TrkFile.load(trk_file, lazy_load=False)
            assert_tractogram_equal(new_trk.tractogram, tractogram)

        # More than 10 data_per_point should raise an error.
        data_per_point[f'#{i + 1}'] = DATA['fa']

        tractogram = Tractogram(
            DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4)
        )

        trk = TrkFile(tractogram)
        with pytest.raises(ValueError):
            trk.save(BytesIO())

        # TRK supports up to 10 data_per_streamline.
        data_per_streamline = {}
        for i in range(10):
            data_per_streamline[f'#{i}'] = DATA['mean_torsion']

            tractogram = Tractogram(
                DATA['streamlines'],
                data_per_streamline=data_per_streamline,
                affine_to_rasmm=np.eye(4),
            )

            trk_file = BytesIO()
            trk = TrkFile(tractogram)
            trk.save(trk_file)
            trk_file.seek(0, os.SEEK_SET)

            new_trk = TrkFile.load(trk_file, lazy_load=False)
            assert_tractogram_equal(new_trk.tractogram, tractogram)

        # More than 10 data_per_streamline should raise an error.
        data_per_streamline[f'#{i + 1}'] = DATA['mean_torsion']

        tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline)

        trk = TrkFile(tractogram)
        with pytest.raises(ValueError):
            trk.save(BytesIO())

    def test_write_scalars_and_properties_name_too_long(self):
        # TRK supports data_per_point name up to 20 characters.
        # However, we reserve the last two characters to store
        # the number of values associated to each data_per_point.
        # So in reality we allow name of 18 characters, otherwise
        # the name is truncated and warning is issue.
        for nb_chars in range(22):
            data_per_point = {'A' * nb_chars: DATA['colors']}
            tractogram = Tractogram(
                DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4)
            )

            trk = TrkFile(tractogram)
            if nb_chars > 18:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())

            data_per_point = {'A' * nb_chars: DATA['fa']}
            tractogram = Tractogram(
                DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4)
            )

            trk = TrkFile(tractogram)
            if nb_chars > 20:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())

        # TRK supports data_per_streamline name up to 20 characters.
        # However, we reserve the last two characters to store
        # the number of values associated to each data_per_streamline.
        # So in reality we allow name of 18 characters, otherwise
        # the name is truncated and warning is issue.
        for nb_chars in range(22):
            data_per_streamline = {'A' * nb_chars: DATA['mean_colors']}
            tractogram = Tractogram(
                DATA['streamlines'],
                data_per_streamline=data_per_streamline,
                affine_to_rasmm=np.eye(4),
            )

            trk = TrkFile(tractogram)
            if nb_chars > 18:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())

            data_per_streamline = {'A' * nb_chars: DATA['mean_torsion']}
            tractogram = Tractogram(
                DATA['streamlines'],
                data_per_streamline=data_per_streamline,
                affine_to_rasmm=np.eye(4),
            )

            trk = TrkFile(tractogram)
            if nb_chars > 20:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())

    def test_str(self):
        trk = TrkFile.load(DATA['complex_trk_fname'])
        str(trk)  # Simply test it's not failing when called.

    def test_header_read_restore(self):
        # Test that reading a header restores the file position
        trk_fname = DATA['simple_trk_fname']
        bio = BytesIO()
        bio.write(b'Along my very merry way')
        hdr_pos = bio.tell()
        hdr_from_fname = TrkFile._read_header(trk_fname)
        with open(trk_fname, 'rb') as fobj:
            bio.write(fobj.read())
        bio.seek(hdr_pos)
        # Check header is as expected
        hdr_from_fname['_offset_data'] += hdr_pos  # Correct for start position
        assert_arr_dict_equal(TrkFile._read_header(bio), hdr_from_fname)
        # Check fileobject file position has not changed
        assert bio.tell() == hdr_pos


def test_encode_names():
    # Test function for encoding numbers into property names
    b0 = b'\x00'
    assert encode_value_in_name(0, 'foo', 10) == b'foo' + b0 * 7
    assert encode_value_in_name(1, 'foo', 10) == b'foo' + b0 * 7
    assert encode_value_in_name(8, 'foo', 10) == b'foo' + b0 + b'8' + b0 * 5
    assert encode_value_in_name(40, 'foobar', 10) == b'foobar' + b0 + b'40' + b0
    assert encode_value_in_name(1, 'foobarbazz', 10) == b'foobarbazz'

    with pytest.raises(ValueError):
        encode_value_in_name(1, 'foobarbazzz', 10)

    with pytest.raises(ValueError):
        encode_value_in_name(2, 'foobarbazzz', 10)

    assert encode_value_in_name(2, 'foobarba', 10) == b'foobarba\x002'


def test_decode_names():
    # Test function for decoding name string into name, number
    b0 = b'\x00'
    assert decode_value_from_name(b'') == ('', 0)
    assert decode_value_from_name(b'foo' + b0 * 7) == ('foo', 1)
    assert decode_value_from_name(b'foo\x008' + b0 * 5) == ('foo', 8)
    assert decode_value_from_name(b'foobar\x0010\x00') == ('foobar', 10)

    with pytest.raises(ValueError):
        decode_value_from_name(b'foobar\x0010\x01')

    with pytest.raises(HeaderError):
        decode_value_from_name(b'foo\x0010\x00111')
