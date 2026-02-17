"""Testing Siemens CSA header reader"""

import struct
import sys

from ..structreader import Unpacker


def test_unpacker():
    s = b'1234\x00\x01'
    (le_int,) = struct.unpack('<h', b'\x00\x01')
    (be_int,) = struct.unpack('>h', b'\x00\x01')
    if sys.byteorder == 'little':
        native_int = le_int
        swapped_int = be_int
        swapped_code = '>'
    else:
        native_int = be_int
        swapped_int = le_int
        swapped_code = '<'
    up_str = Unpacker(s, endian='<')
    assert up_str.read(4) == b'1234'
    up_str.ptr = 0
    assert up_str.unpack('4s') == (b'1234',)
    assert up_str.unpack('h') == (le_int,)
    up_str = Unpacker(s, endian='>')
    assert up_str.unpack('4s') == (b'1234',)
    assert up_str.unpack('h') == (be_int,)
    # now test conflict of endian
    up_str = Unpacker(s, ptr=4, endian='>')
    assert up_str.unpack('<h') == (le_int,)
    up_str = Unpacker(s, ptr=4, endian=swapped_code)
    assert up_str.unpack('h') == (swapped_int,)
    up_str.ptr = 4
    assert up_str.unpack('<h') == (le_int,)
    up_str.ptr = 4
    assert up_str.unpack('>h') == (be_int,)
    up_str.ptr = 4
    assert up_str.unpack('@h') == (native_int,)
    # test -1 for read
    up_str.ptr = 2
    assert up_str.read() == b'34\x00\x01'
    # past end
    assert up_str.read() == b''
    # with n_bytes
    up_str.ptr = 2
    assert up_str.read(2) == b'34'
    assert up_str.read(2) == b'\x00\x01'
