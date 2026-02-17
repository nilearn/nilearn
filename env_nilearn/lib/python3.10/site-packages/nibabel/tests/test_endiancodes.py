# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for endiancodes module"""

import sys

from ..volumeutils import endian_codes, native_code, swapped_code


def test_native_swapped():
    native_is_le = sys.byteorder == 'little'
    if native_is_le:
        assert (native_code, swapped_code) == ('<', '>')
    else:
        assert (native_code, swapped_code) == ('>', '<')


def test_to_numpy():
    if sys.byteorder == 'little':
        assert endian_codes['native'] == '<'
        assert endian_codes['swapped'] == '>'
    else:
        assert endian_codes['native'] == '>'
        assert endian_codes['swapped'] == '<'
    assert endian_codes['native'] == endian_codes['=']
    assert endian_codes['big'] == '>'
    for code in ('little', '<', 'l', 'L', 'le'):
        assert endian_codes[code] == '<'
    for code in ('big', '>', 'b', 'B', 'be'):
        assert endian_codes[code] == '>'
