# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests recoder class"""

import numpy as np
import pytest

from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code


def test_recoder_1():
    # simplest case, no aliases
    codes = ((1,), (2,))
    rc = Recoder(codes)
    assert rc.code[1] == 1
    assert rc.code[2] == 2
    with pytest.raises(KeyError):
        rc.code[3]


def test_recoder_2():
    # with explicit name for code
    codes = ((1,), (2,))
    rc = Recoder(codes, ['code1'])
    with pytest.raises(AttributeError):
        rc.code
    assert rc.code1[1] == 1
    assert rc.code1[2] == 2


def test_recoder_3():
    # code and label
    codes = ((1, 'one'), (2, 'two'))
    rc = Recoder(codes)  # just with implicit alias
    assert rc.code[1] == 1
    assert rc.code[2] == 2
    with pytest.raises(KeyError):
        rc.code[3]
    assert rc.code['one'] == 1
    assert rc.code['two'] == 2
    with pytest.raises(KeyError):
        rc.code['three']
    with pytest.raises(AttributeError):
        rc.label


def test_recoder_4():
    # with explicit column names
    codes = ((1, 'one'), (2, 'two'))
    rc = Recoder(codes, ['code1', 'label'])
    with pytest.raises(AttributeError):
        rc.code
    assert rc.code1[1] == 1
    assert rc.code1['one'] == 1
    assert rc.label[1] == 'one'
    assert rc.label['one'] == 'one'


def test_recoder_5():
    # code, label, aliases
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes)  # just with implicit alias
    assert rc.code[1] == 1
    assert rc.code['one'] == 1
    assert rc.code['first'] == 1


def test_recoder_6():
    # with explicit column names
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes, ['code1', 'label'])
    assert rc.code1[1] == 1
    assert rc.code1['first'] == 1
    assert rc.label[1] == 'one'
    assert rc.label['first'] == 'one'
    # Don't allow funny names
    with pytest.raises(KeyError):
        Recoder(codes, ['field1'])


def test_custom_dicter():
    # Allow custom dict-like object in constructor
    class MyDict:
        def __init__(self):
            self._keys = []

        def __setitem__(self, key, value):
            self._keys.append(key)

        def __getitem__(self, key):
            if key in self._keys:
                return 'spam'
            return 'eggs'

        def keys(self):
            return ['some', 'keys']

        def values(self):
            return ['funny', 'list']

    # code, label, aliases
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes, map_maker=MyDict)
    assert rc.code[1] == 'spam'
    assert rc.code['one'] == 'spam'
    assert rc.code['first'] == 'spam'
    assert rc.code['bizarre'] == 'eggs'
    assert rc.value_set() == {'funny', 'list'}
    assert list(rc.keys()) == ['some', 'keys']


def test_add_codes():
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes)
    assert rc.code['two'] == 2
    with pytest.raises(KeyError):
        rc.code['three']
    rc.add_codes(((3, 'three'), (1, 'number 1')))
    assert rc.code['three'] == 3
    assert rc.code['number 1'] == 1


def test_sugar():
    # Syntactic sugar for recoder class
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes)
    # Field1 is synonym for first named dict
    assert rc.code == rc.field1
    rc = Recoder(codes, fields=('code1', 'label'))
    assert rc.code1 == rc.field1
    # Direct key access identical to key access for first named
    assert rc[1] == rc.field1[1]
    assert rc['two'] == rc.field1['two']
    # keys gets all keys
    assert set(rc.keys()) == {1, 'one', '1', 'first', 2, 'two'}
    # value_set gets set of values from first column
    assert rc.value_set() == {1, 2}
    # or named column if given
    assert rc.value_set('label') == {'one', 'two'}
    # "in" works for values in and outside the set
    assert 'one' in rc
    assert 'three' not in rc


def test_dtmapper():
    # dict-like that will lookup on dtypes, even if they don't hash properly
    d = DtypeMapper()
    with pytest.raises(KeyError):
        d[1]
    d[1] = 'something'
    assert d[1] == 'something'
    assert list(d.keys()) == [1]
    assert list(d.values()) == ['something']
    intp_dt = np.dtype('intp')
    if intp_dt == np.dtype('int32'):
        canonical_dt = np.dtype('int32')
    elif intp_dt == np.dtype('int64'):
        canonical_dt = np.dtype('int64')
    else:
        raise RuntimeError('Can I borrow your computer?')
    native_dt = canonical_dt.newbyteorder('=')
    explicit_dt = canonical_dt.newbyteorder(native_code)
    d[canonical_dt] = 'spam'
    assert d[canonical_dt] == 'spam'
    assert d[native_dt] == 'spam'
    assert d[explicit_dt] == 'spam'

    # Test keys, values
    d = DtypeMapper()
    assert list(d.keys()) == []
    assert list(d.keys()) == []
    d[canonical_dt] = 'spam'
    assert list(d.keys()) == [canonical_dt]
    assert list(d.values()) == ['spam']
    # With other byte order
    d = DtypeMapper()
    sw_dt = canonical_dt.newbyteorder(swapped_code)
    d[sw_dt] = 'spam'
    with pytest.raises(KeyError):
        d[canonical_dt]
    assert d[sw_dt] == 'spam'
    sw_intp_dt = intp_dt.newbyteorder(swapped_code)
    assert d[sw_intp_dt] == 'spam'
