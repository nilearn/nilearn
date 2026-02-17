# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Test diff"""

from os.path import abspath, dirname
from os.path import join as pjoin

import numpy as np

DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

from nibabel.cmdline.diff import are_values_different


def test_diff_values_int():
    large = 10**30
    assert not are_values_different(0, 0)
    assert not are_values_different(1, 1)
    assert not are_values_different(large, large)
    assert are_values_different(0, 1)
    assert are_values_different(1, 2)
    assert are_values_different(1, large)


def test_diff_values_float():
    assert not are_values_different(0.0, 0.0)
    assert not are_values_different(0.0, 0.0, 0.0)  # can take more
    assert not are_values_different(1.1, 1.1)
    assert are_values_different(0.0, 1.1)
    assert are_values_different(0.0, 0, 1.1)
    assert are_values_different(1.0, 2.0)


def test_diff_values_mixed():
    assert are_values_different(1.0, 1)
    assert are_values_different(1.0, '1')
    assert are_values_different(1, '1')
    assert are_values_different(1, None)
    assert are_values_different(np.ndarray([0]), 'hey')
    assert not are_values_different(None, None)


def test_diff_values_array():
    from numpy import array, inf, nan

    a_int = array([1, 2])
    a_float = a_int.astype(float)

    assert are_values_different(a_int, a_float)
    assert are_values_different(a_int, a_int, a_float)
    assert are_values_different(np.arange(3), np.arange(1, 4))
    assert are_values_different(np.arange(3), np.arange(4))
    assert are_values_different(np.arange(4), np.arange(4).reshape((2, 2)))
    # no broadcasting should kick in - shape difference
    assert are_values_different(array([1]), array([1, 1]))
    assert not are_values_different(a_int, a_int)
    assert not are_values_different(a_float, a_float)

    # nans - we consider them "the same" for the purpose of these comparisons
    assert not are_values_different(nan, nan)
    assert not are_values_different(nan, nan, nan)
    assert are_values_different(nan, nan, 1)
    assert are_values_different(1, nan, nan)
    assert not are_values_different(array([nan, nan]), array([nan, nan]))
    assert not are_values_different(array([nan, nan]), array([nan, nan]), array([nan, nan]))
    assert not are_values_different(array([nan, 1]), array([nan, 1]))
    assert are_values_different(array([nan, nan]), array([nan, 1]))
    assert are_values_different(array([0, nan]), array([nan, 0]))
    assert are_values_different(array([1, 2, 3, nan]), array([nan, 3, 5, 4]))
    assert are_values_different(nan, 1.0)
    assert are_values_different(array([1, 2, 3, nan]), array([3, 4, 5, nan]))
    # and some inf should not be a problem
    assert not are_values_different(array([0, inf]), array([0, inf]))
    assert are_values_different(array([0, inf]), array([inf, 0]))

    # we will allow for types to be of different endianness but the
    # same in "instantiation" type and value
    assert not are_values_different(np.array(1, dtype='<i4'), np.array(1, dtype='>i4'))
    # but do report difference if instantiation type is different:
    assert are_values_different(np.array(1, dtype='<i4'), np.array(1, dtype='<i2'))
