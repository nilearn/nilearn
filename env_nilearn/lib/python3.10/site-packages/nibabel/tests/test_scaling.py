# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test for scaling / rounding in volumeutils module"""

import warnings
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..casting import sctypes, type_info
from ..testing import suppress_warnings
from ..volumeutils import apply_read_scaling, array_from_file, array_to_file, finite_range
from .test_volumeutils import _calculate_scale

# Debug print statements
DEBUG = True


@pytest.mark.parametrize(
    ('in_arr', 'res'),
    [
        ([[-1, 0, 1], [np.inf, np.nan, -np.inf]], (-1, 1)),
        (np.array([[-1, 0, 1], [np.inf, np.nan, -np.inf]]), (-1, 1)),
        ([[np.nan], [np.nan]], (np.inf, -np.inf)),  # all nans slices
        (np.zeros((3, 4, 5)) + np.nan, (np.inf, -np.inf)),
        ([[-np.inf], [np.inf]], (np.inf, -np.inf)),  # all infs slices
        (np.zeros((3, 4, 5)) + np.inf, (np.inf, -np.inf)),
        ([[np.nan, -1, 2], [-2, np.nan, 1]], (-2, 2)),
        ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)),
        ([[-np.inf, 2], [np.nan, 1]], (1, 2)),  # good max case
        ([np.nan], (np.inf, -np.inf)),
        ([np.inf], (np.inf, -np.inf)),
        ([-np.inf], (np.inf, -np.inf)),
        ([np.inf, 1], (1, 1)),  # only look at finite values
        ([-np.inf, 1], (1, 1)),
        ([[], []], (np.inf, -np.inf)),  # empty array
        (np.array([[-3, 0, 1], [2, -1, 4]], dtype=int), (-3, 4)),
        (np.array([[1, 0, 1], [2, 3, 4]], dtype=np.uint), (0, 4)),
        ([0.0, 1, 2, 3], (0, 3)),
        # Complex comparison works as if they are floats
        ([[np.nan, -1 - 100j, 2], [-2, np.nan, 1 + 100j]], (-2, 2)),
        ([[np.nan, -1, 2 - 100j], [-2 + 100j, np.nan, 1]], (-2 + 100j, 2 - 100j)),
    ],
)
def test_finite_range(in_arr, res):
    # Finite range utility function
    assert finite_range(in_arr) == res
    assert finite_range(in_arr, False) == res
    assert finite_range(in_arr, check_nan=False) == res
    has_nan = np.any(np.isnan(in_arr))
    assert finite_range(in_arr, True) == res + (has_nan,)
    assert finite_range(in_arr, check_nan=True) == res + (has_nan,)
    in_arr = np.array(in_arr)
    flat_arr = in_arr.ravel()
    assert finite_range(flat_arr) == res
    assert finite_range(flat_arr, True) == res + (has_nan,)
    # Check float types work as complex
    if in_arr.dtype.kind == 'f':
        c_arr = in_arr.astype(np.complex128)
        assert finite_range(c_arr) == res
        assert finite_range(c_arr, True) == res + (has_nan,)


def test_finite_range_err():
    # Test error cases
    a = np.array([[1.0, 0, 1], [2, 3, 4]]).view([('f1', 'f')])
    with pytest.raises(TypeError):
        finite_range(a)


@pytest.mark.parametrize('out_type', [np.int16, np.float32])
def test_a2f_mn_mx(out_type):
    # Test array to file mn, mx handling
    str_io = BytesIO()
    arr = np.arange(6, dtype=out_type)
    arr_orig = arr.copy()  # safe backup for testing against
    # Basic round trip to warm up
    array_to_file(arr, str_io)
    data_back = array_from_file(arr.shape, out_type, str_io)
    assert_array_equal(arr, data_back)
    # Clip low
    array_to_file(arr, str_io, mn=2)
    data_back = array_from_file(arr.shape, out_type, str_io)
    # arr unchanged
    assert_array_equal(arr, arr_orig)
    # returned value clipped low
    assert_array_equal(data_back, [2, 2, 2, 3, 4, 5])
    # Clip high
    array_to_file(arr, str_io, mx=4)
    data_back = array_from_file(arr.shape, out_type, str_io)
    # arr unchanged
    assert_array_equal(arr, arr_orig)
    # returned value clipped high
    assert_array_equal(data_back, [0, 1, 2, 3, 4, 4])
    # Clip both
    array_to_file(arr, str_io, mn=2, mx=4)
    data_back = array_from_file(arr.shape, out_type, str_io)
    # arr unchanged
    assert_array_equal(arr, arr_orig)
    # returned value clipped high
    assert_array_equal(data_back, [2, 2, 2, 3, 4, 4])


def test_a2f_nan2zero():
    # Test conditions under which nans written to zero
    arr = np.array([np.nan, 99.0], dtype=np.float32)
    str_io = BytesIO()
    array_to_file(arr, str_io)
    data_back = array_from_file(arr.shape, np.float32, str_io)
    assert_array_equal(np.isnan(data_back), [True, False])
    # nan2zero ignored for floats
    array_to_file(arr, str_io, nan2zero=True)
    data_back = array_from_file(arr.shape, np.float32, str_io)
    assert_array_equal(np.isnan(data_back), [True, False])
    # Integer output with nan2zero gives zero
    with np.errstate(invalid='ignore'):
        array_to_file(arr, str_io, np.int32, nan2zero=True)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, [0, 99])
    # Integer output with nan2zero=False gives whatever astype gives
    with np.errstate(invalid='ignore'):
        array_to_file(arr, str_io, np.int32, nan2zero=False)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, [np.array(np.nan).astype(np.int32), 99])


@pytest.mark.parametrize(
    ('in_type', 'out_type'),
    [
        (np.int16, np.int16),
        (np.int16, np.int8),
        (np.uint16, np.uint8),
        (np.int32, np.int8),
        (np.float32, np.uint8),
        (np.float32, np.int16),
    ],
)
def test_array_file_scales(in_type, out_type):
    # Test scaling works for max, min when going from larger to smaller type,
    # and from float to integer.
    bio = BytesIO()
    out_dtype = np.dtype(out_type)
    arr = np.zeros((3,), dtype=in_type)
    info = type_info(in_type)
    arr[0], arr[1] = info['min'], info['max']
    slope, inter, mn, mx = _calculate_scale(arr, out_dtype, True)
    array_to_file(arr, bio, out_type, 0, inter, slope, mn, mx)
    bio.seek(0)
    arr2 = array_from_file(arr.shape, out_dtype, bio)
    arr3 = apply_read_scaling(arr2, slope, inter)
    # Max rounding error for integer type
    max_miss = slope / 2.0
    assert np.all(np.abs(arr - arr3) <= max_miss)


@pytest.mark.parametrize(
    ('category0', 'category1', 'overflow'),
    [
        # Confirm that, for all ints and uints as input, and all possible outputs,
        # for any simple way of doing the calculation, the result is near enough
        ('int', 'int', False),
        ('uint', 'int', False),
        # Converting floats to integer
        ('float', 'int', True),
        ('float', 'uint', True),
        ('complex', 'int', True),
        ('complex', 'uint', True),
    ],
)
def test_scaling_in_abstract(category0, category1, overflow):
    for in_type in sctypes[category0]:
        for out_type in sctypes[category1]:
            if overflow:
                with suppress_warnings():
                    check_int_a2f(in_type, out_type)
            else:
                check_int_a2f(in_type, out_type)


def check_int_a2f(in_type, out_type):
    # Check that array to / from file returns roughly the same as input
    big_floater = sctypes['float'][-1]
    info = type_info(in_type)
    this_min, this_max = info['min'], info['max']
    if not in_type in sctypes['complex']:
        data = np.array([this_min, this_max], in_type)
        # Bug in numpy 1.6.2 on PPC leading to infs - abort
        if not np.all(np.isfinite(data)):
            if DEBUG:
                print(f'Hit PPC max -> inf bug; skip in_type {in_type}')
            return
    else:  # Funny behavior with complex256
        data = np.zeros((2,), in_type)
        data[0] = this_min + 0j
        data[1] = this_max + 0j
    str_io = BytesIO()
    try:
        scale, inter, mn, mx = _calculate_scale(data, out_type, True)
    except ValueError as e:
        if DEBUG:
            warnings.warn(str((in_type, out_type, e)))
        return
    array_to_file(data, str_io, out_type, 0, inter, scale, mn, mx)
    data_back = array_from_file(data.shape, out_type, str_io)
    data_back = apply_read_scaling(data_back, scale, inter)
    assert np.allclose(big_floater(data), big_floater(data_back))
    # Try with analyze-size scale and inter
    scale32 = np.float32(scale)
    inter32 = np.float32(inter)
    if scale32 == np.inf or inter32 == np.inf:
        return
    data_back = array_from_file(data.shape, out_type, str_io)
    data_back = apply_read_scaling(data_back, scale32, inter32)
    # Clip at extremes to remove inf
    info = type_info(in_type)
    out_min, out_max = info['min'], info['max']
    assert np.allclose(big_floater(data), big_floater(np.clip(data_back, out_min, out_max)))
