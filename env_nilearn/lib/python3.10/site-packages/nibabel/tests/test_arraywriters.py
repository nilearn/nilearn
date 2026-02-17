"""Testing array writer objects

See docstring of :mod:`nibabel.arraywriters` for API.
"""

import itertools
from io import BytesIO
from platform import machine, python_compiler

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..arraywriters import (
    ArrayWriter,
    ScalingError,
    SlopeArrayWriter,
    SlopeInterArrayWriter,
    WriterError,
    get_slope_inter,
    make_array_writer,
)
from ..casting import int_abs, sctypes, shared_range, type_info
from ..testing import assert_allclose_safely, suppress_warnings
from ..volumeutils import _dt_min_max, apply_read_scaling, array_from_file

FLOAT_TYPES = sctypes['float']
COMPLEX_TYPES = sctypes['complex']
INT_TYPES = sctypes['int']
UINT_TYPES = sctypes['uint']
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
IUINT_TYPES = INT_TYPES + UINT_TYPES
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES


def round_trip(writer, order='F', apply_scale=True):
    sio = BytesIO()
    arr = writer.array
    with np.errstate(invalid='ignore'):
        writer.to_fileobj(sio, order)
    data_back = array_from_file(arr.shape, writer.out_dtype, sio, order=order)
    slope, inter = get_slope_inter(writer)
    if apply_scale:
        data_back = apply_read_scaling(data_back, slope, inter)
    return data_back


def test_arraywriters():
    # Test initialize
    # Simple cases
    if machine() == 'sparc64' and python_compiler().startswith('GCC'):
        # bus errors on at least np 1.4.1 through 1.6.1 for complex
        test_types = FLOAT_TYPES + IUINT_TYPES
    else:
        test_types = NUMERIC_TYPES
    for klass in (SlopeInterArrayWriter, SlopeArrayWriter, ArrayWriter):
        for type in test_types:
            arr = np.arange(10, dtype=type)
            aw = klass(arr)
            assert aw.array is arr
            assert aw.out_dtype == arr.dtype
            assert_array_equal(arr, round_trip(aw))
            # Byteswapped should be OK
            bs_arr = arr.byteswap()
            bs_arr = bs_arr.view(bs_arr.dtype.newbyteorder('S'))
            bs_aw = klass(bs_arr)
            bs_aw_rt = round_trip(bs_aw)
            # assert against original array because POWER7 was running into
            # trouble using the byteswapped array (bs_arr)
            assert_array_equal(arr, bs_aw_rt)
            bs_aw2 = klass(bs_arr, arr.dtype)
            bs_aw2_rt = round_trip(bs_aw2)
            assert_array_equal(arr, bs_aw2_rt)
            # 2D array
            arr2 = np.reshape(arr, (2, 5))
            a2w = klass(arr2)
            # Default out - in order is Fortran
            arr_back = round_trip(a2w)
            assert_array_equal(arr2, arr_back)
            arr_back = round_trip(a2w, 'F')
            assert_array_equal(arr2, arr_back)
            # C order works as well
            arr_back = round_trip(a2w, 'C')
            assert_array_equal(arr2, arr_back)
            assert arr_back.flags.c_contiguous


def test_arraywriter_check_scaling():
    # Check keyword-only argument to ArrayWriter
    # Within range - OK
    arr = np.array([0, 1, 128, 255], np.uint8)
    aw = ArrayWriter(arr)
    # Out of range, scaling needed, default is error
    with pytest.raises(WriterError):
        ArrayWriter(arr, np.int8)
    # Make default explicit
    with pytest.raises(WriterError):
        ArrayWriter(arr, np.int8, check_scaling=True)
    # Turn off scaling check
    aw = ArrayWriter(arr, np.int8, check_scaling=False)
    assert_array_equal(round_trip(aw), np.clip(arr, 0, 127))
    # Has to be keyword
    with pytest.raises(TypeError):
        ArrayWriter(arr, np.int8, False)


def test_no_scaling():
    # Test arraywriter when writing different types without scaling
    for in_dtype, out_dtype, awt in itertools.product(
        NUMERIC_TYPES, NUMERIC_TYPES, (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter)
    ):
        mn_in, mx_in = _dt_min_max(in_dtype)
        arr = np.array([mn_in, 0, 1, mx_in], dtype=in_dtype)
        kwargs = dict(check_scaling=False) if awt == ArrayWriter else dict(calc_scale=False)
        aw = awt(arr, out_dtype, **kwargs)
        with suppress_warnings():
            back_arr = round_trip(aw)
        exp_back = arr.copy()
        # If converting to floating point type, casting is direct.
        # Otherwise we will need to do float-(u)int casting at some point.
        if out_dtype in IUINT_TYPES:
            if in_dtype in CFLOAT_TYPES:
                # Working precision is (at least) float
                with suppress_warnings():
                    exp_back = exp_back.astype(float)
                # Float to iu conversion will always round, clip
                with np.errstate(invalid='ignore'):
                    exp_back = np.round(exp_back)
                if hasattr(aw, 'slope') and in_dtype in FLOAT_TYPES:
                    # Finite scaling sets infs to min / max
                    exp_back = np.clip(exp_back, 0, 1)
                else:
                    # Clip to shared range of working precision
                    exp_back = np.clip(exp_back, *shared_range(float, out_dtype))
            else:  # iu input and output type
                # No scaling, never gets converted to float.
                # Does get clipped to range of output type
                mn_out, mx_out = _dt_min_max(out_dtype)
                if (mn_in, mx_in) != (mn_out, mx_out):
                    # Use smaller of input, output range to avoid np.clip
                    # upcasting the array because of large clip limits.
                    exp_back = np.clip(exp_back, max(mn_in, mn_out), min(mx_in, mx_out))
        elif in_dtype in COMPLEX_TYPES:
            # always cast to real from complex
            with suppress_warnings():
                exp_back = exp_back.astype(float)
        exp_back = exp_back.astype(out_dtype)
        # Sometimes working precision is float32 - allow for small differences
        assert_allclose_safely(back_arr, exp_back)


def test_scaling_needed():
    # Structured types return True if dtypes same, raise error otherwise
    dt_def = [('f', 'i4')]
    arr = np.ones(10, dt_def)
    for t in NUMERIC_TYPES:
        with pytest.raises(WriterError):
            ArrayWriter(arr, t)
        narr = np.ones(10, t)
        with pytest.raises(WriterError):
            ArrayWriter(narr, dt_def)
    assert not ArrayWriter(arr).scaling_needed()
    assert not ArrayWriter(arr, dt_def).scaling_needed()
    # Any numeric type that can cast, needs no scaling
    for in_t in NUMERIC_TYPES:
        for out_t in NUMERIC_TYPES:
            if np.can_cast(in_t, out_t):
                aw = ArrayWriter(np.ones(10, in_t), out_t)
                assert not aw.scaling_needed()
    for in_t in NUMERIC_TYPES:
        # Numeric types to complex never need scaling
        arr = np.ones(10, in_t)
        for out_t in COMPLEX_TYPES:
            assert not ArrayWriter(arr, out_t).scaling_needed()
    # Attempts to scale from complex to anything else fails
    for in_t in COMPLEX_TYPES:
        for out_t in FLOAT_TYPES + IUINT_TYPES:
            arr = np.ones(10, in_t)
            with pytest.raises(WriterError):
                ArrayWriter(arr, out_t)
    # Scaling from anything but complex to floats is OK
    for in_t in FLOAT_TYPES + IUINT_TYPES:
        arr = np.ones(10, in_t)
        for out_t in FLOAT_TYPES:
            assert not ArrayWriter(arr, out_t).scaling_needed()
    # For any other output type, arrays with no data don't need scaling
    for in_t in FLOAT_TYPES + IUINT_TYPES:
        arr_0 = np.zeros(10, in_t)
        arr_e = []
        for out_t in IUINT_TYPES:
            assert not ArrayWriter(arr_0, out_t).scaling_needed()
            assert not ArrayWriter(arr_e, out_t).scaling_needed()
    # Going to (u)ints, non-finite arrays don't need scaling for writers that
    # can do scaling because these use finite_range to threshold the input data,
    # but ArrayWriter does not do this. so scaling_needed is True
    for in_t in FLOAT_TYPES:
        arr_nan = np.zeros(10, in_t) + np.nan
        arr_inf = np.zeros(10, in_t) + np.inf
        arr_minf = np.zeros(10, in_t) - np.inf
        arr_mix = np.array([np.nan, np.inf, -np.inf], dtype=in_t)
        for out_t in IUINT_TYPES:
            for arr in (arr_nan, arr_inf, arr_minf, arr_mix):
                assert ArrayWriter(arr, out_t, check_scaling=False).scaling_needed()
                assert not SlopeArrayWriter(arr, out_t).scaling_needed()
                assert not SlopeInterArrayWriter(arr, out_t).scaling_needed()
    # Floats as input always need scaling
    for in_t in FLOAT_TYPES:
        arr = np.ones(10, in_t)
        for out_t in IUINT_TYPES:
            # We need an arraywriter that will tolerate construction when
            # scaling is needed
            assert SlopeArrayWriter(arr, out_t).scaling_needed()
    # in-range (u)ints don't need scaling
    for in_t in IUINT_TYPES:
        in_info = np.iinfo(in_t)
        in_min, in_max = in_info.min, in_info.max
        for out_t in IUINT_TYPES:
            out_info = np.iinfo(out_t)
            out_min, out_max = out_info.min, out_info.max
            if in_min >= out_min and in_max <= out_max:
                arr = np.array([in_min, in_max], in_t)
                assert np.can_cast(arr.dtype, out_t)
                # We've already tested this with can_cast above, but...
                assert not ArrayWriter(arr, out_t).scaling_needed()
                continue
            # The output data type does not include the input data range
            max_min = max(in_min, out_min)  # 0 for input or output uint
            min_max = min(in_max, out_max)
            arr = np.array([max_min, min_max], in_t)
            assert not ArrayWriter(arr, out_t).scaling_needed()
            assert SlopeInterArrayWriter(arr + 1, out_t).scaling_needed()
            if in_t in INT_TYPES:
                assert SlopeInterArrayWriter(arr - 1, out_t).scaling_needed()


def test_special_rt():
    # Test that zeros; none finite - round trip to zeros for scalable types
    # For ArrayWriter, these error for default creation, when forced to create
    # the writer, they round trip to out_dtype max
    arr = np.array([np.inf, np.nan, -np.inf])
    for in_dtt in FLOAT_TYPES:
        for out_dtt in IUINT_TYPES:
            in_arr = arr.astype(in_dtt)
            with pytest.raises(WriterError):
                ArrayWriter(in_arr, out_dtt)
            aw = ArrayWriter(in_arr, out_dtt, check_scaling=False)
            mn, mx = shared_range(float, out_dtt)
            assert np.allclose(round_trip(aw).astype(float), [mx, 0, mn])
            for klass in (SlopeArrayWriter, SlopeInterArrayWriter):
                aw = klass(in_arr, out_dtt)
                assert get_slope_inter(aw) == (1, 0)
                assert_array_equal(round_trip(aw), 0)
    for in_dtt, out_dtt, awt in itertools.product(
        FLOAT_TYPES, IUINT_TYPES, (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter)
    ):
        arr = np.zeros((3,), dtype=in_dtt)
        aw = awt(arr, out_dtt)
        assert get_slope_inter(aw) == (1, 0)
        assert_array_equal(round_trip(aw), 0)


def test_high_int2uint():
    # Need to take care of high values when testing whether values are already
    # in range.  There was a bug here were the comparison was in floating point,
    # and therefore not exact, and 2**63 appeared to be in range for np.int64
    arr = np.array([2**63], dtype=np.uint64)
    out_type = np.int64
    aw = SlopeInterArrayWriter(arr, out_type)
    assert aw.inter == 2**63


def test_slope_inter_castable():
    # Test scaling for arraywriter instances
    # Test special case of all zeros
    for in_dtt in FLOAT_TYPES + IUINT_TYPES:
        for out_dtt in NUMERIC_TYPES:
            for klass in (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter):
                arr = np.zeros((5,), dtype=in_dtt)
                klass(arr, out_dtt)  # no error
    # Test special case of none finite
    # This raises error for ArrayWriter, but not for the others
    arr = np.array([np.inf, np.nan, -np.inf])
    for in_dtt in FLOAT_TYPES:
        for out_dtt in IUINT_TYPES:
            in_arr = arr.astype(in_dtt)
            with pytest.raises(WriterError):
                ArrayWriter(in_arr, out_dtt)
            SlopeArrayWriter(arr.astype(in_dtt), out_dtt)  # no error
            SlopeInterArrayWriter(arr.astype(in_dtt), out_dtt)  # no error
    for in_dtt, out_dtt, arr, slope_only, slope_inter, neither in (
        (np.float32, np.float32, 1, True, True, True),
        (np.float64, np.float32, 1, True, True, True),
        (np.float32, np.complex128, 1, True, True, True),
        (np.uint32, np.complex128, 1, True, True, True),
        (np.int64, np.float32, 1, True, True, True),
        (np.float32, np.int16, 1, True, True, False),
        (np.complex128, np.float32, 1, False, False, False),
        (np.complex128, np.int16, 1, False, False, False),
        (np.uint8, np.int16, 1, True, True, True),
        # The following tests depend on the input data
        (np.uint16, np.int16, 1, True, True, True),  # 1 is in range
        (np.uint16, np.int16, 2**16 - 1, True, True, False),  # This not in range
        (np.uint16, np.int16, (0, 2**16 - 1), True, True, False),
        (np.uint16, np.uint8, 1, True, True, True),
        (np.int16, np.uint16, 1, True, True, True),  # in range
        (np.int16, np.uint16, -1, True, True, False),  # flip works for scaling
        (np.int16, np.uint16, (-1, 1), False, True, False),  # not with +-
        (np.int8, np.uint16, 1, True, True, True),  # in range
        (np.int8, np.uint16, -1, True, True, False),  # flip works for scaling
        (np.int8, np.uint16, (-1, 1), False, True, False),  # not with +-
    ):
        # data for casting
        data = np.array(arr, dtype=in_dtt)
        # With scaling but no intercept
        if slope_only:
            SlopeArrayWriter(data, out_dtt)
        else:
            with pytest.raises(WriterError):
                SlopeArrayWriter(data, out_dtt)
        # With scaling and intercept
        if slope_inter:
            SlopeInterArrayWriter(data, out_dtt)
        else:
            with pytest.raises(WriterError):
                SlopeInterArrayWriter(data, out_dtt)
        # With neither
        if neither:
            ArrayWriter(data, out_dtt)
        else:
            with pytest.raises(WriterError):
                ArrayWriter(data, out_dtt)


def test_calculate_scale():
    # Test for special cases in scale calculation
    npa = np.array
    SIAW = SlopeInterArrayWriter
    SAW = SlopeArrayWriter
    # Offset handles scaling when it can
    aw = SIAW(npa([-2, -1], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (1.0, -2.0)
    # Sign flip handles these cases
    aw = SAW(npa([-2, -1], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (-1.0, 0.0)
    aw = SAW(npa([-2, 0], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (-1.0, 0.0)
    # But not when min magnitude is too large (scaling mechanism kicks in)
    aw = SAW(npa([-510, 0], dtype=np.int16), np.uint8)
    assert get_slope_inter(aw) == (-2.0, 0.0)
    # Or for floats (attempts to expand across range)
    aw = SAW(npa([-2, 0], dtype=np.float32), np.uint8)
    assert get_slope_inter(aw) != (-1.0, 0.0)
    # Case where offset handles scaling
    aw = SIAW(npa([-1, 1], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (1.0, -1.0)
    # Can't work for no offset case
    with pytest.raises(WriterError):
        SAW(npa([-1, 1], dtype=np.int8), np.uint8)
    # Offset trick can't work when max is out of range
    aw = SIAW(npa([-1, 255], dtype=np.int16), np.uint8)
    slope_inter = get_slope_inter(aw)
    assert slope_inter != (1.0, -1.0)


def test_resets():
    # Test reset of values, caching of scales
    for klass, inp, outp in (
        (SlopeInterArrayWriter, (1, 511), (2.0, 1.0)),
        (SlopeArrayWriter, (0, 510), (2.0, 0.0)),
    ):
        arr = np.array(inp)
        outp = np.array(outp)
        aw = klass(arr, np.uint8)
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale()  # cached no change
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale(force=True)  # same data, no change
        assert_array_equal(get_slope_inter(aw), outp)
        # Change underlying array
        aw.array[:] = aw.array * 2
        aw.calc_scale()  # cached still
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale(force=True)  # new data, change
        assert_array_equal(get_slope_inter(aw), outp * 2)
        # Test reset
        aw.reset()
        assert_array_equal(get_slope_inter(aw), (1.0, 0.0))


def test_no_offset_scale():
    # Specific tests of no-offset scaling
    SAW = SlopeArrayWriter
    # Floating point
    for data in (
        (-128, 127),
        (-128, 126),
        (-128, -127),
        (-128, 0),
        (-128, -1),
        (126, 127),
        (-127, 127),
    ):
        aw = SAW(np.array(data, dtype=np.float32), np.int8)
        assert aw.slope == 1.0
    aw = SAW(np.array([-126, 127 * 2.0], dtype=np.float32), np.int8)
    assert aw.slope == 2
    aw = SAW(np.array([-128 * 2.0, 127], dtype=np.float32), np.int8)
    assert aw.slope == 2
    # Test that nasty abs behavior does not upset us
    n = -(2**15)
    aw = SAW(np.array([n, n], dtype=np.int16), np.uint8)
    assert_array_almost_equal(aw.slope, n / 255.0, 5)


def test_with_offset_scale():
    # Tests of specific cases in slope, inter
    SIAW = SlopeInterArrayWriter
    aw = SIAW(np.array([0, 127], dtype=np.int8), np.uint8)
    assert (aw.slope, aw.inter) == (1, 0)  # in range
    aw = SIAW(np.array([-1, 126], dtype=np.int8), np.uint8)
    assert (aw.slope, aw.inter) == (1, -1)  # offset only
    aw = SIAW(np.array([-1, 254], dtype=np.int16), np.uint8)
    assert (aw.slope, aw.inter) == (1, -1)  # offset only
    aw = SIAW(np.array([-1, 255], dtype=np.int16), np.uint8)
    assert (aw.slope, aw.inter) != (1, -1)  # Too big for offset only
    aw = SIAW(np.array([-256, -2], dtype=np.int16), np.uint8)
    assert (aw.slope, aw.inter) == (1, -256)  # offset only
    aw = SIAW(np.array([-256, -2], dtype=np.int16), np.int8)
    assert (aw.slope, aw.inter) == (1, -129)  # offset only


def test_io_scaling():
    # Test scaling works for max, min when going from larger to smaller type,
    # and from float to integer.
    bio = BytesIO()
    for in_type, out_type in itertools.product(
        (np.int16, np.uint16, np.float32), (np.int8, np.uint8, np.int16, np.uint16)
    ):
        out_dtype = np.dtype(out_type)
        info = type_info(in_type)
        imin, imax = info['min'], info['max']
        if imin == 0:  # unsigned int
            val_tuples = ((0, imax), (100, imax))
        else:
            val_tuples = ((imin, 0, imax), (imin, 0), (0, imax), (imin, 100, imax))
        if imin != 0:
            val_tuples += ((imin, 0), (0, imax))
        for vals in val_tuples:
            arr = np.array(vals, dtype=in_type)
            aw = SlopeInterArrayWriter(arr, out_dtype)
            aw.to_fileobj(bio)
            arr2 = array_from_file(arr.shape, out_dtype, bio)
            arr3 = apply_read_scaling(arr2, aw.slope, aw.inter)
            # Max rounding error for integer type
            # Slope might be negative
            max_miss = np.abs(aw.slope) / 2.0
            abs_err = np.abs(arr - arr3)
            assert np.all(abs_err <= max_miss)
            if out_type in UINT_TYPES and 0 in (min(arr), max(arr)):
                # Check that error is minimized for 0 as min or max
                assert min(abs_err) == abs_err[arr == 0]
            bio.truncate(0)
            bio.seek(0)


def test_input_ranges():
    # Test we get good precision for a range of input data
    arr = np.arange(-500, 501, 10, dtype=np.float64)
    bio = BytesIO()
    working_type = np.float32
    work_eps = np.finfo(working_type).eps
    for out_type, offset in itertools.product(IUINT_TYPES, range(-1000, 1000, 100)):
        aw = SlopeInterArrayWriter(arr, out_type)
        aw.to_fileobj(bio)
        arr2 = array_from_file(arr.shape, out_type, bio)
        arr3 = apply_read_scaling(arr2, aw.slope, aw.inter)
        # Max rounding error for integer type
        # Slope might be negative
        max_miss = np.abs(aw.slope) / working_type(2.0) + work_eps * 10
        abs_err = np.abs(arr - arr3)
        max_err = np.abs(arr) * work_eps + max_miss
        assert np.all(abs_err <= max_err)
        if out_type in UINT_TYPES and 0 in (min(arr), max(arr)):
            # Check that error is minimized for 0 as min or max
            assert min(abs_err) == abs_err[arr == 0]
        bio.truncate(0)
        bio.seek(0)


def test_nan2zero():
    # Test conditions under which nans written to zero, and error conditions
    # nan2zero as argument to `to_fileobj` deprecated, raises error if not the
    # same as input nan2zero - meaning that by default, nan2zero of False will
    # raise an error.
    arr = np.array([np.nan, 99.0], dtype=np.float32)
    for awt, kwargs in (
        (ArrayWriter, dict(check_scaling=False)),
        (SlopeArrayWriter, dict(calc_scale=False)),
        (SlopeInterArrayWriter, dict(calc_scale=False)),
    ):
        # nan2zero default is True
        # nan2zero ignored for floats
        aw = awt(arr, np.float32, **kwargs)
        data_back = round_trip(aw)
        assert_array_equal(np.isnan(data_back), [True, False])
        # set explicitly
        aw = awt(arr, np.float32, nan2zero=True, **kwargs)
        data_back = round_trip(aw)
        assert_array_equal(np.isnan(data_back), [True, False])
        # Integer output with nan2zero gives zero
        aw = awt(arr, np.int32, **kwargs)
        data_back = round_trip(aw)
        assert_array_equal(data_back, [0, 99])
        # Integer output with nan2zero=False gives whatever astype gives
        aw = awt(arr, np.int32, nan2zero=False, **kwargs)
        data_back = round_trip(aw)
        astype_res = np.array(np.nan).astype(np.int32)
        assert_array_equal(data_back, [astype_res, 99])


def test_byte_orders():
    arr = np.arange(10, dtype=np.int32)
    # Test endian read/write of types not requiring scaling
    for tp in (np.uint64, np.float64, np.complex128):
        dt = np.dtype(tp)
        for code in '<>':
            ndt = dt.newbyteorder(code)
            for klass in (SlopeInterArrayWriter, SlopeArrayWriter, ArrayWriter):
                aw = klass(arr, ndt)
                data_back = round_trip(aw)
                assert_array_almost_equal(arr, data_back)


def test_writers_roundtrip():
    ndt = np.dtype(np.float64)
    arr = np.arange(3, dtype=ndt)
    # intercept
    aw = SlopeInterArrayWriter(arr, ndt, calc_scale=False)
    aw.inter = 1.0
    data_back = round_trip(aw)
    assert_array_equal(data_back, arr)
    # scaling
    aw.slope = 2.0
    data_back = round_trip(aw)
    assert_array_equal(data_back, arr)
    # if there is no valid data, we get zeros
    aw = SlopeInterArrayWriter(arr + np.nan, np.int32)
    data_back = round_trip(aw)
    assert_array_equal(data_back, np.zeros(arr.shape))
    # infs generate ints at same value as max
    arr[0] = np.inf
    aw = SlopeInterArrayWriter(arr, np.int32)
    data_back = round_trip(aw)
    assert_array_almost_equal(data_back, [2, 1, 2])


def test_to_float():
    start, stop = 0, 100
    for in_type in NUMERIC_TYPES:
        step = 1 if in_type in IUINT_TYPES else 0.5
        info = type_info(in_type)
        mn, mx = info['min'], info['max']
        arr = np.arange(start, stop, step, dtype=in_type)
        arr[0] = mn
        arr[-1] = mx
        for out_type in CFLOAT_TYPES:
            out_info = type_info(out_type)
            for klass in (SlopeInterArrayWriter, SlopeArrayWriter, ArrayWriter):
                if in_type in COMPLEX_TYPES and out_type in FLOAT_TYPES:
                    with pytest.raises(WriterError):
                        klass(arr, out_type)
                    continue
                aw = klass(arr, out_type)
                assert aw.array is arr
                assert aw.out_dtype == out_type
                arr_back = round_trip(aw)
                assert_array_equal(arr.astype(out_type), arr_back)
                # Check too-big values overflowed correctly
                out_min, out_max = out_info['min'], out_info['max']
                assert np.all(arr_back[arr > out_max] == np.inf)
                assert np.all(arr_back[arr < out_min] == -np.inf)


def test_dumber_writers():
    arr = np.arange(10, dtype=np.float64)
    aw = SlopeArrayWriter(arr)
    aw.slope = 2.0
    assert aw.slope == 2.0
    with pytest.raises(AttributeError):
        aw.inter
    aw = ArrayWriter(arr)
    with pytest.raises(AttributeError):
        aw.slope
    with pytest.raises(AttributeError):
        aw.inter
    # Attempt at scaling should raise error for dumb type
    with pytest.raises(WriterError):
        ArrayWriter(arr, np.int16)


def test_writer_maker():
    arr = np.arange(10, dtype=np.float64)
    aw = make_array_writer(arr, np.float64)
    assert isinstance(aw, SlopeInterArrayWriter)
    aw = make_array_writer(arr, np.float64, True, True)
    assert isinstance(aw, SlopeInterArrayWriter)
    aw = make_array_writer(arr, np.float64, True, False)
    assert isinstance(aw, SlopeArrayWriter)
    aw = make_array_writer(arr, np.float64, False, False)
    assert isinstance(aw, ArrayWriter)
    with pytest.raises(ValueError):
        make_array_writer(arr, np.float64, False)
    with pytest.raises(ValueError):
        make_array_writer(arr, np.float64, False, True)
    # Does calc_scale get run by default?
    aw = make_array_writer(arr, np.int16, calc_scale=False)
    assert (aw.slope, aw.inter) == (1, 0)
    aw.calc_scale()
    slope, inter = aw.slope, aw.inter
    assert not (slope, inter) == (1, 0)
    # Should run by default
    aw = make_array_writer(arr, np.int16)
    assert (aw.slope, aw.inter) == (slope, inter)
    aw = make_array_writer(arr, np.int16, calc_scale=True)
    assert (aw.slope, aw.inter) == (slope, inter)


def test_float_int_min_max():
    # Conversion between float and int
    for in_dt in FLOAT_TYPES:
        finf = type_info(in_dt)
        arr = np.array([finf['min'], finf['max']], dtype=in_dt)
        # Bug in numpy 1.6.2 on PPC leading to infs - abort
        if not np.all(np.isfinite(arr)):
            print(f'Hit PPC max -> inf bug; skip in_type {in_dt}')
            continue
        for out_dt in IUINT_TYPES:
            try:
                with suppress_warnings():  # overflow
                    aw = SlopeInterArrayWriter(arr, out_dt)
            except ScalingError:
                continue
            arr_back_sc = round_trip(aw)
            assert np.allclose(arr, arr_back_sc)


def test_int_int_min_max():
    # Conversion between (u)int and (u)int
    eps = np.finfo(np.float64).eps
    rtol = 1e-6
    for in_dt in IUINT_TYPES:
        iinf = np.iinfo(in_dt)
        arr = np.array([iinf.min, iinf.max], dtype=in_dt)
        for out_dt in IUINT_TYPES:
            try:
                aw = SlopeInterArrayWriter(arr, out_dt)
            except ScalingError:
                continue
            arr_back_sc = round_trip(aw)
            # integer allclose
            adiff = int_abs(arr - arr_back_sc)
            rdiff = adiff / (arr + eps)
            assert np.all(rdiff < rtol)


def test_int_int_slope():
    # Conversion between (u)int and (u)int for slopes only
    eps = np.finfo(np.float64).eps
    rtol = 1e-7
    for in_dt in IUINT_TYPES:
        iinf = np.iinfo(in_dt)
        for out_dt in IUINT_TYPES:
            kinds = np.dtype(in_dt).kind + np.dtype(out_dt).kind
            if kinds in ('ii', 'uu', 'ui'):
                arrs = (np.array([iinf.min, iinf.max], dtype=in_dt),)
            elif kinds == 'iu':
                arrs = (np.array([iinf.min, 0], dtype=in_dt), np.array([0, iinf.max], dtype=in_dt))
            for arr in arrs:
                try:
                    aw = SlopeArrayWriter(arr, out_dt)
                except ScalingError:
                    continue
                assert not aw.slope == 0
                arr_back_sc = round_trip(aw)
                # integer allclose
                adiff = int_abs(arr - arr_back_sc)
                rdiff = adiff / (arr + eps)
                assert np.all(rdiff < rtol)


def test_float_int_spread():
    # Test rounding error for spread of values
    powers = np.arange(-10, 10, 0.5)
    arr = np.concatenate((-(10**powers), 10**powers))
    for in_dt in (np.float32, np.float64):
        arr_t = arr.astype(in_dt)
        for out_dt in IUINT_TYPES:
            aw = SlopeInterArrayWriter(arr_t, out_dt)
            arr_back_sc = round_trip(aw)
            # Get estimate for error
            max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, aw.slope, aw.inter)
            # Simulate allclose test with large atol
            diff = np.abs(arr_t - arr_back_sc)
            rdiff = diff / np.abs(arr_t)
            assert np.all((diff <= max_miss) | (rdiff <= 1e-5))


def rt_err_estimate(arr_t, out_dtype, slope, inter):
    # Error attributable to rounding
    slope = 1 if slope is None else slope
    inter = 1 if inter is None else inter
    max_int_miss = slope / 2.0
    # Estimate error attributable to floating point slope / inter;
    # Remove inter / slope, put in a float type to simulate the type
    # promotion for the multiplication, apply slope / inter
    flt_there = (arr_t - inter) / slope
    flt_back = flt_there.astype(out_dtype) * slope + inter
    max_flt_miss = np.abs(arr_t - flt_back).max()
    # Max error is sum of rounding and fp error
    return max_int_miss + max_flt_miss


def test_rt_bias():
    # Check for bias in round trip
    rng = np.random.RandomState(20111214)
    mu, std, count = 100, 10, 100
    arr = rng.normal(mu, std, size=(count,))
    eps = np.finfo(np.float32).eps
    for in_dt in (np.float32, np.float64):
        arr_t = arr.astype(in_dt)
        for out_dt in IUINT_TYPES:
            aw = SlopeInterArrayWriter(arr_t, out_dt)
            arr_back_sc = round_trip(aw)
            bias = np.mean(arr_t - arr_back_sc)
            # Get estimate for error
            max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, aw.slope, aw.inter)
            # Hokey use of max_miss as a std estimate
            bias_thresh = np.max([max_miss / np.sqrt(count), eps])
            assert np.abs(bias) < bias_thresh


def test_nan2zero_scaling():
    # Scaling needs to take into account whether nan can be represented as zero
    # in the input data (before scaling).
    # nan can be represented as zero of we can store (0 - intercept) / divslope
    # in the output data - because reading back the data as `stored_array  * divslope +
    # intercept` will reconstruct zeros for the nans in the original input.
    #
    # Make array requiring scaling for which range does not cover zero -> arr
    # Append nan to arr -> nan_arr
    # Append 0 to arr -> zero_arr
    # Write / read nan_arr, zero_arr
    # Confirm nan, 0 generated same output value
    for awt, in_dt, out_dt, sign in itertools.product(
        (SlopeArrayWriter, SlopeInterArrayWriter),
        FLOAT_TYPES,
        IUINT_TYPES,
        (-1, 1),
    ):
        # Use fixed-up type information to avoid bugs, especially on PPC
        in_info = type_info(in_dt)
        out_info = type_info(out_dt)
        # Skip impossible combinations
        if in_info['min'] == 0 and sign == -1:
            continue
        mx = min(in_info['max'], out_info['max'] * 2.0, 2**32)
        vals = [np.nan] + [100, mx]
        nan_arr = np.array(vals, dtype=in_dt) * sign
        # Check that nan scales to same value as zero within same array
        nan_arr_0 = np.array([0] + vals, dtype=in_dt) * sign
        # Check that nan scales to almost the same value as zero in another array
        zero_arr = np.nan_to_num(nan_arr)
        nan_aw = awt(nan_arr, out_dt, nan2zero=True)
        back_nan = round_trip(nan_aw) * float(sign)
        nan_0_aw = awt(nan_arr_0, out_dt, nan2zero=True)
        back_nan_0 = round_trip(nan_0_aw) * float(sign)
        zero_aw = awt(zero_arr, out_dt, nan2zero=True)
        back_zero = round_trip(zero_aw) * float(sign)
        assert np.allclose(back_nan[1:], back_zero[1:])
        assert_array_equal(back_nan[1:], back_nan_0[2:])
        assert np.abs(back_nan[0] - back_zero[0]) < 1e-2
        assert back_nan_0[0] == back_nan_0[1]


def test_finite_range_nan():
    # Test finite range method and has_nan property
    for in_arr, res in (
        ([[-1, 0, 1], [np.inf, np.nan, -np.inf]], (-1, 1)),
        (np.array([[-1, 0, 1], [np.inf, np.nan, -np.inf]]), (-1, 1)),
        ([[np.nan], [np.nan]], (np.inf, -np.inf)),  # all nans slices
        (np.zeros((3, 4, 5)) + np.nan, (np.inf, -np.inf)),
        ([[-np.inf], [np.inf]], (np.inf, -np.inf)),  # all infs slices
        (np.zeros((3, 4, 5)) + np.inf, (np.inf, -np.inf)),
        ([[np.nan, -1, 2], [-2, np.nan, 1]], (-2, 2)),
        ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)),
        ([[-np.inf, 2], [np.nan, 1]], (1, 2)),  # good max case
        ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)),
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
    ):
        for awt, kwargs in (
            (ArrayWriter, dict(check_scaling=False)),
            (SlopeArrayWriter, {}),
            (SlopeArrayWriter, dict(calc_scale=False)),
            (SlopeInterArrayWriter, {}),
            (SlopeInterArrayWriter, dict(calc_scale=False)),
        ):
            for out_type in NUMERIC_TYPES:
                has_nan = np.any(np.isnan(in_arr))
                try:
                    aw = awt(in_arr, out_type, **kwargs)
                except WriterError:
                    continue
                # Should not matter about the order of finite range method call
                # and has_nan property - test this is true
                assert aw.has_nan == has_nan
                assert aw.finite_range() == res
                aw = awt(in_arr, out_type, **kwargs)
                assert aw.finite_range() == res
                assert aw.has_nan == has_nan
                # Check float types work as complex
                in_arr = np.array(in_arr)
                if in_arr.dtype.kind == 'f':
                    c_arr = in_arr.astype(np.complex128)
                    try:
                        aw = awt(c_arr, out_type, **kwargs)
                    except WriterError:
                        continue
                    aw = awt(c_arr, out_type, **kwargs)
                    assert aw.has_nan == has_nan
                    assert aw.finite_range() == res
            # Structured type cannot be nan and we can test this
            a = np.array([[1.0, 0, 1], [2, 3, 4]]).view([('f1', 'f')])
            aw = awt(a, a.dtype, **kwargs)
            with pytest.raises(TypeError):
                aw.finite_range()
            assert not aw.has_nan
