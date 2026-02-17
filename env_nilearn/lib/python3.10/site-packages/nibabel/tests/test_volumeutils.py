# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test for volumeutils module"""

import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version

from nibabel.testing import (
    assert_allclose_safely,
    assert_dt_equal,
    error_warnings,
    suppress_warnings,
)

from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
    _dt_min_max,
    _ftype4scaled_finite,
    _is_compressed_fobj,
    _write_data,
    apply_read_scaling,
    array_from_file,
    array_to_file,
    best_write_scale_ftype,
    better_float_of,
    fname_ext_ul_case,
    int_scinter_ftype,
    make_dt_codes,
    native_code,
    rec2dict,
    seek_tell,
    shape_zoom_affine,
    working_type,
    write_zeros,
)

pyzstd, HAVE_ZSTD, _ = optional_package('pyzstd')

# convenience variables for numpy types
FLOAT_TYPES = sctypes['float']
COMPLEX_TYPES = sctypes['complex']
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
INT_TYPES = sctypes['int']
IUINT_TYPES = INT_TYPES + sctypes['uint']
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES

FP_RUNTIME_WARN = Version(np.__version__) >= Version('1.24.0.dev0+239')
NP_2 = Version(np.__version__) >= Version('2.0.0.dev0')

try:
    from numpy.exceptions import ComplexWarning
except ModuleNotFoundError:  # NumPy < 1.25
    from numpy import ComplexWarning


def test__is_compressed_fobj():
    # _is_compressed helper function
    with InTemporaryDirectory():
        file_openers = [('', open, False), ('.gz', gzip.open, True), ('.bz2', BZ2File, True)]
        if HAVE_ZSTD:
            file_openers += [('.zst', pyzstd.ZstdFile, True)]
        for ext, opener, compressed in file_openers:
            fname = 'test.bin' + ext
            for mode in ('wb', 'rb'):
                fobj = opener(fname, mode)
                assert _is_compressed_fobj(fobj) == compressed
                fobj.close()


def test_fobj_string_assumptions():
    # Test assumptions made in array_from_file about whether string returned
    # from file read needs a copy.
    dtype = np.dtype(np.int32)

    def make_array(n, bytes):
        arr = np.ndarray(n, dtype, buffer=bytes)
        arr.flags.writeable = True
        return arr

    # Check whether file, gzip file, bz2, zst file reread memory from cache
    fname = 'test.bin'
    with InTemporaryDirectory():
        openers = [open, gzip.open, BZ2File]
        if HAVE_ZSTD:
            openers += [pyzstd.ZstdFile]
        for n, opener in itertools.product((256, 1024, 2560, 25600), openers):
            in_arr = np.arange(n, dtype=dtype)
            # Write array to file
            fobj_w = opener(fname, 'wb')
            fobj_w.write(in_arr.tobytes())
            fobj_w.close()
            # Read back from file
            fobj_r = opener(fname, 'rb')
            try:
                contents1 = bytearray(4 * n)
                fobj_r.readinto(contents1)
                # Second element is 1
                assert contents1[0:8] != b'\x00' * 8
                out_arr = make_array(n, contents1)
                assert_array_equal(in_arr, out_arr)
                # Set second element to 0
                out_arr[1] = 0
                # Show this changed the bytes string
                assert contents1[:8] == b'\x00' * 8
                # Reread, to get unmodified contents
                fobj_r.seek(0)
                contents2 = bytearray(4 * n)
                fobj_r.readinto(contents2)
                out_arr2 = make_array(n, contents2)
                assert_array_equal(in_arr, out_arr2)
                assert out_arr[1] == 0
            finally:
                fobj_r.close()
            os.unlink(fname)


def test_array_from_file():
    shape = (2, 3, 4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    # Check on string buffers
    offset = 0
    assert buf_chk(in_arr, BytesIO(), None, offset)
    offset = 10
    assert buf_chk(in_arr, BytesIO(), None, offset)
    # check on real file
    fname = 'test.bin'
    with InTemporaryDirectory():
        # fortran ordered
        out_buf = open(fname, 'wb')
        in_buf = open(fname, 'rb')
        assert buf_chk(in_arr, out_buf, in_buf, offset)
        # Drop offset to check that shape's not coming from file length
        out_buf.seek(0)
        in_buf.seek(0)
        offset = 5
        assert buf_chk(in_arr, out_buf, in_buf, offset)
        del out_buf, in_buf
    # Make sure empty shape, and zero length, give empty arrays
    arr = array_from_file((), np.dtype('f8'), BytesIO())
    assert len(arr) == 0
    arr = array_from_file((0,), np.dtype('f8'), BytesIO())
    assert len(arr) == 0
    # Check error from small file
    with pytest.raises(OSError):
        array_from_file(shape, dtype, BytesIO())
    # check on real file
    fd, fname = tempfile.mkstemp()
    with InTemporaryDirectory():
        open(fname, 'wb').write(b'1')
        in_buf = open(fname, 'rb')
        with pytest.raises(OSError):
            array_from_file(shape, dtype, in_buf)
        del in_buf


def test_array_from_file_mmap():
    # Test memory mapping
    shape = (2, 21)
    with InTemporaryDirectory():
        for dt in (np.int16, np.float64):
            arr = np.arange(np.prod(shape), dtype=dt).reshape(shape)
            with open('test.bin', 'wb') as fobj:
                fobj.write(arr.tobytes(order='F'))
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj)
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'c'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap=True)
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'c'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap='c')
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'c'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap='r')
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'r'
            with open('test.bin', 'rb+') as fobj:
                res = array_from_file(shape, dt, fobj, mmap='r+')
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'r+'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap=False)
                assert_array_equal(res, arr)
                assert not isinstance(res, np.memmap)
            with open('test.bin', 'rb') as fobj:
                with pytest.raises(ValueError):
                    array_from_file(shape, dt, fobj, mmap='p')


def buf_chk(in_arr, out_buf, in_buf, offset):
    """Write contents of in_arr into fileobj, read back, check same"""
    instr = b' ' * offset + in_arr.tobytes(order='F')
    out_buf.write(instr)
    out_buf.flush()
    if in_buf is None:  # we're using in_buf from out_buf
        out_buf.seek(0)
        in_buf = out_buf
    arr = array_from_file(in_arr.shape, in_arr.dtype, in_buf, offset)
    return np.allclose(in_arr, arr)


def test_array_from_file_openers():
    # Test array_from_file also works with Opener objects
    shape = (2, 3, 4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    with InTemporaryDirectory():
        extensions = ['', '.gz', '.bz2']
        if HAVE_ZSTD:
            extensions += ['.zst']
        for ext, offset in itertools.product(extensions, (0, 5, 10)):
            fname = 'test.bin' + ext
            with Opener(fname, 'wb') as out_buf:
                if offset != 0:  # avoid https://bugs.python.org/issue16828
                    out_buf.write(b' ' * offset)
                out_buf.write(in_arr.tobytes(order='F'))
            with Opener(fname, 'rb') as in_buf:
                out_arr = array_from_file(shape, dtype, in_buf, offset)
                assert_array_almost_equal(in_arr, out_arr)
            # Delete object holding onto file for Windows
            del out_arr


def test_array_from_file_reread():
    # Check that reading, modifying, reading again returns original.
    # This is the live check for the generic checks in
    # test_fobj_string_assumptions
    offset = 9
    fname = 'test.bin'
    with InTemporaryDirectory():
        openers = [open, gzip.open, bz2.BZ2File, BytesIO]
        if HAVE_ZSTD:
            openers += [pyzstd.ZstdFile]
        for shape, opener, dtt, order in itertools.product(
            ((64,), (64, 65), (64, 65, 66)), openers, (np.int16, np.float32), ('F', 'C')
        ):
            n_els = np.prod(shape)
            in_arr = np.arange(n_els, dtype=dtt).reshape(shape)
            is_bio = hasattr(opener, 'getvalue')
            # Write array to file
            fobj_w = opener() if is_bio else opener(fname, 'wb')
            fobj_w.write(b' ' * offset)
            fobj_w.write(in_arr.tobytes(order=order))
            if is_bio:
                fobj_r = fobj_w
            else:
                fobj_w.close()
                fobj_r = opener(fname, 'rb')
            # Read back from file
            try:
                out_arr = array_from_file(shape, dtt, fobj_r, offset, order)
                assert_array_equal(in_arr, out_arr)
                out_arr[..., 0] = -1
                assert not np.allclose(in_arr, out_arr)
                out_arr2 = array_from_file(shape, dtt, fobj_r, offset, order)
                assert_array_equal(in_arr, out_arr2)
            finally:
                fobj_r.close()
            # Delete arrays holding onto file objects so Windows can delete
            del out_arr, out_arr2
            if not is_bio:
                os.unlink(fname)


def test_array_to_file():
    arr = np.arange(10).reshape(5, 2)
    str_io = BytesIO()
    for tp in (np.uint64, np.float64, np.complex128):
        dt = np.dtype(tp)
        for code in '<>':
            ndt = dt.newbyteorder(code)
            for allow_intercept in (True, False):
                scale, intercept, mn, mx = _calculate_scale(arr, ndt, allow_intercept)
                data_back = write_return(arr, str_io, ndt, 0, intercept, scale)
                assert_array_almost_equal(arr, data_back)
    # Test array-like
    str_io = BytesIO()
    array_to_file(arr.tolist(), str_io, float)
    data_back = array_from_file(arr.shape, float, str_io)
    assert_array_almost_equal(arr, data_back)


def test_a2f_intercept_scale():
    arr = np.array([0.0, 1.0, 2.0])
    str_io = BytesIO()
    # intercept
    data_back = write_return(arr, str_io, np.float64, 0, 1.0)
    assert_array_equal(data_back, arr - 1)
    # scaling
    data_back = write_return(arr, str_io, np.float64, 0, 1.0, 2.0)
    assert_array_equal(data_back, (arr - 1) / 2.0)


def test_a2f_upscale():
    # Test working type scales with needed range
    info = type_info(np.float32)
    # Test values discovered from stress testing.  The largish value (2**115)
    # overflows to inf after the intercept is subtracted, using float32 as the
    # working precision.  The difference between inf and this value is lost.
    arr = np.array([[info['min'], 2**115, info['max']]], dtype=np.float32)
    slope = np.float32(2**121)
    inter = info['min']
    str_io = BytesIO()
    # We need to provide mn, mx for function to be able to calculate upcasting
    array_to_file(
        arr, str_io, np.uint8, intercept=inter, divslope=slope, mn=info['min'], mx=info['max']
    )
    raw = array_from_file(arr.shape, np.uint8, str_io)
    back = apply_read_scaling(raw, slope, inter)
    top = back - arr
    score = np.abs(top / arr)
    assert np.all(score < 10)


def test_a2f_min_max():
    # Check min and max thresholding of array to file
    str_io = BytesIO()
    for in_dt in (np.float32, np.int8):
        for out_dt in (np.float32, np.int8):
            arr = np.arange(4, dtype=in_dt)
            # min thresholding
            with np.errstate(invalid='ignore'):
                data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1)
            assert_array_equal(data_back, [1, 1, 2, 3])
            # max thresholding
            with np.errstate(invalid='ignore'):
                data_back = write_return(arr, str_io, out_dt, 0, 0, 1, None, 2)
            assert_array_equal(data_back, [0, 1, 2, 2])
            # min max thresholding
            data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1, 2)
            assert_array_equal(data_back, [1, 1, 2, 2])
    # Check that works OK with scaling and intercept
    arr = np.arange(4, dtype=np.float32)
    data_back = write_return(arr, str_io, int, 0, -1, 0.5, 1, 2)
    assert_array_equal(data_back * 0.5 - 1, [1, 1, 2, 2])
    # Even when scaling is negative
    data_back = write_return(arr, str_io, int, 0, 1, -0.5, 1, 2)
    assert_array_equal(data_back * -0.5 + 1, [1, 1, 2, 2])
    # Check complex numbers
    arr = np.arange(4, dtype=np.complex64) + 100j
    with suppress_warnings():  # cast to real
        data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1, 2)
    assert_array_equal(data_back, [1, 1, 2, 2])


def test_a2f_order():
    ndt = np.dtype(np.float64)
    arr = np.array([0.0, 1.0, 2.0])
    str_io = BytesIO()
    # order makes no difference in 1D case
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, [0.0, 1.0, 2.0])
    # but does in the 2D case
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    data_back = write_return(arr, str_io, ndt, order='F')
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, arr.T)


def test_a2f_nan2zero():
    ndt = np.dtype(np.float64)
    str_io = BytesIO()
    # nans set to 0 for integer output case, not float
    arr = np.array([[np.nan, 0], [0, np.nan]])
    data_back = write_return(arr, str_io, ndt)  # float, thus no effect
    assert_array_equal(data_back, arr)
    # True is the default, but just to show it's possible
    data_back = write_return(arr, str_io, ndt, nan2zero=True)
    assert_array_equal(data_back, arr)
    with np.errstate(invalid='ignore'):
        data_back = write_return(arr, str_io, np.int64, nan2zero=True)
    assert_array_equal(data_back, [[0, 0], [0, 0]])
    # otherwise things get a bit weird; tidied here
    # How weird?  Look at arr.astype(np.int64)
    with np.errstate(invalid='ignore'):
        data_back = write_return(arr, str_io, np.int64, nan2zero=False)
        assert_array_equal(data_back, arr.astype(np.int64))


def test_a2f_nan2zero_scaling():
    # Check that nan gets translated to the nearest equivalent to zero
    #
    # nan can be represented as zero of we can store (0 - intercept) / divslope
    # in the output data - because reading back the data as `stored_array  * divslope +
    # intercept` will reconstruct zeros for the nans in the original input.
    #
    # Check with array containing nan, matching array containing zero and
    # Array containing zero
    # Array values otherwise not including zero without scaling
    # Same with negative sign
    # Array values including zero before scaling but not after
    bio = BytesIO()
    for in_dt, out_dt, zero_in, inter in itertools.product(
        FLOAT_TYPES, IUINT_TYPES, (True, False), (0, -100)
    ):
        in_info = np.finfo(in_dt)
        out_info = np.iinfo(out_dt)
        mx = min(in_info.max, out_info.max * 2.0, 2**32) + inter
        mn = 0 if zero_in or inter else 100
        vals = [np.nan] + [mn, mx]
        nan_arr = np.array(vals, dtype=in_dt)
        zero_arr = np.nan_to_num(nan_arr)
        with np.errstate(invalid='ignore'):
            back_nan = write_return(nan_arr, bio, np.int64, intercept=inter)
            back_zero = write_return(zero_arr, bio, np.int64, intercept=inter)
        assert_array_equal(back_nan, back_zero)


def test_a2f_offset():
    # check that non-zero file offset works
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    str_io = BytesIO()
    str_io.write(b'a' * 42)
    array_to_file(arr, str_io, np.float64, 42)
    data_back = array_from_file(arr.shape, np.float64, str_io, 42)
    assert_array_equal(data_back, arr.astype(np.float64))
    # And that offset=None respected
    str_io.truncate(22)
    str_io.seek(22)
    array_to_file(arr, str_io, np.float64, None)
    data_back = array_from_file(arr.shape, np.float64, str_io, 22)
    assert_array_equal(data_back, arr.astype(np.float64))


def test_a2f_dtype_default():
    # that default dtype is input dtype
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    str_io = BytesIO()
    array_to_file(arr.astype(np.int16), str_io)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    assert_array_equal(data_back, arr.astype(np.int16))


def test_a2f_zeros():
    # Check that, if there is no valid data, we get zeros
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    str_io = BytesIO()
    # With slope=None signal
    array_to_file(arr + np.inf, str_io, np.int32, 0, 0.0, None)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))
    # With  mn, mx = 0 signal
    array_to_file(arr, str_io, np.int32, 0, 0.0, 1.0, 0, 0)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))
    # With  mx < mn signal
    array_to_file(arr, str_io, np.int32, 0, 0.0, 1.0, 4, 2)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))


def test_a2f_big_scalers():
    # Check that clip works even for overflowing scalers / data
    info = type_info(np.float32)
    arr = np.array([info['min'], 0, info['max']], dtype=np.float32)
    str_io = BytesIO()
    # Intercept causes overflow - does routine scale correctly?
    # We check whether the routine correctly clips extreme values.
    # We need nan2zero=False because we can't represent 0 in the input, given
    # the scaling and the output range.
    with suppress_warnings():  # overflow
        array_to_file(arr, str_io, np.int8, intercept=np.float32(2**120), nan2zero=False)
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, -128, 127])
    # Scales also if mx, mn specified? Same notes and complaints as for the test
    # above.
    str_io.seek(0)
    array_to_file(
        arr,
        str_io,
        np.int8,
        mn=info['min'],
        mx=info['max'],
        intercept=np.float32(2**120),
        nan2zero=False,
    )
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, -128, 127])
    # And if slope causes overflow?
    str_io.seek(0)
    with suppress_warnings():  # overflow in divide
        array_to_file(arr, str_io, np.int8, divslope=np.float32(0.5))
        data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])
    # with mn, mx specified?
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, mn=info['min'], mx=info['max'], divslope=np.float32(0.5))
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])


def test_a2f_int_scaling():
    # Check that we can use integers for intercept and divslope
    arr = np.array([0, 1, 128, 255], dtype=np.uint8)
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, np.uint8, intercept=1)
    assert_array_equal(back_arr, np.clip(arr - 1.0, 0, 255))
    back_arr = write_return(arr, fobj, np.uint8, divslope=2)
    assert_array_equal(back_arr, np.round(np.clip(arr / 2.0, 0, 255)))
    back_arr = write_return(arr, fobj, np.uint8, intercept=1, divslope=2)
    assert_array_equal(back_arr, np.round(np.clip((arr - 1.0) / 2.0, 0, 255)))
    back_arr = write_return(arr, fobj, np.int16, intercept=1, divslope=2)
    assert_array_equal(back_arr, np.round((arr - 1.0) / 2.0))


def test_a2f_scaled_unscaled():
    # Test behavior of array_to_file when writing different types with and
    # without scaling
    fobj = BytesIO()
    for in_dtype, out_dtype, intercept, divslope in itertools.product(
        NUMERIC_TYPES, NUMERIC_TYPES, (0, 0.5, -1, 1), (1, 0.5, 2)
    ):
        mn_in, mx_in = _dt_min_max(in_dtype)
        vals = [mn_in, 0, 1, mx_in]
        if np.dtype(in_dtype).kind != 'u':
            vals.append(-1)
        if in_dtype in CFLOAT_TYPES:
            vals.append(np.nan)
        arr = np.array(vals, dtype=in_dtype)
        mn_out, mx_out = _dt_min_max(out_dtype)
        # 0 when scaled to output will also be the output value for NaN
        nan_fill = -intercept / divslope
        if out_dtype in IUINT_TYPES:
            nan_fill = np.round(nan_fill)
        # nan2zero will check whether 0 in scaled to a valid value in output
        if in_dtype in CFLOAT_TYPES and not mn_out <= nan_fill <= mx_out:
            with pytest.raises(ValueError):
                array_to_file(
                    arr, fobj, out_dtype=out_dtype, divslope=divslope, intercept=intercept
                )
            continue
        with suppress_warnings():
            back_arr = write_return(
                arr, fobj, out_dtype=out_dtype, divslope=divslope, intercept=intercept
            )
            exp_back = arr.copy()
            if (
                in_dtype in IUINT_TYPES
                and out_dtype in IUINT_TYPES
                and (intercept, divslope) == (0, 1)
            ):
                # Direct iu to iu casting.
                # Need to clip if ranges not the same.
                # Use smaller of input, output range to avoid np.clip upcasting
                # the array because of large clip limits.
                if (mn_in, mx_in) != (mn_out, mx_out):
                    exp_back = np.clip(exp_back, max(mn_in, mn_out), min(mx_in, mx_out))
            else:  # Need to deal with nans, casting to float, clipping
                if in_dtype in CFLOAT_TYPES and out_dtype in IUINT_TYPES:
                    exp_back[np.isnan(exp_back)] = 0
                if in_dtype not in COMPLEX_TYPES:
                    exp_back = exp_back.astype(float)
                if intercept != 0:
                    exp_back -= intercept
                if divslope != 1:
                    exp_back /= divslope
                if exp_back.dtype.type in CFLOAT_TYPES and out_dtype in IUINT_TYPES:
                    exp_back = np.round(exp_back).astype(float)
                    exp_back = np.clip(exp_back, *shared_range(float, out_dtype))
            exp_back = exp_back.astype(out_dtype)
        # Allow for small differences in large numbers
        assert_allclose_safely(back_arr, exp_back)


def test_a2f_nanpos():
    # Strange behavior of nan2zero
    arr = np.array([np.nan])
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, np.int8, divslope=2)
    assert_array_equal(back_arr, 0)
    back_arr = write_return(arr, fobj, np.int8, intercept=10, divslope=2)
    assert_array_equal(back_arr, -5)


def test_a2f_bad_scaling():
    # Test that pathological scalers raise an error
    NUMERICAL_TYPES = sum((sctypes[key] for key in ['int', 'uint', 'float', 'complex']), [])
    for in_type, out_type, slope, inter in itertools.product(
        NUMERICAL_TYPES,
        NUMERICAL_TYPES,
        (None, 1, 0, np.nan, -np.inf, np.inf),
        (0, np.nan, -np.inf, np.inf),
    ):
        arr = np.ones((2,), dtype=in_type)
        fobj = BytesIO()
        cm = error_warnings()
        if np.issubdtype(in_type, np.complexfloating) and not np.issubdtype(
            out_type, np.complexfloating
        ):
            cm = pytest.warns(ComplexWarning)
        if (slope, inter) == (1, 0):
            with cm:
                assert_array_equal(
                    arr, write_return(arr, fobj, out_type, intercept=inter, divslope=slope)
                )
        elif (slope, inter) == (None, 0):
            assert_array_equal(
                0, write_return(arr, fobj, out_type, intercept=inter, divslope=slope)
            )
        else:
            with pytest.raises(ValueError):
                array_to_file(arr, fobj, np.int8, intercept=inter, divslope=slope)


def test_a2f_nan2zero_range():
    # array_to_file should check if nan can be represented as zero
    # This comes about when the writer can't write the value (-intercept /
    # divslope) because it does not fit in the output range.  Input clipping
    # should not affect this
    fobj = BytesIO()
    # No problem for input integer types - they don't have NaNs
    for dt in INT_TYPES:
        arr_no_nan = np.array([-1, 0, 1, 2], dtype=dt)
        # No errors from explicit thresholding (nor for input float types)
        back_arr = write_return(arr_no_nan, fobj, np.int8, mn=1, nan2zero=True)
        assert_array_equal([1, 1, 1, 2], back_arr)
        back_arr = write_return(arr_no_nan, fobj, np.int8, mx=-1, nan2zero=True)
        assert_array_equal([-1, -1, -1, -1], back_arr)
        # Pushing zero outside the output data range does not generate error
        back_arr = write_return(arr_no_nan, fobj, np.int8, intercept=129, nan2zero=True)
        assert_array_equal([-128, -128, -128, -127], back_arr)
        back_arr = write_return(
            arr_no_nan, fobj, np.int8, intercept=257.1, divslope=2, nan2zero=True
        )
        assert_array_equal([-128, -128, -128, -128], back_arr)
    for dt in CFLOAT_TYPES:
        arr = np.array([-1, 0, 1, np.nan], dtype=dt)
        # Error occurs for arrays without nans too
        arr_no_nan = np.array([-1, 0, 1, 2], dtype=dt)
        complex_warn = (ComplexWarning,) if np.issubdtype(dt, np.complexfloating) else ()
        # Casting nan to int will produce a RuntimeWarning in numpy 1.24
        nan_warn = (RuntimeWarning,) if FP_RUNTIME_WARN else ()
        c_and_n_warn = complex_warn + nan_warn
        # No errors from explicit thresholding
        # mn thresholding excluding zero
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            assert_array_equal([1, 1, 1, 0], write_return(arr, fobj, np.int8, mn=1))
        # mx thresholding excluding zero
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            assert_array_equal([-1, -1, -1, 0], write_return(arr, fobj, np.int8, mx=-1))
        # Errors from datatype threshold after scaling
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=128)
        assert_array_equal([-128, -128, -127, -128], back_arr)
        with pytest.raises(ValueError):
            write_return(arr, fobj, np.int8, intercept=129)
        with pytest.raises(ValueError):
            write_return(arr_no_nan, fobj, np.int8, intercept=129)
        # OK with nan2zero false, but we get whatever nan casts to
        with pytest.warns(c_and_n_warn) if c_and_n_warn else error_warnings():
            nan_cast = np.array(np.nan, dtype=dt).astype(np.int8)
        with pytest.warns(c_and_n_warn) if c_and_n_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=129, nan2zero=False)
        assert_array_equal([-128, -128, -128, nan_cast], back_arr)
        # divslope
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=256, divslope=2)
        assert_array_equal([-128, -128, -128, -128], back_arr)
        with pytest.raises(ValueError):
            write_return(arr, fobj, np.int8, intercept=257.1, divslope=2)
        with pytest.raises(ValueError):
            write_return(arr_no_nan, fobj, np.int8, intercept=257.1, divslope=2)
        # OK with nan2zero false
        with pytest.warns(c_and_n_warn) if c_and_n_warn else error_warnings():
            back_arr = write_return(
                arr, fobj, np.int8, intercept=257.1, divslope=2, nan2zero=False
            )
        assert_array_equal([-128, -128, -128, nan_cast], back_arr)


def test_a2f_non_numeric():
    # Reminder that we may get structured dtypes
    dt = np.dtype([('f1', 'f'), ('f2', 'i2')])
    arr = np.zeros((2,), dtype=dt)
    arr['f1'] = 0.4, 0.6
    arr['f2'] = 10, 12
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, dt)
    assert_array_equal(back_arr, arr)
    # Some versions of numpy can cast structured types to float, others not
    try:
        arr.astype(float)
    except (TypeError, ValueError):
        pass
    else:
        back_arr = write_return(arr, fobj, float)
        assert_array_equal(back_arr, arr.astype(float))
    # mn, mx never work for structured types
    with pytest.raises(ValueError):
        write_return(arr, fobj, float, mn=0)
    with pytest.raises(ValueError):
        write_return(arr, fobj, float, mx=10)


def write_return(data, fileobj, out_dtype, *args, **kwargs):
    fileobj.truncate(0)
    fileobj.seek(0)
    array_to_file(data, fileobj, out_dtype, *args, **kwargs)
    data = array_from_file(data.shape, out_dtype, fileobj)
    return data


def test_apply_scaling():
    # Null scaling, same array returned
    arr = np.zeros((3,), dtype=np.int16)
    assert apply_read_scaling(arr) is arr
    assert apply_read_scaling(arr, np.float64(1.0)) is arr
    assert apply_read_scaling(arr, inter=np.float64(0)) is arr
    f32, f64 = np.float32, np.float64
    f32_arr = np.zeros((1,), dtype=f32)
    i16_arr = np.zeros((1,), dtype=np.int16)
    # Check float upcast (not the normal numpy scalar rule)
    # This is the normal rule - no upcast from Python scalar
    assert (f32_arr * 1.0).dtype == np.float32
    assert (f32_arr + 1.0).dtype == np.float32
    # This is the normal rule - no upcast from scalar
    # before NumPy 2.0, after 2.0, it upcasts
    want_dtype = np.float64 if NP_2 else np.float32
    assert (f32_arr * f64(1)).dtype == want_dtype
    assert (f32_arr + f64(1)).dtype == want_dtype
    # The function does upcast though
    ret = apply_read_scaling(np.float32(0), np.float64(2))
    assert ret.dtype == np.float64
    ret = apply_read_scaling(np.float32(0), inter=np.float64(2))
    assert ret.dtype == np.float64
    # Check integer inf upcast
    big = f32(type_info(f32)['max'])
    # Normally this would not upcast
    assert (i16_arr * big).dtype == np.float32
    # An equivalent case is a little hard to find for the intercept
    nmant_32 = type_info(np.float32)['nmant']
    big_delta = np.float32(2 ** (floor_log2(big) - nmant_32))
    assert (i16_arr * big_delta + big).dtype == np.float32
    # Upcasting does occur with this routine
    assert apply_read_scaling(i16_arr, big).dtype == np.float64
    assert apply_read_scaling(i16_arr, big_delta, big).dtype == np.float64
    # If float32 passed, no overflow, float32 returned
    assert apply_read_scaling(np.int8(0), f32(-1.0), f32(0.0)).dtype == np.float32
    # float64 passed, float64 returned
    assert apply_read_scaling(np.int8(0), -1.0, 0.0).dtype == np.float64
    # float32 passed, overflow, float64 returned
    assert apply_read_scaling(np.int8(0), f32(1e38), f32(0.0)).dtype == np.float64
    assert apply_read_scaling(np.int8(0), f32(-1e38), f32(0.0)).dtype == np.float64
    # Non-zero intercept still generates floats
    assert_dt_equal(apply_read_scaling(i16_arr, 1.0, 1.0).dtype, float)
    assert_dt_equal(apply_read_scaling(np.zeros((1,), dtype=np.int32), 1.0, 1.0).dtype, float)
    assert_dt_equal(apply_read_scaling(np.zeros((1,), dtype=np.int64), 1.0, 1.0).dtype, float)


def test_apply_read_scaling_ints():
    # Test that apply_read_scaling copes with integer scaling inputs
    arr = np.arange(10, dtype=np.int16)
    assert_array_equal(apply_read_scaling(arr, 1, 0), arr)
    assert_array_equal(apply_read_scaling(arr, 1, 1), arr + 1)
    assert_array_equal(apply_read_scaling(arr, 2, 1), arr * 2 + 1)


def test_apply_read_scaling_nones():
    # Check that we can pass None as slope and inter to apply read scaling
    arr = np.arange(10, dtype=np.int16)
    assert_array_equal(apply_read_scaling(arr, None, None), arr)
    assert_array_equal(apply_read_scaling(arr, 2, None), arr * 2)
    assert_array_equal(apply_read_scaling(arr, None, 1), arr + 1)


def test_int_scinter():
    # Finding float type needed for applying scale, offset to ints
    assert int_scinter_ftype(np.int8, 1.0, 0.0) == np.float32
    assert int_scinter_ftype(np.int8, -1.0, 0.0) == np.float32
    assert int_scinter_ftype(np.int8, 1e38, 0.0) == np.float64
    assert int_scinter_ftype(np.int8, -1e38, 0.0) == np.float64


def test_working_type():
    # Which type do input types with slope and inter cast to in numpy?
    # Wrapper function because we need to use the dtype str for comparison.  We
    # need this because of the very confusing np.int32 != np.intp (on 32 bit).
    def wt(*args, **kwargs):
        return np.dtype(working_type(*args, **kwargs)).str

    d1 = np.atleast_1d
    for in_type in NUMERIC_TYPES:
        in_ts = np.dtype(in_type).str
        assert wt(in_type) == in_ts
        assert wt(in_type, 1, 0) == in_ts
        assert wt(in_type, 1.0, 0.0) == in_ts
        in_val = d1(in_type(0))
        for slope_type in NUMERIC_TYPES:
            sl_val = slope_type(1)  # no scaling, regardless of type
            assert wt(in_type, sl_val, 0.0) == in_ts
            sl_val = slope_type(2)  # actual scaling
            out_val = in_val / d1(sl_val)
            assert wt(in_type, sl_val) == out_val.dtype.str
            for inter_type in NUMERIC_TYPES:
                i_val = inter_type(0)  # no scaling, regardless of type
                assert wt(in_type, 1, i_val) == in_ts
                i_val = inter_type(1)  # actual scaling
                out_val = in_val - d1(i_val)
                assert wt(in_type, 1, i_val) == out_val.dtype.str
                # Combine scaling and intercept
                out_val = (in_val - d1(i_val)) / d1(sl_val)
                assert wt(in_type, sl_val, i_val) == out_val.dtype.str
    # Confirm that type codes and dtypes work as well
    f32s = np.dtype(np.float32).str
    assert wt('f4', 1, 0) == f32s
    assert wt(np.dtype('f4'), 1, 0) == f32s


def test_better_float():
    # Better float function
    def check_against(f1, f2):
        return f1 if FLOAT_TYPES.index(f1) >= FLOAT_TYPES.index(f2) else f2

    for first in FLOAT_TYPES:
        for other in IUINT_TYPES + sctypes['complex']:
            assert better_float_of(first, other) == first
            assert better_float_of(other, first) == first
            for other2 in IUINT_TYPES + sctypes['complex']:
                assert better_float_of(other, other2) == np.float32
                assert better_float_of(other, other2, np.float64) == np.float64
        for second in FLOAT_TYPES:
            assert better_float_of(first, second) == check_against(first, second)
    # Check codes and dtypes work
    assert better_float_of('f4', 'f8', 'f4') == np.float64
    assert better_float_of('i4', 'i8', 'f8') == np.float64


def test_best_write_scale_ftype():
    # Test best write scaling type
    # Types return better of (default, array type) unless scale overflows.
    # Return float type cannot be less capable than the input array type
    for dtt in IUINT_TYPES + FLOAT_TYPES:
        arr = np.arange(10, dtype=dtt)
        assert best_write_scale_ftype(arr, 1, 0) == better_float_of(dtt, np.float32)
        assert best_write_scale_ftype(arr, 1, 0, np.float64) == better_float_of(dtt, np.float64)
        assert best_write_scale_ftype(arr, np.float32(2), 0) == better_float_of(dtt, np.float32)
        assert best_write_scale_ftype(arr, 1, np.float32(1)) == better_float_of(dtt, np.float32)
    # Overflowing ints with scaling results in upcast
    best_vals = ((np.float32, np.float64),)
    if np.longdouble in OK_FLOATS:
        best_vals += ((np.float64, np.longdouble),)
    for lower_t, higher_t in best_vals:
        # Information on this float
        L_info = type_info(lower_t)
        t_max = L_info['max']
        nmant = L_info['nmant']  # number of significand digits
        big_delta = lower_t(2 ** (floor_log2(t_max) - nmant))  # delta below max
        # Even large values that don't overflow don't change output
        arr = np.array([0, t_max], dtype=lower_t)
        assert best_write_scale_ftype(arr, 1, 0) == lower_t
        # Scaling > 1 reduces output values, so no upcast needed
        assert best_write_scale_ftype(arr, lower_t(1.01), 0) == lower_t
        # Scaling < 1 increases values, so upcast may be needed (and is here)
        assert best_write_scale_ftype(arr, lower_t(0.99), 0) == higher_t
        # Large minus offset on large array can cause upcast
        assert best_write_scale_ftype(arr, 1, -big_delta / 2.01) == lower_t
        assert best_write_scale_ftype(arr, 1, -big_delta / 2.0) == higher_t
        # With infs already in input, default type returns
        arr[0] = np.inf
        assert best_write_scale_ftype(arr, lower_t(0.5), 0) == lower_t
        arr[0] = -np.inf
        assert best_write_scale_ftype(arr, lower_t(0.5), 0) == lower_t


def test_write_zeros():
    bio = BytesIO()
    write_zeros(bio, 10000)
    assert bio.getvalue() == b'\x00' * 10000
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 10000, 256)
    assert bio.getvalue() == b'\x00' * 10000
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 200, 256)
    assert bio.getvalue() == b'\x00' * 200


def test_seek_tell():
    # Test seek tell routine
    bio = BytesIO()
    in_files = [bio, 'test.bin', 'test.gz', 'test.bz2']
    if HAVE_ZSTD:
        in_files += ['test.zst']
    start = 10
    end = 100
    diff = end - start
    tail = 7
    with InTemporaryDirectory():
        for in_file, write0 in itertools.product(in_files, (False, True)):
            st = functools.partial(seek_tell, write0=write0)
            bio.seek(0)
            # First write the file
            with ImageOpener(in_file, 'wb') as fobj:
                assert fobj.tell() == 0
                # already at position - OK
                st(fobj, 0)
                assert fobj.tell() == 0
                # Move position by writing
                fobj.write(b'\x01' * start)
                assert fobj.tell() == start
                # Files other than BZ2Files can seek forward on write, leaving
                # zeros in their wake.  BZ2Files can't seek when writing,
                # unless we enable the write0 flag to seek_tell
                # ZstdFiles also does not support seek forward on write
                if not write0 and in_file in ('test.bz2', 'test.zst'):
                    # write the zeros by hand for the read test below
                    fobj.write(b'\x00' * diff)
                else:
                    st(fobj, end)
                    assert fobj.tell() == end
                # Write tail
                fobj.write(b'\x02' * tail)
            bio.seek(0)
            # Now read back the file testing seek_tell in reading mode
            with ImageOpener(in_file, 'rb') as fobj:
                assert fobj.tell() == 0
                st(fobj, 0)
                assert fobj.tell() == 0
                st(fobj, start)
                assert fobj.tell() == start
                st(fobj, end)
                assert fobj.tell() == end
                # Seek anywhere works in read mode for all files
                st(fobj, 0)
            bio.seek(0)
            # Check we have the expected written output
            with ImageOpener(in_file, 'rb') as fobj:
                assert fobj.read() == b'\x01' * start + b'\x00' * diff + b'\x02' * tail
        input_files = ['test2.gz', 'test2.bz2']
        if HAVE_ZSTD:
            input_files += ['test2.zst']
        for in_file in input_files:
            # Check failure of write seek backwards
            with ImageOpener(in_file, 'wb') as fobj:
                fobj.write(b'g' * 10)
                assert fobj.tell() == 10
                seek_tell(fobj, 10)
                assert fobj.tell() == 10
                with pytest.raises(OSError):
                    seek_tell(fobj, 5)
            # Make sure read seeks don't affect file
            with ImageOpener(in_file, 'rb') as fobj:
                seek_tell(fobj, 10)
                seek_tell(fobj, 0)
            with ImageOpener(in_file, 'rb') as fobj:
                assert fobj.read() == b'g' * 10


def test_seek_tell_logic():
    # Test logic of seek_tell write0 with dummy class
    # Seek works? OK
    bio = BytesIO()
    seek_tell(bio, 10)
    assert bio.tell() == 10

    class BabyBio(BytesIO):
        def seek(self, *args):
            raise OSError

    bio = BabyBio()
    # Fresh fileobj, position 0, can't seek - error
    with pytest.raises(OSError):
        bio.seek(10)
    # Put fileobj in correct position by writing
    ZEROB = b'\x00'
    bio.write(ZEROB * 10)
    seek_tell(bio, 10)  # already there, nothing to do
    assert bio.tell() == 10
    assert bio.getvalue() == ZEROB * 10
    # Try write zeros to get to new position
    with pytest.raises(OSError):
        bio.seek(20)
    seek_tell(bio, 20, write0=True)
    assert bio.getvalue() == ZEROB * 20


def test_fname_ext_ul_case():
    # Get filename ignoring the case of the filename extension
    with InTemporaryDirectory():
        with open('afile.TXT', 'w') as fobj:
            fobj.write('Interesting information')
        # OSX usually has case-insensitive file systems; Windows also
        os_cares_case = not exists('afile.txt')
        with open('bfile.txt', 'w') as fobj:
            fobj.write('More interesting information')
        # If there is no file, the case doesn't change
        assert fname_ext_ul_case('nofile.txt') == 'nofile.txt'
        assert fname_ext_ul_case('nofile.TXT') == 'nofile.TXT'
        # If there is a file, accept upper or lower case for ext
        if os_cares_case:
            assert fname_ext_ul_case('afile.txt') == 'afile.TXT'
            assert fname_ext_ul_case('bfile.TXT') == 'bfile.txt'
        else:
            assert fname_ext_ul_case('afile.txt') == 'afile.txt'
            assert fname_ext_ul_case('bfile.TXT') == 'bfile.TXT'
        assert fname_ext_ul_case('afile.TXT') == 'afile.TXT'
        assert fname_ext_ul_case('bfile.txt') == 'bfile.txt'
        # Not mixed case though
        assert fname_ext_ul_case('afile.TxT') == 'afile.TxT'


def test_shape_zoom_affine():
    shape = (3, 5, 7)
    zooms = (3, 2, 1)
    res = shape_zoom_affine(shape, zooms)
    exp = np.array(
        [
            [-3.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine((3, 5), (3, 2))
    exp = np.array(
        [
            [-3.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine(shape, zooms, False)
    exp = np.array(
        [
            [3.0, 0.0, 0.0, -3.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert_array_almost_equal(res, exp)


def test_rec2dict():
    r = np.zeros((), dtype=[('x', 'i4'), ('s', 'S10')])
    d = rec2dict(r)
    assert d == {'x': 0, 's': b''}


def test_dtypes():
    # numpy - at least up to 1.5.1 - has odd behavior for hashing -
    # specifically:
    # In [9]: hash(dtype('<f4')) == hash(dtype('<f4').newbyteorder('<'))
    # Out[9]: False
    # In [10]: dtype('<f4') == dtype('<f4').newbyteorder('<')
    # Out[10]: True
    # where '<' is the native byte order
    dt_defs = ((16, 'float32', np.float32),)
    dtr = make_dt_codes(dt_defs)
    # check we have the fields we were expecting
    assert dtr.value_set() == {16}
    assert dtr.fields == ('code', 'label', 'type', 'dtype', 'sw_dtype')
    # These of course should pass regardless of dtype
    assert dtr[np.float32] == 16
    assert dtr['float32'] == 16
    # These also pass despite dtype issue
    assert dtr[np.dtype(np.float32)] == 16
    assert dtr[np.dtype('f4')] == 16
    assert dtr[np.dtype('f4').newbyteorder('S')] == 16
    # But this one used to fail
    assert dtr[np.dtype('f4').newbyteorder(native_code)] == 16
    # Check we can pass in niistring as well
    dt_defs = ((16, 'float32', np.float32, 'ASTRING'),)
    dtr = make_dt_codes(dt_defs)
    assert dtr[np.dtype('f4').newbyteorder('S')] == 16
    assert dtr.value_set() == {16}
    assert dtr.fields == ('code', 'label', 'type', 'niistring', 'dtype', 'sw_dtype')
    assert dtr.niistring[16] == 'ASTRING'
    # And that unequal elements raises error
    dt_defs = ((16, 'float32', np.float32, 'ASTRING'), (16, 'float32', np.float32))
    with pytest.raises(ValueError):
        make_dt_codes(dt_defs)
    # And that 2 or 5 elements raises error
    dt_defs = ((16, 'float32'),)
    with pytest.raises(ValueError):
        make_dt_codes(dt_defs)
    dt_defs = ((16, 'float32', np.float32, 'ASTRING', 'ANOTHERSTRING'),)
    with pytest.raises(ValueError):
        make_dt_codes(dt_defs)


def test__write_data():
    # Test private utility function for writing data
    itp = itertools.product

    def assert_rt(
        data,
        shape,
        out_dtype,
        order='F',
        in_cast=None,
        pre_clips=None,
        inter=0.0,
        slope=1.0,
        post_clips=None,
        nan_fill=None,
    ):
        sio = BytesIO()
        to_write = data.reshape(shape)
        # to check that we didn't modify in-place
        backup = to_write.copy()
        nan_positions = np.isnan(to_write)
        have_nans = np.any(nan_positions)
        if have_nans and nan_fill is None and not out_dtype.type == 'f':
            raise ValueError('Cannot handle this case')
        _write_data(
            to_write, sio, out_dtype, order, in_cast, pre_clips, inter, slope, post_clips, nan_fill
        )
        arr = np.ndarray(shape, out_dtype, buffer=sio.getvalue(), order=order)
        expected = to_write.copy()
        if have_nans and not nan_fill is None:
            expected[nan_positions] = nan_fill * slope + inter
        assert_array_equal(arr * slope + inter, expected)
        assert_array_equal(to_write, backup)

    # check shape writing
    for shape, order in itp(
        (
            (24,),
            (24, 1),
            (24, 1, 1),
            (1, 24),
            (1, 1, 24),
            (2, 3, 4),
            (6, 1, 4),
            (1, 6, 4),
            (6, 4, 1),
        ),
        'FC',
    ):
        assert_rt(np.arange(24), shape, np.int16, order=order)

    # check defense against modifying data in-place
    for in_cast, pre_clips, inter, slope, post_clips, nan_fill in itp(
        (None, np.float32),
        (None, (-1, 25)),
        (0.0, 1.0),
        (1.0, 0.5),
        (None, (-2, 49)),
        (None, 1),
    ):
        data = np.arange(24, dtype=np.float32)
        assert_rt(
            data,
            shape,
            np.int16,
            in_cast=in_cast,
            pre_clips=pre_clips,
            inter=inter,
            slope=slope,
            post_clips=post_clips,
            nan_fill=nan_fill,
        )
        # Check defense against in-place modification with nans present
        if not nan_fill is None:
            data[1] = np.nan
            assert_rt(
                data,
                shape,
                np.int16,
                in_cast=in_cast,
                pre_clips=pre_clips,
                inter=inter,
                slope=slope,
                post_clips=post_clips,
                nan_fill=nan_fill,
            )


def test_array_from_file_overflow():
    # Test for int overflow in size calculation in array_from_file
    shape = (1500,) * 6

    class NoStringIO:  # Null file-like for forcing error
        def seek(self, n_bytes):
            pass

        def read(self, n_bytes):
            return b''

    try:
        array_from_file(shape, np.int8, NoStringIO())
    except OSError as err:
        message = str(err)
    assert message == (
        'Expected 11390625000000000000 bytes, got 0 bytes from object\n'
        ' - could the file be damaged?'
    )


def test__ftype4scaled_finite_warningfilters():
    # This test checks our ability to properly manage the thread-unsafe
    # warnings filter list.

    # _ftype4scaled_finite always operates on one-or-two element arrays
    # Ensure that an overflow will happen for < float64
    finfo = np.finfo(np.float32)
    tst_arr = np.array((finfo.min, finfo.max), dtype=np.float32)

    go = threading.Event()
    stop = threading.Event()
    err = []

    class MakeTotalDestroy(threading.Thread):
        def run(self):
            # Restore the warnings filters when we're done testing
            with warnings.catch_warnings():
                go.set()
                while not stop.is_set():
                    warnings.filters[:] = []
                    time.sleep(0)

    class CheckScaling(threading.Thread):
        def run(self):
            go.wait()
            # Give ourselves a few bites at the apple
            # 200 loops through the function takes ~10ms
            # The highest number of iterations I've seen before hitting interference
            # is 131, with 99% under 30, so this should be reasonably reliable.
            for i in range(200):
                try:
                    # Use float16 to ensure two failures and increase time in function
                    _ftype4scaled_finite(tst_arr, 2.0, 1.0, default=np.float16)
                except Exception as e:
                    err.append(e)
                    break
            stop.set()

    thread_a = CheckScaling()
    thread_b = MakeTotalDestroy()
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    if err:
        raise err[0]


def _calculate_scale(data, out_dtype, allow_intercept):
    """Calculate scaling and optional intercept for data

    Copy of the deprecated volumeutils.calculate_scale, to preserve tests

    Parameters
    ----------
    data : array
    out_dtype : dtype
       output data type in some form understood by ``np.dtype``
    allow_intercept : bool
       If True allow non-zero intercept

    Returns
    -------
    scaling : None or float
       scalefactor to divide into data.  None if no valid data
    intercept : None or float
       intercept to subtract from data.  None if no valid data
    mn : None or float
       minimum of finite value in data or None if this will not
       be used to threshold data
    mx : None or float
       minimum of finite value in data, or None if this will not
       be used to threshold data
    """
    # Code here is a compatibility shell around arraywriters refactor
    in_dtype = data.dtype
    out_dtype = np.dtype(out_dtype)
    if np.can_cast(in_dtype, out_dtype):
        return 1.0, 0.0, None, None
    from ..arraywriters import WriterError, get_slope_inter, make_array_writer

    try:
        writer = make_array_writer(data, out_dtype, True, allow_intercept)
    except WriterError as e:
        raise ValueError(str(e))
    if out_dtype.kind in 'fc':
        return (1.0, 0.0, None, None)
    mn, mx = writer.finite_range()
    if (mn, mx) == (np.inf, -np.inf):  # No valid data
        return (None, None, None, None)
    if in_dtype.kind not in 'fc':
        mn, mx = (None, None)
    return get_slope_inter(writer) + (mn, mx)
