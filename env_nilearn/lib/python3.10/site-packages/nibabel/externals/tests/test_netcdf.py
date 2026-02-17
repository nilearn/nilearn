""" Tests for netcdf """

import os
from os.path import join as pjoin, dirname
from io import BytesIO
from glob import glob
from contextlib import contextmanager

import numpy as np

import pytest

from ..netcdf import netcdf_file

TEST_DATA_PATH = pjoin(dirname(__file__), 'data')

N_EG_ELS = 11  # number of elements for example variable
VARTYPE_EG = 'b'  # var type for example variable


@contextmanager
def make_simple(*args, **kwargs):
    f = netcdf_file(*args, **kwargs)
    f.history = 'Created for a test'
    f.createDimension('time', N_EG_ELS)
    time = f.createVariable('time', VARTYPE_EG, ('time',))
    time[:] = np.arange(N_EG_ELS)
    time.units = 'days since 2008-01-01'
    f.flush()
    yield f
    f.close()


def assert_simple_truths(ncfileobj):
    assert ncfileobj.history == b'Created for a test'
    time = ncfileobj.variables['time']
    assert time.units == b'days since 2008-01-01'
    assert time.shape == (N_EG_ELS,)
    assert time[-1] == N_EG_ELS - 1


def test_read_write_files(tmp_path):
    fname = str(tmp_path / 'simple.nc')

    with make_simple(fname, 'w') as f:
        pass
    # To read the NetCDF file we just created::
    with netcdf_file(fname) as f:
        # Using mmap is the default
        assert f.use_mmap
        assert_simple_truths(f)

    # Now without mmap
    with netcdf_file(fname, mmap=False) as f:
        # Using mmap is the default
        assert not f.use_mmap
        assert_simple_truths(f)

    # To read the NetCDF file we just created, as file object, no
    # mmap.  When n * n_bytes(var_type) is not divisible by 4, this
    # raised an error in pupynere 1.0.12 and scipy rev 5893, because
    # calculated vsize was rounding up in units of 4 - see
    # https://www.unidata.ucar.edu/software/netcdf/docs/netcdf.html
    fobj = open(fname, 'rb')
    with netcdf_file(fobj) as f:
        # by default, don't use mmap for file-like
        assert not f.use_mmap
        assert_simple_truths(f)


def test_read_write_sio():
    eg_sio1 = BytesIO()
    with make_simple(eg_sio1, 'w') as f1:
        str_val = eg_sio1.getvalue()

    eg_sio2 = BytesIO(str_val)
    with netcdf_file(eg_sio2) as f2:
        assert_simple_truths(f2)

    # Test that error is raised if attempting mmap for sio
    eg_sio3 = BytesIO(str_val)
    with pytest.raises(ValueError):
        netcdf_file(eg_sio3, 'r', True)
    # Test 64-bit offset write / read
    eg_sio_64 = BytesIO()
    with make_simple(eg_sio_64, 'w', version=2) as f_64:
        str_val = eg_sio_64.getvalue()

    eg_sio_64 = BytesIO(str_val)
    with netcdf_file(eg_sio_64) as f_64:
        assert_simple_truths(f_64)
        assert f_64.version_byte == 2
    # also when version 2 explicitly specified
    eg_sio_64 = BytesIO(str_val)
    with netcdf_file(eg_sio_64, version=2) as f_64:
        assert_simple_truths(f_64)
        assert f_64.version_byte == 2


def test_read_example_data():
    # read any example data files
    for fname in glob(pjoin(TEST_DATA_PATH, '*.nc')):
        with netcdf_file(fname, 'r') as f:
            pass
        with netcdf_file(fname, 'r', mmap=False) as f:
            pass


def test_itemset_no_segfault_on_readonly():
    # Regression test for ticket #1202.
    # Open the test file in read-only mode.
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
    with netcdf_file(filename, 'r') as f:
        time_var = f.variables['time']

    # time_var.assignValue(42) should raise a RuntimeError--not seg. fault!
    with pytest.raises(RuntimeError):
        time_var.assignValue(42)


def test_write_invalid_dtype():
    dtypes = ['int64', 'uint64']
    if np.dtype('int').itemsize == 8:   # 64-bit machines
        dtypes.append('int')
    if np.dtype('uint').itemsize == 8:   # 64-bit machines
        dtypes.append('uint')

    with netcdf_file(BytesIO(), 'w') as f:
        f.createDimension('time', N_EG_ELS)
        for dt in dtypes:
            with pytest.raises(ValueError):
                f.createVariable('time', dt, ('time',))


def test_flush_rewind():
    stream = BytesIO()
    with make_simple(stream, mode='w') as f:
        x = f.createDimension('x', 4)
        v = f.createVariable('v', 'i2', ['x'])
        v[:] = 1
        f.flush()
        len_single = len(stream.getvalue())
        f.flush()
        len_double = len(stream.getvalue())

    assert len_single == len_double


def test_dtype_specifiers():
    # Numpy 1.7.0-dev had a bug where 'i2' wouldn't work.
    # Specifying np.int16 or similar only works from the same commit as this
    # comment was made.
    with make_simple(BytesIO(), mode='w') as f:
        f.createDimension('x',4)
        f.createVariable('v1', 'i2', ['x'])
        f.createVariable('v2', np.int16, ['x'])
        f.createVariable('v3', np.dtype(np.int16), ['x'])


def test_ticket_1720():
    io = BytesIO()

    items = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    with netcdf_file(io, 'w') as f:
        f.history = 'Created for a test'
        f.createDimension('float_var', 10)
        float_var = f.createVariable('float_var', 'f', ('float_var',))
        float_var[:] = items
        float_var.units = 'metres'
        f.flush()
        contents = io.getvalue()

    io = BytesIO(contents)
    with netcdf_file(io, 'r') as f:
        assert f.history == b'Created for a test'
        float_var = f.variables['float_var']
        assert float_var.units == b'metres'
        assert float_var.shape == (10,)
        assert np.allclose(float_var[:], items)
