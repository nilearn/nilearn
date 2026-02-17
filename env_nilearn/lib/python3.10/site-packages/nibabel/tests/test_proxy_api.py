# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Validate image proxy API

Minimum array proxy API is:

* read only ``shape`` property
* read only ``is_proxy`` property set to True
* returns array from ``np.asarray(prox)``
* returns array slice from ``prox[<slice_spec>]`` where ``<slice_spec>`` is any
  non-fancy slice specification.

And:

* that modifying no object outside ``prox`` will affect the result of
  ``np.asarray(obj)``.  Specifically:
  * Changes in position (``obj.tell()``) of any passed file-like objects
    will not affect the output of from ``np.asarray(proxy)``.
  * if you pass a header into the __init__, then modifying the original
    header will not affect the result of the array return.

These last are to allow the proxy to be reused with different images.
"""

import unittest
import warnings
from io import BytesIO
from itertools import product
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import ecat, minc1, minc2, parrec
from ..analyze import AnalyzeHeader
from ..arrayproxy import ArrayProxy, is_proxy
from ..casting import have_binary128, sctypes
from ..externals.netcdf import netcdf_file
from ..freesurfer.mghformat import MGHHeader
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spm2analyze import Spm2AnalyzeHeader
from ..spm99analyze import Spm99AnalyzeHeader
from ..testing import assert_dt_equal, clear_and_catch_warnings
from ..testing import data_path as DATA_PATH
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import apply_read_scaling
from .test_api_validators import ValidateAPI
from .test_parrec import EG_REC, VARY_REC

h5py, have_h5py, _ = optional_package('h5py')

try:
    from numpy.exceptions import ComplexWarning
except ModuleNotFoundError:  # NumPy < 1.25
    from numpy import ComplexWarning


def _some_slicers(shape):
    ndim = len(shape)
    slicers = np.eye(ndim, dtype=int).astype(object)
    slicers[slicers == 0] = slice(None)
    for i in range(ndim):
        if i % 2:
            slicers[i, i] = -1
        elif shape[i] < 2:  # some proxy examples have length 1 axes
            slicers[i, i] = 0
    # Add a newaxis to keep us on our toes
    no_pos = ndim // 2
    slicers = np.hstack(
        (
            slicers[:, :no_pos],
            np.empty((ndim, 1)),
            slicers[:, no_pos:],
        )
    )
    slicers[:, no_pos] = None
    return [tuple(s) for s in slicers]


class _TestProxyAPI(ValidateAPI):
    """Base class for testing proxy APIs

    Assumes that real classes will provide an `obj_params` method which is a
    generator returning 2 tuples of (<proxy_maker>, <param_dict>).
    <proxy_maker> is a function returning a 3 tuple of (<proxy>, <fileobj>,
    <header>).  <param_dict> is a dictionary containing at least keys
    ``arr_out`` (expected output array from proxy), ``dtype_out`` (expected
    output dtype for array) and ``shape`` (shape of array).

    The <header> above should support at least "get_data_dtype",
    "set_data_dtype", "get_data_shape", "set_data_shape"
    """

    # Flag True if offset can be set into header of image
    settable_offset = False

    def validate_shape(self, pmaker, params):
        # Check shape
        prox, fio, hdr = pmaker()
        assert_array_equal(prox.shape, params['shape'])
        # Read only
        with pytest.raises(AttributeError):
            prox.shape = params['shape']

    def validate_ndim(self, pmaker, params):
        # Check shape
        prox, fio, hdr = pmaker()
        assert prox.ndim == len(params['shape'])
        # Read only
        with pytest.raises(AttributeError):
            prox.ndim = len(params['shape'])

    def validate_is_proxy(self, pmaker, params):
        # Check shape
        prox, fio, hdr = pmaker()
        assert prox.is_proxy
        assert is_proxy(prox)
        assert not is_proxy(np.arange(10))
        # Read only
        with pytest.raises(AttributeError):
            prox.is_proxy = False

    def validate_asarray(self, pmaker, params):
        # Check proxy returns expected array from asarray
        prox, fio, hdr = pmaker()
        out = np.asarray(prox)
        assert_array_equal(out, params['arr_out'])
        assert_dt_equal(out.dtype, params['dtype_out'])
        # Shape matches expected shape
        assert out.shape == params['shape']

    def validate_array_interface_with_dtype(self, pmaker, params):
        # Check proxy returns expected array from asarray
        prox, fio, hdr = pmaker()
        orig = np.array(prox, dtype=None)
        assert_array_equal(orig, params['arr_out'])
        assert_dt_equal(orig.dtype, params['dtype_out'])

        context = None
        if np.issubdtype(orig.dtype, np.complexfloating):
            context = clear_and_catch_warnings()
            context.__enter__()
            warnings.simplefilter('ignore', ComplexWarning)

        for dtype in sctypes['float'] + sctypes['int'] + sctypes['uint']:
            # Directly coerce with a dtype
            direct = dtype(prox)
            # Half-precision is imprecise. Obviously. It's a bad idea, but don't break
            # the test over it.
            rtol = 1e-03 if dtype == np.float16 else 1e-05
            assert_allclose(direct, orig.astype(dtype), rtol=rtol, atol=1e-08)
            assert_dt_equal(direct.dtype, np.dtype(dtype))
            assert direct.shape == params['shape']
            # All three methods should produce equivalent results
            for arrmethod in (np.array, np.asarray, np.asanyarray):
                out = arrmethod(prox, dtype=dtype)
                assert_array_equal(out, direct)
                assert_dt_equal(out.dtype, np.dtype(dtype))
                # Shape matches expected shape
                assert out.shape == params['shape']

        if context is not None:
            context.__exit__()

    def validate_header_isolated(self, pmaker, params):
        # Confirm altering input header has no effect
        # Depends on header providing 'get_data_dtype', 'set_data_dtype',
        # 'get_data_shape', 'set_data_shape', 'set_data_offset'
        prox, fio, hdr = pmaker()
        assert_array_equal(prox, params['arr_out'])
        # Mess up header badly and hope for same correct result
        if hdr.get_data_dtype() == np.uint8:
            hdr.set_data_dtype(np.int16)
        else:
            hdr.set_data_dtype(np.uint8)
        hdr.set_data_shape(np.array(hdr.get_data_shape()) + 1)
        if self.settable_offset:
            hdr.set_data_offset(32)
        assert_array_equal(prox, params['arr_out'])

    def validate_fileobj_isolated(self, pmaker, params):
        # Check file position of read independent of file-like object
        prox, fio, hdr = pmaker()
        if isinstance(fio, str):
            return
        assert_array_equal(prox, params['arr_out'])
        fio.read()  # move to end of file
        assert_array_equal(prox, params['arr_out'])

    def validate_proxy_slicing(self, pmaker, params):
        # Confirm that proxy object can be sliced correctly
        arr = params['arr_out']
        shape = arr.shape
        prox, fio, hdr = pmaker()
        for sliceobj in _some_slicers(shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])


class TestAnalyzeProxyAPI(_TestProxyAPI):
    """Specific Analyze-type array proxy API test

    The analyze proxy extends the general API by adding read-only attributes
    ``slope, inter, offset``
    """

    proxy_class = ArrayProxy
    header_class = AnalyzeHeader
    shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))
    has_slope = False
    has_inter = False
    data_dtypes = (np.uint8, np.int16, np.int32, np.float32, np.complex64, np.float64)
    array_order = 'F'
    # Cannot set offset for Freesurfer
    settable_offset = True
    # Freesurfer enforces big-endian. '=' means use native
    data_endian = '='

    def obj_params(self):
        """Iterator returning (``proxy_creator``, ``proxy_params``) pairs

        Each pair will be tested separately.

        ``proxy_creator`` is a function taking no arguments and returning (fresh
        proxy object, fileobj, header).  We need to pass this function rather
        than a proxy instance so we can recreate the proxy objects fresh for
        each of multiple tests run from the ``validate_xxx`` autogenerated test
        methods.  This allows the tests to modify the proxy instance without
        having an effect on the later tests in the same function.
        """
        # Analyze and up wrap binary arrays, Fortran ordered, with given offset
        # and dtype and shape.
        if not self.settable_offset:
            offsets = (self.header_class().get_data_offset(),)
        else:
            offsets = (0, 16)
        # For non-integral parameters, cast to float32 value can be losslessly cast
        # later, enabling exact checks, then back to float for consistency
        slopes = (1.0, 2.0, float(np.float32(3.1416))) if self.has_slope else (1.0,)
        inters = (0.0, 10.0, float(np.float32(2.7183))) if self.has_inter else (0.0,)
        for shape, dtype, offset, slope, inter in product(
            self.shapes,
            self.data_dtypes,
            offsets,
            slopes,
            inters,
        ):
            n_els = np.prod(shape)
            dtype = np.dtype(dtype).newbyteorder(self.data_endian)
            arr = np.arange(n_els, dtype=dtype).reshape(shape)
            data = arr.tobytes(order=self.array_order)
            hdr = self.header_class()
            hdr.set_data_dtype(dtype)
            hdr.set_data_shape(shape)
            if self.settable_offset:
                hdr.set_data_offset(offset)
            if (slope, inter) == (1, 0):  # No scaling applied
                # dtype from array
                dtype_out = dtype
            else:  # scaling or offset applied
                # out dtype predictable from apply_read_scaling
                # and datatypes of slope, inter
                hdr.set_slope_inter(slope, inter)
                s, i = hdr.get_slope_inter()
                tmp = apply_read_scaling(arr, 1.0 if s is None else s, 0.0 if i is None else i)
                dtype_out = tmp.dtype.type

            def sio_func():
                fio = BytesIO()
                fio.truncate(0)
                fio.seek(offset)
                fio.write(data)
                # Use a copy of the header to avoid changing
                # global header in test functions.
                new_hdr = hdr.copy()
                return (self.proxy_class(fio, new_hdr), fio, new_hdr)

            params = dict(
                dtype=dtype,
                dtype_out=dtype_out,
                arr=arr.copy(),
                arr_out=arr.astype(dtype_out) * slope + inter,
                shape=shape,
                offset=offset,
                slope=slope,
                inter=inter,
            )
            yield sio_func, params
            # Same with filenames
            with InTemporaryDirectory():
                fname = 'data.bin'

                def fname_func():
                    with open(fname, 'wb') as fio:
                        fio.seek(offset)
                        fio.write(data)
                    # Use a copy of the header to avoid changing
                    # global header in test functions.
                    new_hdr = hdr.copy()
                    return (self.proxy_class(fname, new_hdr), fname, new_hdr)

                params = params.copy()
                yield fname_func, params

    def validate_dtype(self, pmaker, params):
        # Read-only dtype attribute
        prox, fio, hdr = pmaker()
        assert_dt_equal(prox.dtype, params['dtype'])
        with pytest.raises(AttributeError):
            prox.dtype = np.dtype(prox.dtype)

    def validate_slope_inter_offset(self, pmaker, params):
        # Check slope, inter, offset
        prox, fio, hdr = pmaker()
        for attr_name in ('slope', 'inter', 'offset'):
            expected = params[attr_name]
            assert_array_equal(getattr(prox, attr_name), expected)
            # Read only
            with pytest.raises(AttributeError):
                setattr(prox, attr_name, expected)


class TestSpm99AnalyzeProxyAPI(TestAnalyzeProxyAPI):
    # SPM-type analyze has slope scaling but not intercept
    header_class = Spm99AnalyzeHeader
    has_slope = True


class TestSpm2AnalyzeProxyAPI(TestSpm99AnalyzeProxyAPI):
    header_class = Spm2AnalyzeHeader


class TestNifti1ProxyAPI(TestSpm99AnalyzeProxyAPI):
    header_class = Nifti1Header
    has_inter = True
    data_dtypes = (
        np.uint8,
        np.int16,
        np.int32,
        np.float32,
        np.complex64,
        np.float64,
        np.int8,
        np.uint16,
        np.uint32,
        np.int64,
        np.uint64,
        np.complex128,
    )
    if have_binary128():
        data_dtypes += (np.float128, np.complex256)


class TestMGHAPI(TestAnalyzeProxyAPI):
    header_class = MGHHeader
    shapes = ((2, 3, 4), (2, 3, 4, 5))  # MGH can only do >= 3D
    has_slope = False
    has_inter = False
    settable_offset = False
    data_endian = '>'
    data_dtypes = (np.uint8, np.int16, np.int32, np.float32)


class TestMinc1API(_TestProxyAPI):
    module = minc1
    file_class = minc1.Minc1File
    eg_fname = 'tiny.mnc'
    eg_shape = (10, 20, 20)

    @staticmethod
    def opener(f):
        return netcdf_file(f, mode='r')

    def obj_params(self):
        """Iterator returning (``proxy_creator``, ``proxy_params``) pairs

        Each pair will be tested separately.

        ``proxy_creator`` is a function taking no arguments and returning (fresh
        proxy object, fileobj, header).  We need to pass this function rather
        than a proxy instance so we can recreate the proxy objects fresh for
        each of multiple tests run from the ``validate_xxx`` autogenerated test
        methods.  This allows the tests to modify the proxy instance without
        having an effect on the later tests in the same function.
        """
        eg_path = pjoin(DATA_PATH, self.eg_fname)
        arr_out = self.file_class(self.opener(eg_path)).get_scaled_data()

        def eg_func():
            mf = self.file_class(self.opener(eg_path))
            prox = minc1.MincImageArrayProxy(mf)
            img = self.module.load(eg_path)
            fobj = open(eg_path, 'rb')
            return prox, fobj, img.header

        yield (eg_func, dict(shape=self.eg_shape, dtype_out=np.float64, arr_out=arr_out))


if have_h5py:

    class TestMinc2API(TestMinc1API):
        module = minc2
        file_class = minc2.Minc2File
        eg_fname = 'small.mnc'
        eg_shape = (18, 28, 29)

        @staticmethod
        def opener(f):
            return h5py.File(f, mode='r')


class TestEcatAPI(_TestProxyAPI):
    eg_fname = 'tinypet.v'
    eg_shape = (10, 10, 3, 1)

    def obj_params(self):
        eg_path = pjoin(DATA_PATH, self.eg_fname)
        img = ecat.load(eg_path)
        arr_out = img.get_fdata()

        def eg_func():
            img = ecat.load(eg_path)
            sh = img.get_subheaders()
            prox = ecat.EcatImageArrayProxy(sh)
            fobj = open(eg_path, 'rb')
            return prox, fobj, sh

        yield (eg_func, dict(shape=self.eg_shape, dtype_out=np.float64, arr_out=arr_out))

    def validate_header_isolated(self, pmaker, params):
        raise unittest.SkipTest('ECAT header does not support dtype get')


class TestPARRECAPI(_TestProxyAPI):
    def _func_dict(self, rec_name):
        img = parrec.load(rec_name)
        arr_out = img.get_fdata()

        def eg_func():
            img = parrec.load(rec_name)
            prox = parrec.PARRECArrayProxy(rec_name, img.header, scaling='dv')
            fobj = open(rec_name, 'rb')
            return prox, fobj, img.header

        return (eg_func, dict(shape=img.shape, dtype_out=np.float64, arr_out=arr_out))

    def obj_params(self):
        yield self._func_dict(EG_REC)
        yield self._func_dict(VARY_REC)
