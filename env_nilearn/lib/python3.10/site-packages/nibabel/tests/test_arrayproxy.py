# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for arrayproxy module"""

import contextlib
import gzip
import pickle
from io import BytesIO
from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version

from .. import __version__
from ..arrayproxy import ArrayProxy, get_obj_dtype, is_proxy, reshape_dataobj
from ..deprecator import ExpiredDeprecationError
from ..nifti1 import Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..testing import memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
from .test_fileslice import slicer_samples
from .test_openers import patch_indexed_gzip


class FunkyHeader:
    def __init__(self, shape):
        self.shape = shape

    def get_data_shape(self):
        return self.shape[:]

    def get_data_dtype(self):
        return np.int32

    def get_data_offset(self):
        return 16

    def get_slope_inter(self):
        return 1.0, 0.0

    def copy(self):
        # Not needed when we remove header property
        return FunkyHeader(self.shape)


class CArrayProxy(ArrayProxy):
    # C array memory layout
    _default_order = 'C'


class DeprecatedCArrayProxy(ArrayProxy):
    # Used in test_deprecated_order_classvar. Remove when that test is removed (8.0)
    order = 'C'


def test_init():
    bio = BytesIO()
    shape = [2, 3, 4]
    dtype = np.int32
    arr = np.arange(24, dtype=dtype).reshape(shape)
    bio.seek(16)
    bio.write(arr.tobytes(order='F'))
    hdr = FunkyHeader(shape)
    ap = ArrayProxy(bio, hdr)
    assert ap.file_like is bio
    assert ap.shape == shape
    # shape should be read only
    with pytest.raises(AttributeError):
        ap.shape = shape
    # Get the data
    assert_array_equal(np.asarray(ap), arr)
    # Check we can modify the original header without changing the ap version
    hdr.shape[0] = 6
    assert ap.shape != shape
    # Data stays the same, also
    assert_array_equal(np.asarray(ap), arr)
    # You wouldn't do this, but order=None explicitly requests the default order
    ap2 = ArrayProxy(bio, FunkyHeader(arr.shape), order=None)
    assert_array_equal(np.asarray(ap2), arr)
    # C order also possible
    bio = BytesIO()
    bio.seek(16)
    bio.write(arr.tobytes(order='C'))
    ap = CArrayProxy(bio, FunkyHeader((2, 3, 4)))
    assert_array_equal(np.asarray(ap), arr)
    # Illegal init
    with pytest.raises(TypeError):
        ArrayProxy(bio, object())
    with pytest.raises(ValueError):
        ArrayProxy(bio, hdr, order='badval')


def test_tuplespec():
    bio = BytesIO()
    shape = [2, 3, 4]
    dtype = np.int32
    arr = np.arange(24, dtype=dtype).reshape(shape)
    bio.seek(16)
    bio.write(arr.tobytes(order='F'))
    # Create equivalent header and tuple specs
    hdr = FunkyHeader(shape)
    tuple_spec = (hdr.get_data_shape(), hdr.get_data_dtype(), hdr.get_data_offset(), 1.0, 0.0)
    ap_header = ArrayProxy(bio, hdr)
    ap_tuple = ArrayProxy(bio, tuple_spec)
    # Header and tuple specs produce identical behavior
    for prop in ('shape', 'dtype', 'offset', 'slope', 'inter', 'is_proxy'):
        assert getattr(ap_header, prop) == getattr(ap_tuple, prop)
    for method, args in (('get_unscaled', ()), ('__array__', ()), ('__getitem__', ((0, 2, 1),))):
        assert_array_equal(getattr(ap_header, method)(*args), getattr(ap_tuple, method)(*args))
    # Partial tuples of length 2-4 are also valid
    for n in range(2, 5):
        ArrayProxy(bio, tuple_spec[:n])
    # Bad tuple lengths
    with pytest.raises(TypeError):
        ArrayProxy(bio, ())
    with pytest.raises(TypeError):
        ArrayProxy(bio, tuple_spec[:1])
    with pytest.raises(TypeError):
        ArrayProxy(bio, tuple_spec + ('error',))


def write_raw_data(arr, hdr, fileobj):
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    fileobj.write(b'\x00' * hdr.get_data_offset())
    fileobj.write(arr.tobytes(order='F'))


def test_nifti1_init():
    bio = BytesIO()
    shape = (2, 3, 4)
    hdr = Nifti1Header()
    arr = np.arange(24, dtype=np.int16).reshape(shape)
    write_raw_data(arr, hdr, bio)
    hdr.set_slope_inter(2, 10)
    ap = ArrayProxy(bio, hdr)
    assert ap.file_like == bio
    assert ap.shape == shape
    # Get the data
    assert_array_equal(np.asarray(ap), arr * 2.0 + 10)
    with InTemporaryDirectory():
        f = open('test.nii', 'wb')
        write_raw_data(arr, hdr, f)
        f.close()
        ap = ArrayProxy('test.nii', hdr)
        assert ap.file_like == 'test.nii'
        assert ap.shape == shape
        assert_array_equal(np.asarray(ap), arr * 2.0 + 10)


@pytest.mark.parametrize('n_dim', (1, 2, 3))
@pytest.mark.parametrize('offset', (0, 20))
def test_proxy_slicing(n_dim, offset):
    shape = (15, 16, 17)[:n_dim]
    arr = np.arange(np.prod(shape)).reshape(shape)
    hdr = Nifti1Header()
    hdr.set_data_offset(offset)
    hdr.set_data_dtype(arr.dtype)
    hdr.set_data_shape(shape)
    for order, klass in ('F', ArrayProxy), ('C', CArrayProxy):
        fobj = BytesIO()
        fobj.write(b'\0' * offset)
        fobj.write(arr.tobytes(order=order))
        prox = klass(fobj, hdr)
        assert prox.order == order
        for sliceobj in slicer_samples(shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])


def test_proxy_slicing_with_scaling():
    shape = (15, 16, 17)
    offset = 20
    arr = np.arange(np.prod(shape)).reshape(shape)
    hdr = Nifti1Header()
    hdr.set_data_offset(offset)
    hdr.set_data_dtype(arr.dtype)
    hdr.set_data_shape(shape)
    hdr.set_slope_inter(2.0, 1.0)
    fobj = BytesIO()
    fobj.write(bytes(offset))
    fobj.write(arr.tobytes(order='F'))
    prox = ArrayProxy(fobj, hdr)
    sliceobj = (None, slice(None), 1, -1)
    assert_array_equal(arr[sliceobj] * 2.0 + 1.0, prox[sliceobj])


@pytest.mark.parametrize('order', ('C', 'F'))
def test_order_override(order):
    shape = (15, 16, 17)
    arr = np.arange(np.prod(shape)).reshape(shape)
    fobj = BytesIO()
    fobj.write(arr.tobytes(order=order))
    for klass in (ArrayProxy, CArrayProxy):
        prox = klass(fobj, (shape, arr.dtype), order=order)
        assert prox.order == order
        sliceobj = (None, slice(None), 1, -1)
        assert_array_equal(arr[sliceobj], prox[sliceobj])


def test_deprecated_order_classvar():
    shape = (15, 16, 17)
    arr = np.arange(np.prod(shape)).reshape(shape)
    fobj = BytesIO()
    fobj.write(arr.tobytes(order='C'))
    sliceobj = (None, slice(None), 1, -1)

    # We don't really care about the original order, just that the behavior
    # of the deprecated mode matches the new behavior
    fprox = ArrayProxy(fobj, (shape, arr.dtype), order='F')
    cprox = ArrayProxy(fobj, (shape, arr.dtype), order='C')

    # Start raising errors when we crank the dev version
    if Version(__version__) >= Version('7.0.0.dev0'):
        cm = pytest.raises(ExpiredDeprecationError)
    else:
        cm = pytest.deprecated_call()

    with cm:
        prox = DeprecatedCArrayProxy(fobj, (shape, arr.dtype))
        assert prox.order == 'C'
        assert_array_equal(prox[sliceobj], cprox[sliceobj])
    with cm:
        prox = DeprecatedCArrayProxy(fobj, (shape, arr.dtype), order='C')
        assert prox.order == 'C'
        assert_array_equal(prox[sliceobj], cprox[sliceobj])
    with cm:
        prox = DeprecatedCArrayProxy(fobj, (shape, arr.dtype), order='F')
        assert prox.order == 'F'
        assert_array_equal(prox[sliceobj], fprox[sliceobj])


def test_is_proxy():
    # Test is_proxy function
    hdr = FunkyHeader((2, 3, 4))
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    assert is_proxy(prox)
    assert not is_proxy(bio)
    assert not is_proxy(hdr)
    assert not is_proxy(np.zeros((2, 3, 4)))

    class NP:
        is_proxy = False

    assert not is_proxy(NP())


def test_reshape_dataobj():
    # Test function that reshapes using method if possible
    shape = (1, 2, 3, 4)
    hdr = FunkyHeader(shape)
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    arr = np.arange(np.prod(shape), dtype=prox.dtype).reshape(shape)
    bio.write(b'\x00' * prox.offset + arr.tobytes(order='F'))
    assert_array_equal(prox, arr)
    assert_array_equal(reshape_dataobj(prox, (2, 3, 4)), np.reshape(arr, (2, 3, 4)))
    assert prox.shape == shape
    assert arr.shape == shape
    assert_array_equal(reshape_dataobj(arr, (2, 3, 4)), np.reshape(arr, (2, 3, 4)))
    assert arr.shape == shape

    class ArrGiver:
        def __array__(self):
            return arr

    assert_array_equal(reshape_dataobj(ArrGiver(), (2, 3, 4)), np.reshape(arr, (2, 3, 4)))
    assert arr.shape == shape


def test_reshaped_is_proxy():
    shape = (1, 2, 3, 4)
    hdr = FunkyHeader(shape)
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    assert isinstance(prox.reshape((2, 3, 4)), ArrayProxy)
    minus1 = prox.reshape((2, -1, 4))
    assert isinstance(minus1, ArrayProxy)
    assert minus1.shape == (2, 3, 4)
    with pytest.raises(ValueError):
        prox.reshape((-1, -1, 4))
    with pytest.raises(ValueError):
        prox.reshape((2, 3, 5))
    with pytest.raises(ValueError):
        prox.reshape((2, -1, 5))


def test_get_obj_dtype():
    # Check get_obj_dtype(obj) returns same result as array(obj).dtype
    bio = BytesIO()
    shape = (2, 3, 4)
    hdr = Nifti1Header()
    arr = np.arange(24, dtype=np.int16).reshape(shape)
    write_raw_data(arr, hdr, bio)
    hdr.set_slope_inter(2, 10)
    prox = ArrayProxy(bio, hdr)
    assert get_obj_dtype(prox) == np.dtype('float64')
    assert get_obj_dtype(np.array(prox)) == np.dtype('float64')
    hdr.set_slope_inter(1, 0)
    prox = ArrayProxy(bio, hdr)
    assert get_obj_dtype(prox) == np.dtype('int16')
    assert get_obj_dtype(np.array(prox)) == np.dtype('int16')

    class ArrGiver:
        def __array__(self):
            return arr

    assert get_obj_dtype(ArrGiver()) == np.dtype('int16')


def test_get_unscaled():
    # Test fetch of raw array
    class FunkyHeader2(FunkyHeader):
        def get_slope_inter(self):
            return 2.1, 3.14

    shape = (2, 3, 4)
    hdr = FunkyHeader2(shape)
    bio = BytesIO()
    # Check standard read works
    arr = np.arange(24, dtype=np.int32).reshape(shape, order='F')
    bio.write(b'\x00' * hdr.get_data_offset())
    bio.write(arr.tobytes(order='F'))
    prox = ArrayProxy(bio, hdr)
    assert_array_almost_equal(np.array(prox), arr * 2.1 + 3.14)
    # Check unscaled read works
    assert_array_almost_equal(prox.get_unscaled(), arr)


def test_mmap():
    # Unscaled should return mmap from suitable file, this can be tuned
    hdr = FunkyHeader((2, 3, 4))
    check_mmap(hdr, hdr.get_data_offset(), ArrayProxy)


def check_mmap(hdr, offset, proxy_class, has_scaling=False, unscaled_is_view=True):
    """Assert that array proxies return memory maps as expected

    Parameters
    ----------
    hdr : object
        Image header instance
    offset : int
        Offset in bytes of image data in file (that we will write)
    proxy_class : class
        Class of image array proxy to test
    has_scaling : {False, True}
        True if the `hdr` says to apply scaling to the output data, False
        otherwise.
    unscaled_is_view : {True, False}
        True if getting the unscaled data returns a view of the array.  If
        False, then type of returned array will depend on whether numpy has the
        old viral (< 1.12) memmap behavior (returns memmap) or the new behavior
        (returns ndarray).  See: https://github.com/numpy/numpy/pull/7406
    """
    shape = hdr.get_data_shape()
    arr = np.arange(np.prod(shape), dtype=hdr.get_data_dtype()).reshape(shape)
    fname = 'test.bin'
    # Whether unscaled array memory backed by memory map (regardless of what
    # numpy says).
    unscaled_really_mmap = unscaled_is_view
    # Whether scaled array memory backed by memory map (regardless of what
    # numpy says).
    scaled_really_mmap = unscaled_really_mmap and not has_scaling
    # Whether ufunc on memmap return memmap
    viral_memmap = memmap_after_ufunc()
    with InTemporaryDirectory():
        with open(fname, 'wb') as fobj:
            fobj.write(b' ' * offset)
            fobj.write(arr.tobytes(order='F'))
        for mmap, expected_mode in (
            # mmap value, expected memmap mode
            # mmap=None -> no mmap value
            # expected mode=None -> no memmap returned
            (None, 'c'),
            (True, 'c'),
            ('c', 'c'),
            ('r', 'r'),
            (False, None),
        ):
            kwargs = {}
            if mmap is not None:
                kwargs['mmap'] = mmap
            prox = proxy_class(fname, hdr, **kwargs)
            unscaled = prox.get_unscaled()
            back_data = np.asanyarray(prox)
            unscaled_is_mmap = isinstance(unscaled, np.memmap)
            back_is_mmap = isinstance(back_data, np.memmap)
            if expected_mode is None:
                assert not unscaled_is_mmap
                assert not back_is_mmap
            else:
                assert unscaled_is_mmap == (viral_memmap or unscaled_really_mmap)
                assert back_is_mmap == (viral_memmap or scaled_really_mmap)
                if scaled_really_mmap:
                    assert back_data.mode == expected_mode
            del prox, back_data
            # Check that mmap is keyword-only
            with pytest.raises(TypeError):
                proxy_class(fname, hdr, True)
            # Check invalid values raise error
            with pytest.raises(ValueError):
                proxy_class(fname, hdr, mmap='rw')
            with pytest.raises(ValueError):
                proxy_class(fname, hdr, mmap='r+')


# An image opener class which counts how many instances of itself have been
# created
class CountingImageOpener(ImageOpener):
    num_openers = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountingImageOpener.num_openers += 1


def _count_ImageOpeners(proxy, data, voxels):
    CountingImageOpener.num_openers = 0
    # expected data is defined in the test_keep_file_open_* tests
    for i in range(voxels.shape[0]):
        x, y, z = (int(c) for c in voxels[i, :])
        assert proxy[x, y, z] == x * 100 + y * 10 + z
    return CountingImageOpener.num_openers


@contextlib.contextmanager
def patch_keep_file_open_default(value):
    # Patch arrayproxy.KEEP_FILE_OPEN_DEFAULT with the given value
    with mock.patch('nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT', value):
        yield


def test_keep_file_open_true_false_invalid():
    # Test the behaviour of the keep_file_open __init__ flag, when it is set to
    # True or False. Expected behaviour is as follows:
    # keep_open | igzip present    | persist ImageOpener | igzip.drop_handles
    #           | and is gzip file |                     |
    # ----------|------------------|---------------------|-------------------
    # False     | False            | False               | n/a
    # False     | True             | True                | True
    # True      | False            | True                | n/a
    # True      | True             | True                | False
    #
    # Each test tuple contains:
    #  - file type - gzipped ('gz') or not ('bin'), or an open file handle
    #    ('open')
    #  - keep_file_open value passed to ArrayProxy
    #  - whether or not indexed_gzip is present
    #  - expected value for internal ArrayProxy._persist_opener flag
    #  - expected value for internal ArrayProxy._keep_file_open flag
    tests = [
        # open file handle - kfo and have_igzip are both irrelevant
        ('open', False, False, False, False),
        ('open', False, True, False, False),
        ('open', True, False, False, False),
        ('open', True, True, False, False),
        # non-gzip file - have_igzip is irrelevant, decision should be made
        # solely from kfo flag
        ('bin', False, False, False, False),
        ('bin', False, True, False, False),
        ('bin', True, False, True, True),
        ('bin', True, True, True, True),
        # gzip file. If igzip is present, we persist the ImageOpener.
        ('gz', False, False, False, False),
        ('gz', False, True, True, False),
        ('gz', True, False, True, True),
        ('gz', True, True, True, True),
    ]

    dtype = np.float32
    data = np.arange(1000, dtype=dtype).reshape((10, 10, 10))
    voxels = np.random.randint(0, 10, (10, 3))

    for test in tests:
        filetype, kfo, have_igzip, exp_persist, exp_kfo = test
        with (
            InTemporaryDirectory(),
            mock.patch('nibabel.openers.ImageOpener', CountingImageOpener),
            patch_indexed_gzip(have_igzip),
        ):
            fname = f'testdata.{filetype}'
            # create the test data file
            if filetype == 'gz':
                with gzip.open(fname, 'wb') as fobj:
                    fobj.write(data.tobytes(order='F'))
            else:
                with open(fname, 'wb') as fobj:
                    fobj.write(data.tobytes(order='F'))
            # pass in a file name or open file handle. If the latter, we open
            # two file handles, because we're going to create two proxies
            # below.
            if filetype == 'open':
                fobj1 = open(fname, 'rb')
                fobj2 = open(fname, 'rb')
            else:
                fobj1 = fname
                fobj2 = fname
            try:
                proxy = ArrayProxy(fobj1, ((10, 10, 10), dtype), keep_file_open=kfo)
                # We also test that we get the same behaviour when the
                # KEEP_FILE_OPEN_DEFAULT flag is changed
                with patch_keep_file_open_default(kfo):
                    proxy_def = ArrayProxy(fobj2, ((10, 10, 10), dtype))
                # check internal flags
                assert proxy._persist_opener == exp_persist
                assert proxy._keep_file_open == exp_kfo
                assert proxy_def._persist_opener == exp_persist
                assert proxy_def._keep_file_open == exp_kfo
                # check persist_opener behaviour - whether one imageopener is
                # created for the lifetime of the ArrayProxy, or one is
                # created on each access
                if exp_persist:
                    assert _count_ImageOpeners(proxy, data, voxels) == 1
                    assert _count_ImageOpeners(proxy_def, data, voxels) == 1
                else:
                    assert _count_ImageOpeners(proxy, data, voxels) == 10
                    assert _count_ImageOpeners(proxy_def, data, voxels) == 10
                # if indexed_gzip is active, check that the file object was
                # created correctly - the _opener.fobj will be a
                # MockIndexedGzipFile, defined in test_openers.py
                if filetype == 'gz' and have_igzip:
                    assert proxy._opener.fobj._drop_handles == (not exp_kfo)
                # if we were using an open file handle, check that the proxy
                # didn't close it
                if filetype == 'open':
                    assert not fobj1.closed
                    assert not fobj2.closed
            finally:
                del proxy
                del proxy_def
                if filetype == 'open':
                    fobj1.close()
                    fobj2.close()
    # Test invalid values of keep_file_open
    with InTemporaryDirectory():
        fname = 'testdata'
        with open(fname, 'wb') as fobj:
            fobj.write(data.tobytes(order='F'))

        for invalid_kfo in (55, 'auto', 'cauto'):
            with pytest.raises(ValueError):
                ArrayProxy(fname, ((10, 10, 10), dtype), keep_file_open=invalid_kfo)
            with patch_keep_file_open_default(invalid_kfo):
                with pytest.raises(ValueError):
                    ArrayProxy(fname, ((10, 10, 10), dtype))


def islock(l):
    # isinstance doesn't work on threading.Lock?
    return hasattr(l, 'acquire') and hasattr(l, 'release')


def test_pickle_lock():
    # Test that ArrayProxy can be pickled, and that thread lock is created

    proxy = ArrayProxy('dummyfile', ((10, 10, 10), np.float32))
    assert islock(proxy._lock)
    pickled = pickle.dumps(proxy)
    unpickled = pickle.loads(pickled)
    assert islock(unpickled._lock)
    assert proxy._lock is not unpickled._lock


def test_copy():
    # Test copying array proxies

    # If the file-like is a file name, get a new lock
    proxy = ArrayProxy('dummyfile', ((10, 10, 10), np.float32))
    assert islock(proxy._lock)
    copied = proxy.copy()
    assert islock(copied._lock)
    assert proxy._lock is not copied._lock

    # If an open filehandle, the lock should be shared to
    # avoid changing filehandle state in critical sections
    proxy = ArrayProxy(BytesIO(), ((10, 10, 10), np.float32))
    assert islock(proxy._lock)
    copied = proxy.copy()
    assert islock(copied._lock)
    assert proxy._lock is copied._lock


def test_copy_with_indexed_gzip_handle(tmp_path):
    indexed_gzip = pytest.importorskip('indexed_gzip')

    spec = ((50, 50, 50, 50), np.float32, 352, 1, 0)
    data = np.arange(np.prod(spec[0]), dtype=spec[1]).reshape(spec[0])
    fname = str(tmp_path / 'test.nii.gz')
    Nifti1Image(data, np.eye(4)).to_filename(fname)

    with indexed_gzip.IndexedGzipFile(fname) as fobj:
        proxy = ArrayProxy(fobj, spec)
        copied = proxy.copy()

        assert proxy.file_like is copied.file_like
        assert np.array_equal(proxy[0, 0, 0], copied[0, 0, 0])
        assert np.array_equal(proxy[-1, -1, -1], copied[-1, -1, -1])
