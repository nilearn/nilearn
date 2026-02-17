"""Benchmarks for fileslicing

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_fileslice.py
"""

import sys
from io import BytesIO
from timeit import timeit

import numpy as np

from ..fileslice import fileslice
from ..openers import ImageOpener
from ..optpkg import optional_package
from ..rstutils import rst_table
from ..tmpdirs import InTemporaryDirectory

SHAPE = (64, 64, 32, 100)
ROW_NAMES = [f'axis {i}, len {dim}' for i, dim in enumerate(SHAPE)]
COL_NAMES = ['mid int', 'step 1', 'half step 1', 'step mid int']
HAVE_ZSTD = optional_package('pyzstd')[1]


def _slices_for_len(L):
    # Example slices for a dimension of length L
    return (L // 2, slice(None, None, 1), slice(None, L // 2, 1), slice(None, None, L // 2))


def run_slices(file_like, repeat=3, offset=0, order='F'):
    arr = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    n_dim = len(SHAPE)
    n_slicers = len(_slices_for_len(1))
    times_arr = np.zeros((n_dim, n_slicers))
    with ImageOpener(file_like, 'wb') as fobj:
        fobj.write(b'\0' * offset)
        fobj.write(arr.tobytes(order=order))
    with ImageOpener(file_like, 'rb') as fobj:
        for i, L in enumerate(SHAPE):
            for j, slicer in enumerate(_slices_for_len(L)):
                sliceobj = [slice(None)] * n_dim
                sliceobj[i] = slicer

                def f():
                    fileslice(fobj, tuple(sliceobj), arr.shape, arr.dtype, offset, order)

                times_arr[i, j] = timeit(f, number=repeat)

        def g():
            fobj.seek(offset)
            data = fobj.read()
            np.ndarray(SHAPE, arr.dtype, buffer=data, order=order)

        base_time = timeit(g, number=repeat)
    return times_arr, base_time


def bench_fileslice(bytes=True, file_=True, gz=True, bz2=False, zst=True):
    sys.stdout.flush()
    repeat = 2

    def my_table(title, times, base):
        print()
        print(rst_table(times, ROW_NAMES, COL_NAMES, title, val_fmt='{0[0]:3.2f} ({0[1]:3.2f})'))
        print(f'Base time: {base:3.2f}')

    if bytes:
        fobj = BytesIO()
        times, base = run_slices(fobj, repeat)
        my_table('Bytes slice - raw (ratio)', np.dstack((times, times / base)), base)
    if file_:
        with InTemporaryDirectory():
            file_times, file_base = run_slices('data.bin', repeat)
        my_table(
            'File slice - raw (ratio)', np.dstack((file_times, file_times / file_base)), file_base
        )
    if gz:
        with InTemporaryDirectory():
            gz_times, gz_base = run_slices('data.gz', repeat)
        my_table('gz slice - raw (ratio)', np.dstack((gz_times, gz_times / gz_base)), gz_base)
    if bz2:
        with InTemporaryDirectory():
            bz2_times, bz2_base = run_slices('data.bz2', repeat)
        my_table('bz2 slice - raw (ratio)', np.dstack((bz2_times, bz2_times / bz2_base)), bz2_base)
    if zst and HAVE_ZSTD:
        with InTemporaryDirectory():
            zst_times, zst_base = run_slices('data.zst', repeat)
        my_table('zst slice - raw (ratio)', np.dstack((zst_times, zst_times / zst_base)), zst_base)
    sys.stdout.flush()
