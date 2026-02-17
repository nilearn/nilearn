"""Benchmarks for array_to_file routine

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_array_to_file.py
"""

import sys
from io import BytesIO  # noqa: F401

import numpy as np
from numpy.testing import measure

from nibabel.volumeutils import array_to_file  # noqa: F401

from .butils import print_git_title


def bench_array_to_file():
    rng = np.random.RandomState(20111001)
    repeat = 10
    img_shape = (128, 128, 64, 10)
    arr = rng.normal(size=img_shape)
    sys.stdout.flush()
    print_git_title('\nArray to file')
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16', mtime))
    # Set a lot of NaNs to check timing
    arr[:, :, :, 1] = np.nan
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32, NaNs', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, NaNs', mtime))
    # Set a lot of infs to check timing
    arr[:, :, :, 1] = np.inf
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32, infs', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, infs', mtime))
    # Int16 input, float output
    arr = np.random.random_integers(low=-1000, high=1000, size=img_shape)
    arr = arr.astype(np.int16)
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save Int16 to float32', mtime))
    sys.stdout.flush()
