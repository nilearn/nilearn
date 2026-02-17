"""Benchmarks for finite_range routine

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_finite_range.py
"""

import sys

import numpy as np
from numpy.testing import measure

from nibabel.volumeutils import finite_range  # noqa: F401

from .butils import print_git_title


def bench_finite_range():
    rng = np.random.RandomState(20111001)
    repeat = 10
    img_shape = (128, 128, 64, 10)
    arr = rng.normal(size=img_shape)
    sys.stdout.flush()
    print_git_title('\nFinite range')
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('float64 all finite', mtime))
    arr[:, :, :, 1] = np.nan
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('float64 many NaNs', mtime))
    arr[:, :, :, 1] = np.inf
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('float64 many infs', mtime))
    # Int16 input, float output
    arr = np.random.random_integers(low=-1000, high=1000, size=img_shape)
    arr = arr.astype(np.int16)
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('int16', mtime))
    sys.stdout.flush()
