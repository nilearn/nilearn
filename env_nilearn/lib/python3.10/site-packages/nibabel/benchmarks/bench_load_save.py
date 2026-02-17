"""Benchmarks for load and save of image arrays

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_load_save.py
"""

import sys
from io import BytesIO

import numpy as np
from numpy.testing import measure

from .. import Nifti1Image
from .butils import print_git_title


def bench_load_save():
    rng = np.random.RandomState(20111001)
    repeat = 10
    img_shape = (128, 128, 64, 10)
    arr = rng.normal(size=img_shape)
    img = Nifti1Image(arr, np.eye(4))
    sio = BytesIO()
    img.file_map['image'].fileobj = sio
    hdr = img.header
    sys.stdout.flush()
    print()
    print_git_title('Image load save')
    hdr.set_data_dtype(np.float32)
    mtime = measure('sio.truncate(0); img.to_file_map()', repeat)
    print('%30s %6.2f' % ('Save float64 to float32', mtime))
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print('%30s %6.2f' % ('Load from float32', mtime))
    hdr.set_data_dtype(np.int16)
    mtime = measure('sio.truncate(0); img.to_file_map()', repeat)
    print('%30s %6.2f' % ('Save float64 to int16', mtime))
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print('%30s %6.2f' % ('Load from int16', mtime))
    # Set a lot of NaNs to check timing
    arr[:, :, :20] = np.nan
    mtime = measure('sio.truncate(0); img.to_file_map()', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, NaNs', mtime))
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print('%30s %6.2f' % ('Load from int16, NaNs', mtime))
    # Int16 input, float output
    arr = np.random.random_integers(low=-1000, high=1000, size=img_shape)
    arr = arr.astype(np.int16)
    img = Nifti1Image(arr, np.eye(4))
    sio = BytesIO()
    img.file_map['image'].fileobj = sio
    hdr = img.header
    hdr.set_data_dtype(np.float32)
    mtime = measure('sio.truncate(0); img.to_file_map()', repeat)
    print('%30s %6.2f' % ('Save Int16 to float32', mtime))
    sys.stdout.flush()
