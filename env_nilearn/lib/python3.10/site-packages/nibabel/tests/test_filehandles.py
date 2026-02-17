"""
Check that loading an image does not use up filehandles.
"""

import shutil
import unittest
from os.path import join as pjoin
from tempfile import mkdtemp

import numpy as np

try:
    import resource as res
except ImportError:
    # Not on Unix, guess limit
    SOFT_LIMIT = 512
else:
    SOFT_LIMIT, HARD_LIMIT = res.getrlimit(res.RLIMIT_NOFILE)

from ..loadsave import load, save
from ..nifti1 import Nifti1Image


@unittest.skipIf(SOFT_LIMIT > 4900, 'It would take too long to test filehandles')
def test_multiload():
    # Make a tiny image, save, load many times.  If we are leaking filehandles,
    # this will cause us to run out and generate an error
    N = SOFT_LIMIT + 100
    arr = np.arange(24, dtype='int32').reshape((2, 3, 4))
    img = Nifti1Image(arr, np.eye(4))
    imgs = []
    try:
        tmpdir = mkdtemp()
        fname = pjoin(tmpdir, 'test.img')
        save(img, fname)
        imgs.extend(load(fname) for _ in range(N))
    finally:
        del img, imgs
        shutil.rmtree(tmpdir)
