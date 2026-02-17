# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from os.path import join as pjoin

import numpy as np
import pytest

from .. import minc2
from ..minc2 import Minc2File, Minc2Image
from ..optpkg import optional_package
from ..testing import data_path
from . import test_minc1 as tm2

h5py, have_h5py, setup_module = optional_package('h5py')

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'small.mnc'),
        shape=(18, 28, 29),
        dtype=np.int16,
        affine=np.array(
            [
                [0, 0, 7.0, -98],
                [0, 8.0, 0, -134],
                [9.0, 0, 0, -72],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(9.0, 8.0, 7.0),
        # These values from mincstats
        data_summary=dict(min=0.1185331417, max=92.87690699, mean=31.2127952),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2_1_scale.mnc'),
        shape=(10, 20, 20),
        dtype=np.uint8,
        affine=np.array(
            [
                [0, 0, 2.0, -20],
                [0, 2.0, 0, -20],
                [2.0, 0, 0, -10],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(2.0, 2.0, 2.0),
        # These values from mincstats
        data_summary=dict(min=0.2082842439, max=0.2094327615, mean=0.2091292083),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2_4d.mnc'),
        shape=(2, 10, 20, 20),
        dtype=np.uint8,
        affine=np.array(
            [
                [0, 0, 2.0, -20],
                [0, 2.0, 0, -20],
                [2.0, 0, 0, -10],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(1.0, 2.0, 2.0, 2.0),
        # These values from mincstats
        data_summary=dict(min=0.2078431373, max=1.498039216, mean=0.9090422837),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2-no-att.mnc'),
        shape=(10, 20, 20),
        dtype=np.uint8,
        affine=np.array(
            [
                [0, 0, 1.0, 0],
                [0, 1.0, 0, 0],
                [1.0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(1.0, 1.0, 1.0),
        # These values from SPM2/mincstats
        data_summary=dict(min=0.20784314, max=0.74901961, mean=0.6061103),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2-4d-d.mnc'),
        shape=(5, 16, 16, 16),
        dtype=np.float64,
        affine=np.array(
            [
                [1.0, 0.0, 0.0, -6.96],
                [0.0, 1.0, 0.0, -12.453],
                [0.0, 0.0, 1.0, -9.48],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        zooms=(1.0, 1.0, 1.0, 1.0),
        # These values from mincstats
        data_summary=dict(min=0.0, max=5.0, mean=2.00078125),
        is_proxy=True,
    ),
]

if have_h5py:

    class TestMinc2File(tm2._TestMincFile):
        module = minc2
        file_class = Minc2File
        opener = h5py.File
        test_files = EXAMPLE_IMAGES

    class TestMinc2Image(tm2.TestMinc1Image):
        image_class = Minc2Image
        eg_images = (pjoin(data_path, 'small.mnc'),)
        module = minc2


def test_bad_diminfo():
    fname = pjoin(data_path, 'minc2_baddim.mnc')
    # File has a bad spacing field 'xspace' when it should be
    # `irregular`, `regular__` or absent (default to regular__).
    # We interpret an invalid spacing as absent, but warn.
    with pytest.warns(UserWarning):
        Minc2Image.from_filename(fname)
