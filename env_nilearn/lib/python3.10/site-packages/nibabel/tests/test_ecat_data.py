# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test we can correctly import example ECAT files"""

import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_almost_equal

from ..ecat import load
from .nibabel_data import get_nibabel_data, needs_nibabel_data

ECAT_TEST_PATH = pjoin(get_nibabel_data(), 'nipy-ecattest')


class TestNegatives:
    opener = staticmethod(load)
    example_params = dict(
        fname=os.path.join(ECAT_TEST_PATH, 'ECAT7_testcaste_neg_values.v'),
        shape=(256, 256, 63, 1),
        type=np.int16,
        # These values from freec64
        min=-0.00061576,
        max=0.19215,
        mean=0.04933,
        # unit: 1/cm
    )

    @needs_nibabel_data('nipy-ecattest')
    def test_load(self):
        # Check highest level load of minc works
        img = self.opener(self.example_params['fname'])
        assert img.shape == self.example_params['shape']
        assert img.get_data_dtype(0).type == self.example_params['type']
        # Check correspondence of data and recorded shape
        data = img.get_fdata()
        assert data.shape == self.example_params['shape']
        # min, max, mean values from given parameters
        assert_almost_equal(data.min(), self.example_params['min'], 4)
        assert_almost_equal(data.max(), self.example_params['max'], 4)
        assert_almost_equal(data.mean(), self.example_params['mean'], 4)


class TestMultiframe(TestNegatives):
    example_params = dict(
        fname=os.path.join(ECAT_TEST_PATH, 'ECAT7_testcase_multiframe.v'),
        shape=(256, 256, 207, 3),
        type=np.int16,
        # Zeroed out image
        min=0.0,
        max=29170.67905,
        mean=121.454,
    )
