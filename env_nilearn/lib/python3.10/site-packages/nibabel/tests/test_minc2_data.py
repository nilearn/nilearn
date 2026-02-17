# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test we can correctly import example MINC2_PATH files"""

import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data

h5py, have_h5py, setup_module = optional_package('h5py')

MINC2_PATH = pjoin(get_nibabel_data(), 'nitest-minc2')


def _make_affine(coses, zooms, starts):
    R = np.column_stack(coses)
    Z = np.diag(zooms)
    affine = np.eye(4)
    affine[:3, :3] = np.dot(R, Z)
    affine[:3, 3] = np.dot(R, starts)
    return affine


class TestEPIFrame:
    opener = staticmethod(top_load)
    x_cos = [1, 0, 0]
    y_cos = [0.0, 1, 0]
    z_cos = [0, 0, 1]
    zooms = [-0.8984375, -0.8984375, 3.0]
    starts = [117.25609125, 138.89861125, -54.442028]
    example_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_EPI-frame.mnc'),
        shape=(40, 256, 256),
        type=np.int16,
        affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]),
        zooms=[abs(v) for v in zooms[::-1]],
        # These values from mincstats
        min=0.0,
        max=1273,
        mean=93.52085367,
    )

    @needs_nibabel_data('nitest-minc2')
    def test_load(self):
        # Check highest level load of minc works
        img = self.opener(self.example_params['fname'])
        assert img.shape == self.example_params['shape']
        assert_almost_equal(img.header.get_zooms(), self.example_params['zooms'], 5)
        assert_almost_equal(img.affine, self.example_params['affine'], 4)
        assert img.get_data_dtype().type == self.example_params['type']
        # Check correspondence of data and recorded shape
        data = img.get_fdata()
        assert data.shape == self.example_params['shape']
        # min, max, mean values from read in SPM2
        assert_almost_equal(data.min(), self.example_params['min'], 4)
        assert_almost_equal(data.max(), self.example_params['max'], 4)
        assert_almost_equal(data.mean(), self.example_params['mean'], 4)
        # check if mnc can be converted to nifti
        ni_img = Nifti1Image.from_image(img)
        assert_almost_equal(ni_img.affine, self.example_params['affine'], 2)
        assert_array_equal(ni_img.get_fdata(), data)


class TestB0(TestEPIFrame):
    x_cos = [0.9970527523765, 0.0, 0.0767190261828617]
    y_cos = [0.0, 1.0, -6.9388939e-18]
    z_cos = [-0.0767190261828617, 6.9184432614435e-18, 0.9970527523765]
    zooms = [-0.8984375, -0.8984375, 6.49999990444107]
    starts = [105.473101260826, 151.74885125, -61.8714747993248]
    example_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_diff-B0.mnc'),
        shape=(19, 256, 256),
        type=np.int16,
        affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]),
        zooms=[abs(v) for v in zooms[::-1]],
        # These values from mincstats
        min=4.566971917,
        max=3260.121093,
        mean=163.8305553,
    )


class TestFA(TestEPIFrame):
    example_params = TestB0.example_params.copy()
    new_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_diff-FA.mnc'),
        # These values from mincstats
        min=0.008068881038,
        max=1.224754546,
        mean=0.7520087469,
    )
    example_params.update(new_params)


class TestGado(TestEPIFrame):
    x_cos = [0.999695413509548, -0.0174524064372835, 0.0174497483512505]
    y_cos = [0.0174497483512505, 0.999847695156391, 0.000304586490452135]
    z_cos = [-0.0174524064372835, 0.0, 0.999847695156391]
    zooms = [1, -1, -1]
    starts = [-75.76775, 115.80462, 81.38605]
    example_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_gado-contrast.mnc'),
        shape=(100, 170, 146),
        type=np.int16,
        affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]),
        zooms=[abs(v) for v in zooms[::-1]],
        # These values from mincstats
        min=0,
        max=938668.8698,
        mean=128169.3488,
    )


class TestT1(TestEPIFrame):
    x_cos = [1, 0, 0]
    y_cos = [0, 1, 0]
    z_cos = [0, 0, 1]
    zooms = [1, 1, 1]
    starts = [-90, -126, -12]
    example_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_t1.mnc'),
        shape=(110, 217, 181),
        type=np.int16,
        affine=_make_affine((z_cos, y_cos, x_cos), zooms[::-1], starts[::-1]),
        zooms=[abs(v) for v in zooms[::-1]],
        # These values from mincstats
        min=0,
        max=100,
        mean=23.1659928,
    )


class TestPD(TestEPIFrame):
    example_params = TestT1.example_params.copy()
    new_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_pd.mnc'),
        # These values from mincstats
        min=0,
        max=102.5024482,
        mean=23.82625718,
    )
    example_params.update(new_params)


class TestMask(TestEPIFrame):
    example_params = TestT1.example_params.copy()
    new_params = dict(
        fname=os.path.join(MINC2_PATH, 'mincex_mask.mnc'),
        type=np.uint8,
        # These values from mincstats
        min=0,
        max=1,
        mean=0.3817466618,
    )
    example_params.update(new_params)
