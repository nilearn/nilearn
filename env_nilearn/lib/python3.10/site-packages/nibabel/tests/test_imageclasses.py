"""Testing imageclasses module"""

from os.path import dirname
from os.path import join as pjoin

import numpy as np

import nibabel as nib
from nibabel.analyze import AnalyzeImage
from nibabel.imageclasses import spatial_axes_first
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.optpkg import optional_package

have_h5py = optional_package('h5py')[1]

DATA_DIR = pjoin(dirname(__file__), 'data')

MINC_3DS = ('minc1_1_scale.mnc',)
MINC_4DS = ('minc1_4d.mnc',)
if have_h5py:
    MINC_3DS = MINC_3DS + ('minc2_1_scale.mnc',)
    MINC_4DS = MINC_4DS + ('minc2_4d.mnc',)


def test_spatial_axes_first():
    # Function tests is spatial axes are first three axes in image
    # Always True for Nifti and friends
    affine = np.eye(4)
    for shape in ((2, 3), (4, 3, 2), (5, 4, 1, 2), (2, 3, 5, 2, 1)):
        for img_class in (AnalyzeImage, Nifti1Image, Nifti2Image):
            data = np.zeros(shape)
            img = img_class(data, affine)
            assert spatial_axes_first(img)
    # True for MINC images < 4D
    for fname in MINC_3DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        assert len(img.shape) == 3
        assert spatial_axes_first(img)
    # False for MINC images < 4D
    for fname in MINC_4DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        assert len(img.shape) == 4
        assert not spatial_axes_first(img)
