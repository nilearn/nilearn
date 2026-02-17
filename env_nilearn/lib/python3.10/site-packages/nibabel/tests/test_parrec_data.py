"""Test we can correctly import example PARREC files"""

import unittest
from glob import glob
from os.path import basename, exists, splitext
from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_almost_equal

from .. import load as top_load
from ..affines import voxel_sizes
from ..parrec import load
from .nibabel_data import get_nibabel_data, needs_nibabel_data

BALLS = pjoin(get_nibabel_data(), 'nitest-balls1')
OBLIQUE = pjoin(get_nibabel_data(), 'parrec_oblique')

# Amount by which affine translation differs from NIFTI conversion
AFF_OFF = [-0.93575081, -0.95657335, 0.03264122]


@needs_nibabel_data('nitest-balls1')
def test_loading():
    # Test loading of parrec files
    for par in glob(pjoin(BALLS, 'PARREC', '*.PAR')):
        par_root, ext = splitext(basename(par))
        # NA.PAR appears to be a localizer, with three slices in each of the
        # three orientations: sagittal; coronal, transverse
        if par_root == 'NA':
            continue
        # Check we can load the image
        pimg = load(par)
        assert pimg.shape[:3] == (80, 80, 10)
        # Compare against NIFTI if present
        nifti_fname = pjoin(BALLS, 'NIFTI', par_root + '.nii.gz')
        if exists(nifti_fname):
            nimg = top_load(nifti_fname)
            assert_almost_equal(nimg.affine[:3, :3], pimg.affine[:3, :3], 3)
            # The translation part is always off by the same ammout
            aff_off = pimg.affine[:3, 3] - nimg.affine[:3, 3]
            assert_almost_equal(aff_off, AFF_OFF, 4)
            # The difference is max in the order of 0.5 voxel
            vox_sizes = voxel_sizes(nimg.affine)
            assert np.all(np.abs(aff_off / vox_sizes) <= 0.501)
            # The data is very close, unless it's the fieldmap
            if par_root != 'fieldmap':
                assert np.allclose(pimg.dataobj, nimg.dataobj)
            # Not sure what's going on with the fieldmap image - TBA


@needs_nibabel_data('nitest-balls1')
def test_fieldmap():
    # Test fieldmap image
    # Exploring the DICOM suggests that the first volume is magnitude and the
    # second is phase.  The NIfTI has very odd scaling, being all negative.
    fieldmap_par = pjoin(BALLS, 'PARREC', 'fieldmap.PAR')
    fieldmap_nii = pjoin(BALLS, 'NIFTI', 'fieldmap.nii.gz')
    load(fieldmap_par)
    top_load(fieldmap_nii)
    raise unittest.SkipTest('Fieldmap remains puzzling')


@needs_nibabel_data('parrec_oblique')
def test_oblique_loading():
    # Test loading of oblique parrec files
    for par in glob(pjoin(OBLIQUE, 'PARREC', '*.PAR')):
        par_root, ext = splitext(basename(par))
        # Check we can load the image
        pimg = load(par)
        assert pimg.shape == (560, 560, 1)
        # Compare against NIFTI if present
        nifti_fname = pjoin(OBLIQUE, 'NIFTI', par_root + '.nii')
        nimg = top_load(nifti_fname)
        assert_almost_equal(nimg.affine[:3, :3], pimg.affine[:3, :3], 3)
        # The translation part is always off
        # The amount differs by rotation
        aff_off = pimg.affine[:3, 3] - nimg.affine[:3, 3]
        # The difference is max in the order of 0.5 voxel
        vox_sizes = voxel_sizes(nimg.affine)
        assert np.all(np.abs(aff_off / vox_sizes) <= 0.5)
        # The data is very close
        assert np.allclose(pimg.dataobj, nimg.dataobj)
