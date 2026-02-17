"""Tests for the parrec2nii exe code"""

from os.path import basename, isfile, join
from unittest.mock import MagicMock, Mock, patch

import numpy
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal

import nibabel
from nibabel.cmdline import parrec2nii
from nibabel.tests.test_parrec import EG_PAR, VARY_PAR
from nibabel.tmpdirs import InTemporaryDirectory

AN_OLD_AFFINE = numpy.array(
    [
        [-3.64994708, 0.0, 1.83564171, 123.66276611],
        [0.0, -3.75, 0.0, 115.617],
        [0.86045705, 0.0, 7.78655376, -27.91161211],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

PAR_AFFINE = numpy.array(
    [
        [-3.64994708, 0.0, 1.83564171, 107.63076611],
        [0.0, 3.75, 0.0, -118.125],
        [0.86045705, 0.0, 7.78655376, -58.25061211],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@patch('nibabel.cmdline.parrec2nii.verbose')
@patch('nibabel.cmdline.parrec2nii.io_orientation')
@patch('nibabel.cmdline.parrec2nii.nifti1')
@patch('nibabel.cmdline.parrec2nii.pr')
def test_parrec2nii_sets_qform_sform_code1(*args):
    # Check that set_sform(), set_qform() are called on the new header.
    parrec2nii.verbose.switch = False

    parrec2nii.io_orientation.return_value = [[0, 1], [1, 1], [2, 1]]  # LAS+

    nimg = Mock()
    nhdr = MagicMock()
    nimg.header = nhdr
    parrec2nii.nifti1.Nifti1Image.return_value = nimg

    pr_img = Mock()
    pr_hdr = Mock()
    pr_hdr.get_data_scaling.return_value = (npa([]), npa([]))
    pr_hdr.get_bvals_bvecs.return_value = (None, None)
    pr_hdr.get_affine.return_value = AN_OLD_AFFINE
    pr_img.header = pr_hdr
    parrec2nii.pr.load.return_value = pr_img

    opts = Mock()
    opts.outdir = None
    opts.scaling = 'off'
    opts.minmax = [1, 1]
    opts.store_header = False
    opts.bvs = False
    opts.vol_info = False
    opts.dwell_time = False

    infile = 'nonexistent.PAR'
    parrec2nii.proc_file(infile, opts)
    nhdr.set_qform.assert_called_with(AN_OLD_AFFINE, code=1)
    nhdr.set_sform.assert_called_with(AN_OLD_AFFINE, code=1)


@patch('nibabel.cmdline.parrec2nii.verbose')
def test_parrec2nii_save_load_qform_code(*args):
    # Tests that after parrec2nii saves file, it has the sform and qform 'code'
    # set to '1', which means 'scanner', so that other software, e.g. FSL picks
    # up the qform.
    parrec2nii.verbose.switch = False

    opts = Mock()
    opts.outdir = None
    opts.scaling = 'off'
    opts.minmax = [1, 1]
    opts.store_header = False
    opts.bvs = False
    opts.vol_info = False
    opts.dwell_time = False
    opts.compressed = False

    with InTemporaryDirectory() as pth:
        opts.outdir = pth
        for fname in [EG_PAR, VARY_PAR]:
            parrec2nii.proc_file(fname, opts)
            outfname = join(pth, basename(fname)).replace('.PAR', '.nii')
            assert isfile(outfname)
            img = nibabel.load(outfname)
            assert_almost_equal(img.affine, PAR_AFFINE, 4)
            assert img.header['qform_code'] == 1
            assert_array_equal(img.header['sform_code'], 1)
