#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import unittest

import pytest

import nibabel as nib
from nibabel.cmdline.conform import main
from nibabel.optpkg import optional_package
from nibabel.testing import get_test_data

_, have_scipy, _ = optional_package('scipy.ndimage')
needs_scipy = unittest.skipUnless(have_scipy, 'These tests need scipy')


@needs_scipy
def test_default(tmpdir):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmpdir / 'output.nii.gz'
    main([str(infile), str(outfile)])
    assert outfile.isfile()
    c = nib.load(outfile)
    assert c.shape == (256, 256, 256)
    assert c.header.get_zooms() == (1, 1, 1)
    assert nib.orientations.aff2axcodes(c.affine) == ('R', 'A', 'S')

    with pytest.raises(FileExistsError):
        main([str(infile), str(outfile)])

    main([str(infile), str(outfile), '--force'])
    assert outfile.isfile()


@needs_scipy
def test_nondefault(tmpdir):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmpdir / 'output.nii.gz'
    out_shape = (100, 100, 150)
    voxel_size = (1, 2, 4)
    orientation = 'LAS'
    args = (
        f"{infile} {outfile} --out-shape {' '.join(map(str, out_shape))} "
        f"--voxel-size {' '.join(map(str, voxel_size))} --orientation {orientation}"
    )
    main(args.split())
    assert outfile.isfile()
    c = nib.load(outfile)
    assert c.shape == out_shape
    assert c.header.get_zooms() == voxel_size
    assert nib.orientations.aff2axcodes(c.affine) == tuple(orientation)
