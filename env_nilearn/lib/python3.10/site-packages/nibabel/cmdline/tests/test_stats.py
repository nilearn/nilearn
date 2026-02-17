#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np

from nibabel import Nifti1Image
from nibabel.cmdline.stats import main
from nibabel.loadsave import save


def test_volume(tmpdir, capsys):
    mask_data = np.zeros((20, 20, 20), dtype='u1')
    mask_data[5:15, 5:15, 5:15] = 1
    img = Nifti1Image(mask_data, np.eye(4))

    infile = tmpdir / 'input.nii'
    save(img, infile)

    args = f'{infile} --Volume'
    main(args.split())
    vol_mm3 = capsys.readouterr()
    args = f'{infile} --Volume --units vox'
    main(args.split())
    vol_vox = capsys.readouterr()

    assert float(vol_mm3[0]) == 1000.0
    assert int(vol_vox[0]) == 1000
