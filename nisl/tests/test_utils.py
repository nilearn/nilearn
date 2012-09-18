"""
Test the utils module
"""

import nose

import numpy as np

from nibabel import Nifti1Image

from .. import utils

def test_check_niimg():
    nose.tools.assert_raises(TypeError, utils.check_niimg, 0)


def test_check_niimgs():
    nose.tools.assert_raises(TypeError, utils.check_niimgs, 0)
    affine = np.eye(4)
    niimg = Nifti1Image(np.ones((10, 10, 10)), affine)

    utils.check_niimgs([niimg, niimg])
    # This should raise an error: a 3D niimg is given and we want a 4D
    nose.tools.assert_raises(TypeError, utils.check_niimgs,
                             niimg)
    # This shouldn't raise an error
    utils.check_niimgs(niimg, accept_3d=True)


def test_concat_niimgs():
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones((10, 10, 10)), affine)
    niimg2 = Nifti1Image(np.ones((10, 10, 10)), 2*affine)

    nose.tools.assert_raises(ValueError, utils.concat_niimgs,
                            [niimg1, niimg2])

