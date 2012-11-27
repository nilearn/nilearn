"""
Test the utils module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD


import os
import tempfile

import nose
from nose.tools import assert_raises, assert_equal

import numpy as np

import nibabel
from nibabel import Nifti1Image

from .. import utils


class PhonyNiimage:

    def __init__(self):
        self.data = np.ones((9, 9, 9, 9))
        self.affine = np.ones((4, 4))

    def get_data(self):
        return self.data

    def get_affine(self):
        return self.affine


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    yield assert_raises, ValueError, utils.largest_connected_component, a
    a[1:3, 1:3, 1:3] = 1
    yield np.testing.assert_equal, a, utils.largest_connected_component(a)
    b = a.copy()
    b[5, 5, 5] = 1
    yield np.testing.assert_equal, a, utils.largest_connected_component(b)


def test_check_niimg():
    assert_raises(TypeError, utils.check_niimg, 0)


def test_check_niimgs():
    yield assert_raises, TypeError, utils.check_niimgs, 0
    affine = np.eye(4)
    niimg = Nifti1Image(np.ones((10, 10, 10)), affine)

    utils.check_niimgs([niimg, niimg])
    # This should raise an error: a 3D niimg is given and we want a 4D
    yield assert_raises, TypeError, utils.check_niimgs, niimg
    # This shouldn't raise an error
    utils.check_niimgs(niimg, accept_3d=True)

    # Test a Niimage that does not hold a shape attribute
    phony_niimg = PhonyNiimage()
    utils.check_niimgs(phony_niimg)


def test_repr_niimgs():
    # Test with file path
    yield assert_equal, utils._repr_niimgs("test"), "test"
    yield assert_equal, utils._repr_niimgs(["test", "retest"]), \
        "[test, retest]"
    # Create phony Niimg with filename
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones((10, 10, 10)), affine)
    yield assert_equal, utils._repr_niimgs(niimg1), \
        ("%s('%s')" % (niimg1.__class__.__name__, niimg1.get_filename()))
    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    nibabel.save(niimg1, tmpimg1)
    yield assert_equal, utils._repr_niimgs(niimg1), \
        ("%s('%s')" % (niimg1.__class__.__name__, niimg1.get_filename()))


def _remove_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


def test_concat_niimgs():
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones((10, 10, 10)), affine)
    niimg2 = Nifti1Image(np.ones((10, 10, 10)), 2 * affine)

    assert_raises(ValueError, utils.concat_niimgs, [niimg1, niimg2])

    _, tmpimg1 = tempfile.mkstemp(suffix='.nii')
    _, tmpimg2 = tempfile.mkstemp(suffix='.nii')
    try:
        nibabel.save(niimg1, tmpimg1)
        nibabel.save(niimg2, tmpimg2)
        nose.tools.assert_raises(ValueError, utils.concat_niimgs,
                                 [tmpimg1, tmpimg2])
    finally:
        _remove_if_exists(tmpimg1)
        _remove_if_exists(tmpimg2)
