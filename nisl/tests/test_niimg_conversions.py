"""
Test the niimg_conversions

This test file is in nisl/tests because nosetests seems to ignore modules whose
name starts with an underscore
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


def test_check_niimg():
    assert_raises(TypeError, utils.check_niimg, 0)
    # Check that a list of 3D images is valid


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
    shape = (10, 11, 12)
    affine = np.eye(4)
    niimg1 = Nifti1Image(np.ones(shape), affine)
    niimg2 = Nifti1Image(np.ones(shape), 2 * affine)
    niimg3 = Nifti1Image(np.zeros(shape), affine)

    concatenated = utils.concat_niimgs((niimg1, niimg3, niimg1))
    concatenate_true = np.ones(shape + (3,))
    concatenate_true[..., 1] = 0
    np.testing.assert_almost_equal(concatenated.get_data(), concatenate_true)

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
