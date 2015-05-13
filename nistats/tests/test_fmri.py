# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import with_statement

import gc
import warnings

import numpy as np

from nipy.modalities.fmri.api import axis0_generator, FmriImageList
from nipy.core.api import parcels, Image, AffineTransform as AfT
from nipy.io.api import  load_image, save_image

from nose.tools import assert_equal, assert_true

from nibabel.tmpdirs import InTemporaryDirectory
from nipy.testing import funcfile


def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_write():
    fname = 'myfile.nii'
    img = load_image(funcfile)
    with InTemporaryDirectory():
        save_image(img, fname)
        test = FmriImageList.from_image(load_image(fname))
        assert_equal(test[0].affine.shape, (4,4))
        assert_equal(img[0].affine.shape, (5,4))
        # Check the affine...
        A = np.identity(4)
        A[:3,:3] = img[:,:,:,0].affine[:3,:3]
        A[:3,-1] = img[:,:,:,0].affine[:3,-1]
        assert_true(np.allclose(test[0].affine, A))
        del test


def test_iter():
    img = load_image(funcfile)
    img_shape = img.shape
    exp_shape = (img_shape[0],) + img_shape[2:]
    j = 0
    for i, d in axis0_generator(img.get_data()):
        j += 1
        assert_equal(d.shape, exp_shape)
        del(i); gc.collect()
    assert_equal(j, img_shape[1])


def test_subcoordmap():
    img = load_image(funcfile)
    subcoordmap = img[3].coordmap
    xform = img.affine[:,1:]
    assert_true(np.allclose(subcoordmap.affine[1:], xform[1:]))
    assert_true(np.allclose(subcoordmap.affine[0], [0,0,0,img.coordmap([3,0,0,0])[0]]))


def test_labels1():
    img = load_image(funcfile)
    data = img.get_data()
    parcelmap = Image(img[0].get_data(), AfT('kji', 'zyx', np.eye(4)))
    parcelmap = (parcelmap.get_data() * 100).astype(np.int32)
    v = 0
    for i, d in axis0_generator(data, parcels(parcelmap)):
        v += d.shape[1]
    assert_equal(v, parcelmap.size)
