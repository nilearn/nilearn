import numpy as np
import pytest

from nilearn._utils.data_gen import generate_fake_fmri


def test_indx_1dto3d():
    sz = np.random.rand(10, 10, 10)
    idx = 45

    x, y, z = clustools.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None


def test_indx_3dto1d():
    sz = np.random.rand(10, 10, 10)
    idx = 45

    x, y, z = clustools.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None


def test_make_local_connectivity_tcorr():
    func_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=5)
    W = clustools.make_local_connectivity_tcorr(func_img, mask_img, thresh=0.50
    assert W is not None


def test_make_local_connectivity_scorr():
    func_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=5)
    W = clustools.make_local_connectivity_scorr(func_img, mask_img, thresh=0.50)
    assert W is not None
