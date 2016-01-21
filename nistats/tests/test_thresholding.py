""" Test the thresholding utilities
"""
import numpy as np
from scipy.stats import norm
from nose.tools import assert_true
from numpy.testing import assert_almost_equal, assert_equal
import nibabel as nib
from nistats.thresholding import (fdr_threshold, map_threshold)


def test_fdr():
    n = 100
    x = np.linspace(.5 / n, 1. - .5 / n, n)
    x[:10] = .0005
    x = norm.isf(x)
    np.random.shuffle(x)
    assert_almost_equal(fdr_threshold(x, .1), norm.isf(.0005))
    assert_true(fdr_threshold(x, .001) == np.infty)


def test_map_threshold():
    shape = (9, 10, 11)
    p = np.prod(shape)
    data = norm.isf(np.linspace(1. / p, 1. - 1. / p, p)).reshape(shape)
    threshold = .001
    data[2:4, 5:7, 6:8] = 5.
    stat_img = nib.Nifti1Image(data, np.eye(4))
    mask_img = nib.Nifti1Image(np.ones(shape), np.eye(4))

    # test 1
    th_map, _ = map_threshold(
        stat_img, mask_img, threshold, height_control='fpr',
        cluster_threshold=0)
    vals = th_map.get_data()
    assert_equal(np.sum(vals > 0), 8)

    # test 2: excessive cluster forming threshold
    th_map, _ = map_threshold(
        stat_img, mask_img, 100, height_control=None,
        cluster_threshold=0)
    vals = th_map.get_data()
    assert_true(np.sum(vals > 0) == 0)

    # test 3:excessive size threshold
    th_map, z_th = map_threshold(
        stat_img, mask_img, threshold, height_control='fpr',
        cluster_threshold=10)
    vals = th_map.get_data()
    assert_true(np.sum(vals > 0) == 0)
    assert_equal(z_th, norm.isf(.001))

    # test 4: fdr threshold + bonferroni
    for control in ['fdr', 'bonferroni']:
        th_map, _ = map_threshold(
            stat_img, mask_img, .05, height_control=control,
            cluster_threshold=5)
        vals = th_map.get_data()
        assert_equal(np.sum(vals > 0), 8)

    # test 5: direct threshold
    th_map, _ = map_threshold(
        stat_img, mask_img, 4.0, height_control=None,
        cluster_threshold=0)
    vals = th_map.get_data()
    assert_equal(np.sum(vals > 0), 8)

    # test 6: without mask
    th_map, _ = map_threshold(
        stat_img, None, 4.0, height_control=None,
        cluster_threshold=0)
    vals = th_map.get_data()
    assert_equal(np.sum(vals > 0), 8)
