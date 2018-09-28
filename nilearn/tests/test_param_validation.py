"""
Test the _utils.param_validation module
"""

import numpy as np
import warnings
import os
import nibabel

from nose.tools import assert_equal, assert_true, assert_raises
from sklearn.base import BaseEstimator

from nilearn._utils.testing import assert_raises_regex, assert_warns

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (MNI152_BRAIN_VOLUME,
                                             _get_mask_volume,
                                             check_feature_screening,
                                             check_threshold)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")


def test_check_threshold():
    matrix = np.array([[1., 2.],
                       [2., 1.]])

    name = 'threshold'
    # few not correctly formatted strings for 'threshold'
    wrong_thresholds = ['0.1', '10', '10.2.3%', 'asdf%']
    for wrong_threshold in wrong_thresholds:
        assert_raises_regex(ValueError,
                            '{0}.+should be a number followed by '
                            'the percent sign'.format(name),
                            check_threshold,
                            wrong_threshold, matrix,
                            'fast_abs_percentile', name)

    threshold = object()
    assert_raises_regex(TypeError,
                        '{0}.+should be either a number '
                        'or a string'.format(name),
                        check_threshold, threshold, matrix,
                        'fast_abs_percentile', name)

    # Test threshold as int, threshold=2 should return as it is
    # since it is not string
    assert_equal(check_threshold(2, matrix, percentile_func=fast_abs_percentile), 2)

    # check whether raises a warning if given threshold is higher than expected
    assert_warns(UserWarning, check_threshold, 3., matrix,
                 percentile_func=fast_abs_percentile)

    # test with numpy scalar as argument
    threshold = 2.
    threshold_numpy_scalar = np.float64(threshold)
    assert_equal(
        check_threshold(threshold, matrix, percentile_func=fast_abs_percentile),
        check_threshold(threshold_numpy_scalar, matrix,
                        percentile_func=fast_abs_percentile))

    # Test for threshold provided as a percentile of the data (str ending with a
    # %)
    assert_true(1. < check_threshold("50%", matrix,
                                     percentile_func=fast_abs_percentile,
                                     name=name) <= 2.)


def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (mni152_brain_mask))


def test_feature_screening():
    # dummy
    mask_img_data = np.zeros((182, 218, 182))
    mask_img_data[30:-30, 30:-30, 30:-30] = 1
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(mask_img_data, affine=affine)

    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:

            if screening_percentile == 100 or screening_percentile is None:
                assert_equal(check_feature_screening(
                    screening_percentile, mask_img, is_classif), None)
            elif screening_percentile == 101 or screening_percentile == -1:
                assert_raises(ValueError, check_feature_screening,
                              screening_percentile, mask_img, is_classif)
            elif screening_percentile == 20:
                assert_true(isinstance(check_feature_screening(
                    screening_percentile, mask_img, is_classif),
                    BaseEstimator))
