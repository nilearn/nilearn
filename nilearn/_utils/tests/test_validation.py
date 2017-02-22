
from nose.tools import assert_equal, assert_true, assert_raises
import warnings
import os
import nibabel
from sklearn.base import BaseEstimator

from nilearn._utils.testing import assert_warns
from nilearn._utils.param_validation import (MNI152_BRAIN_VOLUME,
                                             _get_mask_volume,
                                             check_feature_screening)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")


def test_feature_screening():
    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:

            if screening_percentile == 100 or screening_percentile is None:
                assert_equal(check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    None)
            elif screening_percentile == 101 or screening_percentile == -1:
                assert_raises(ValueError, check_feature_screening,
                              screening_percentile, MNI152_BRAIN_VOLUME,
                              is_classif)
            elif screening_percentile == 20:
                assert_true(isinstance(check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    BaseEstimator))
            else:
                assert_warns(UserWarning, check_feature_screening,
                             screening_percentile, MNI152_BRAIN_VOLUME * 2,
                             is_classif)



def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (mni152_brain_mask))

